// src/core/exit_teardown_test.cpp - [claude] exitfix-20261121 (council-20261121 "Final design, topic 2", todo (c)): the
// process-lifetime state that detached threads reach (the host memory lock and its sweep, the bounce slots, the aux-CPU
// state) must survive exit(), and the sweep must be quiet once exit has begun.  CPU only, against host-only fakes of the
// runtime (test_stubs/hostlock, test_stubs/bounce).
//
// Build (ASan; the use-after-destroy this guards against is a heap-use-after-free there):
//   c++ -std=c++20 -O1 -g -fsanitize=address -fno-omit-frame-pointer -pthread -DSTRATA_USE_HIP=1
//       -Isrc/core/test_stubs/hostlock -Isrc/core/test_stubs/bounce -Iinclude -Isrc/core
//       src/core/exit_teardown_test.cpp -o exit_teardown_test                 (one line; or -fsanitize=thread)
// Run:   ./exit_teardown_test [runs of the sweep case, default 500]
//
// Each case is a child process (this binary again, with the case's name), killed at a 10 s deadline:
//   sweep   a slow exit handler (200 ms: long enough for a sanitizer report to finish before the process ends), registered
//           before any static is made, so on the unfixed tree it runs after the statics' destructors; the sweep runs
//           every 1 ms (STRATA_TEST_HOST_LOCK_SWEEP_MS=1); main calls exit(0).
//           Expected: exit code 0.  (The unfixed tree: the sweep thread inserts into the destroyed seen map.)
//   worker  three threads keep using the state (host_alloc / host_free, bounce_slot, aux_cpus::place_off_host) and a
//           fourth calls exit(3).  Expected: exit code 3.
//   free    an exit handler registered before any static calls host_free on a locked block (a static destructor's
//           host_free, run after exit began).  Expected: exit code 0, the block forgotten (5 otherwise).
// The parent prints, per case, the runs, the runs that ended otherwise (code / signal / deadline) and how many
// "stays unlocked" lines the children printed (the residual the gate also counts), and exits 1 on any bad run.
#include "strata/hip_compat/host_lock.hpp"
#include "conversation_copy.hpp"

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

namespace HL = strata::hip_compat;
namespace CD = strata::core::conversation_detail;

static void slow_handler() { std::this_thread::sleep_for(std::chrono::milliseconds(200)); }
static void* g_late = nullptr;
static void late_free() {
    const uint64_t before = HL::host_lock_state().bytes;
    (void) HL::host_free(g_late);
    HL::HostLock& s = HL::host_lock_state();
    std::lock_guard<std::mutex> g(s.mu);
    if (s.blocks.count(g_late) != 0 || s.bytes + 65536 != before) _exit(5);
}

static std::vector<void*> make_blocks(int n) {
    HL::HostLock& s = HL::host_lock_state();
    { std::lock_guard<std::mutex> g(s.mu); s.state = 1; }   // lock on whatever the hard limit is (the blocks are small)
    std::vector<void*> v;
    for (int i = 0; i < n; ++i) {
        void* p = nullptr;
        if (HL::host_alloc(&p, 65536, 0) == hipSuccess) v.push_back(p);
    }
    return v;
}

static int child(const std::string& mode) {
    if (mode == "sweep") {
        std::atexit(slow_handler);   // first: before any static (on the unfixed tree it runs after their destructors)
        (void) make_blocks(32);
        HL::host_lock_report("test");
        std::this_thread::sleep_for(std::chrono::microseconds(500 + (getpid() % 2500)));
        std::exit(0);
    }
    if (mode == "worker") {
        std::atexit(slow_handler);
        std::string err;
        (void) strata::aux_cpus::configure("off", err);
        (void) make_blocks(16);
        HL::host_lock_report("test");
        CD::BounceSlot* b = nullptr;
        (void) CD::bounce_slot(b);
        std::thread([&] { for (;;) { void* p = nullptr; if (HL::host_alloc(&p, 65536, 0) == hipSuccess) (void) HL::host_free(p); } }).detach();
        std::thread([&] { for (;;) { CD::BounceSlot* s = nullptr; if (CD::bounce_slot(s) == cudaSuccess) { std::lock_guard<std::mutex> l(s->mu); } } }).detach();
        std::thread([&] { for (;;) (void) strata::aux_cpus::place_off_host(); }).detach();
        std::thread([&] {
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            std::exit(3);   // exit from a worker thread while the others run
        }).detach();
        for (;;) std::this_thread::sleep_for(std::chrono::seconds(1));
    }
    if (mode == "free") {
        std::atexit(late_free);   // first: on the unfixed tree it runs after HostLock's destructor
        std::vector<void*> v = make_blocks(8);
        if (v.empty()) return 6;
        g_late = v.back();
        HL::host_lock_report("test");
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
        std::exit(0);
    }
    return 7;
}

struct Tally { int runs = 0, bad = 0, unlocked = 0; std::string first_bad; };

static void run_case(const char* self, const char* mode, int want, int n, Tally& t, const std::string& dir) {
    for (int i = 0; i < n; ++i) {
        const std::string log = dir + "/" + mode + "-" + std::to_string(i) + ".err";
        const pid_t pid = fork();
        if (pid == 0) {
            const int fd = open(log.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
            if (fd >= 0) { dup2(fd, 2); close(fd); }
            setenv("STRATA_TEST_HOST_LOCK_SWEEP_MS", "1", 1);
            setenv("ASAN_OPTIONS", "exitcode=66:detect_leaks=0:symbolize=0", 0);
            setenv("TSAN_OPTIONS", "exitcode=67:report_thread_leaks=0", 0);
            execl(self, self, mode, (char*) nullptr);
            _exit(127);
        }
        int st = 0; bool done = false;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        while (!done) {
            const pid_t r = waitpid(pid, &st, WNOHANG);
            if (r == pid) { done = true; break; }
            if (std::chrono::steady_clock::now() > deadline) { kill(pid, SIGKILL); waitpid(pid, &st, 0); st = -1; break; }
            std::this_thread::sleep_for(std::chrono::milliseconds(1));
        }
        ++t.runs;
        std::string what;
        if (st == -1) what = "deadline";
        else if (WIFSIGNALED(st)) what = "signal " + std::to_string(WTERMSIG(st));
        else if (WEXITSTATUS(st) != want) what = "exit " + std::to_string(WEXITSTATUS(st));
        if (std::FILE* f = std::fopen(log.c_str(), "r")) {
            char line[1024];
            while (std::fgets(line, sizeof line, f)) if (std::strstr(line, "stays unlocked")) ++t.unlocked;
            std::fclose(f);
        }
        if (!what.empty()) { if (t.bad++ == 0) t.first_bad = what + " (" + log + ")"; }
        else std::remove(log.c_str());
    }
}

int main(int argc, char** argv) {
    if (argc >= 2 && (!std::strcmp(argv[1], "sweep") || !std::strcmp(argv[1], "worker") || !std::strcmp(argv[1], "free")))
        return child(argv[1]);
    const int n_sweep = argc >= 2 ? std::atoi(argv[1]) : 500;
    char dirt[] = "/tmp/exit_teardown_XXXXXX";
    const char* d = getenv("EXIT_TEST_DIR");
    const std::string dir = d ? d : (mkdtemp(dirt) ? dirt : "/tmp");
    struct Case { const char* mode; int want, n; } cases[] = {{"sweep", 0, n_sweep}, {"worker", 3, 100}, {"free", 0, 100}};
    int bad = 0;
    for (const Case& c : cases) {
        Tally t;
        run_case("/proc/self/exe", c.mode, c.want, c.n, t, dir);
        std::printf("exit_teardown_test %-6s: %d runs, %d bad%s%s, %d \"stays unlocked\" lines\n", c.mode, t.runs, t.bad,
                    t.bad ? ", first: " : "", t.first_bad.c_str(), t.unlocked);
        bad += t.bad;
    }
    std::printf("exit_teardown_test: %s\n", bad ? "FAILED" : "passed");
    return bad ? 1 : 0;
}
