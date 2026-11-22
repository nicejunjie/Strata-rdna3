// crashfix-20261117 design O: the pending-block latch and the checkpoint helper, host only (fake runtime), meant for TSan.
// A save fills a staging area and hands the blocks to the device's helper; readers (data, ==, unique, mutable_data,
// assign_for_overwrite, a copy then a read), a share without waiting, and the last reference dropped while pending must all
// see the final bytes or wait, never torn ones; three saves in a row reuse the two areas.  (d): the load-time setup.
// (d trace): with STRATA_TRACE the helper's per-job line, a reader's latch-wait line and a save's area-wait line appear, on
// one clock and in order; without it none of them (a child process of this binary each way, its stderr read back).
// Build: c++ -std=c++20 -fsanitize=thread -g -DSTRATA_USE_HIP=1 -I src/core/test_stubs/bounce -I include -I src/core src/core/conversation_latch_test.cpp
#include "conversation_copy.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <string>
#include <thread>
#include <unistd.h>
#include <vector>
using namespace strata::core;
using namespace strata::core::conversation_detail;
namespace {
int checks = 0; std::atomic<int> failures{0};
void check(bool ok, const char* what) { ++checks; if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s\n", what); } }
// one "save": fill the staging area with `v`, mark the blocks pending, submit
void save(CkptHelper& h, CheckpointBlock& a, CheckpointBlock& b, size_t n, uint8_t v) {
    const int slot = ckpt_acquire(h, 2 * n);
    std::memset(h.area[slot], v, n); std::memset(static_cast<uint8_t*>(h.area[slot]) + n, uint8_t(v + 1), n);
    std::vector<CkptPart> parts{{a.begin_pending(n), 0, n}, {b.begin_pending(n), n, n}};
    ckpt_submit(h, slot, std::move(parts));
}
bool all(const CheckpointBlock& b, uint8_t v) { const uint8_t* p = b.data(); for (size_t i = 0; i < b.size(); ++i) if (p[i] != v) return false; return true; }
// (d trace) the child: one save read at once (the reader blocks on the latch: the helper sleeps 20 ms), then three saves with
// no read (the third waits for an area)
int trace_child() {
    setenv("STRATA_TEST_CKPT_HELPER_DELAY_MS", "20", 1);
    CkptHelper& h = ckpt_helper(2);
    const size_t n = 1 << 16;
    CheckpointBlock a, b;
    save(h, a, b, n, 5);
    const bool ok1 = all(a, 5) && all(b, 6);
    CheckpointBlock x[3], y[3];
    for (int k = 0; k < 3; ++k) save(h, x[k], y[k], n, uint8_t(k + 10));
    bool ok2 = true;
    for (int k = 0; k < 3; ++k) ok2 = ok2 && all(x[k], uint8_t(k + 10)) && all(y[k], uint8_t(k + 11));
    return ok1 && ok2 ? 0 : 1;
}
std::string run_child(const char* self, bool trace, int& rc) {
    char tmpl[] = "/tmp/latch_trace_XXXXXX";
    const char* td = std::getenv("TMPDIR");
    std::string path = td ? std::string(td) + "/latch_trace_XXXXXX" : std::string(tmpl);
    std::vector<char> buf(path.begin(), path.end()); buf.push_back('\0');
    const int fd = mkstemp(buf.data());
    if (fd >= 0) close(fd);
    const std::string out(buf.data());
    const std::string cmd = std::string(trace ? "STRATA_TRACE=1 " : "env -u STRATA_TRACE ") + "'" + self + "' trace-child 2> '" + out + "'";
    rc = std::system(cmd.c_str());
    std::ifstream f(out); std::stringstream ss; ss << f.rdbuf();
    std::remove(out.c_str());
    return ss.str();
}
void trace_lines(const char* self) {
    int rc = -1;
    const std::string on = run_child(self, true, rc);
    check(rc == 0, "trace child (trace on) ran clean");
    double sub = 0, deq = 0, c0 = 0, c1 = 0, o0 = 0, o1 = 0, w0 = 0, w1 = 0, wd = 0;
    unsigned job = 0; size_t parts = 0; double mb = 0; void* blk = nullptr; int dev = -1;
    const size_t pc = on.find("strata trace: ckpt copy dev 2 job 1 ");
    check(pc != std::string::npos && std::sscanf(on.c_str() + pc, "strata trace: ckpt copy dev %d job %u block %p: %zu part(s), %lf MB, submit @%lf ms, "
                                                 "dequeue @%lf ms, first copy @%lf ms, copy end @%lf ms, latch open @%lf-@%lf ms",
                                                 &dev, &job, &blk, &parts, &mb, &sub, &deq, &c0, &c1, &o0, &o1) == 11,
          "trace on: the helper's line for job 1 (device, job, block, parts, stamps)");
    // each part is copied, then its latch opened: the first open follows the first copy, the last open the last copy end
    check(parts == 2 && sub <= deq && deq <= c0 && c0 <= c1 && c0 <= o0 && o0 <= o1 && c1 <= o1, "trace on: the helper's stamps in order");
    check(deq - sub >= 0.0 && c0 - deq >= 19.0, "trace on: the copy starts after the helper's 20 ms delay");
    const size_t pw = on.find("strata trace: ckpt latch wait dev 2 job 1 ");
    check(pw != std::string::npos && std::sscanf(on.c_str() + pw, "strata trace: ckpt latch wait dev %d job %u block %p: begin @%lf ms end @%lf ms (%lf ms)",
                                                 &dev, &job, &blk, &w0, &w1, &wd) == 6,
          "trace on: the reader's latch wait line for job 1");
    check(w0 <= o0 + 1e-6 && w1 >= o0 - 1e-6, "trace on: the reader waits across the latch open (one clock)");
    check(on.find("strata trace: ckpt area wait dev 2: begin @") != std::string::npos, "trace on: the third save's area wait line");
    const std::string off = run_child(self, false, rc);
    check(rc == 0, "trace child (trace off) ran clean");
    check(off.find("strata trace:") == std::string::npos, "trace off: no trace line at all");
}
}  // namespace
int main(int argc, char** argv) {
    if (argc >= 2 && std::string(argv[1]) == "trace-child") return trace_child();
    {   // (d trace) first, in children: this process's trace switch is read once
        char self[4096] = {0};
        const ssize_t k = readlink("/proc/self/exe", self, sizeof(self) - 1);
        if (k > 0) trace_lines(self); else check(false, "readlink /proc/self/exe");
    }
    setenv("STRATA_TEST_CKPT_HELPER_DELAY_MS", "20", 1);
    CkptHelper& h = ckpt_helper(0);
    const size_t n = 1 << 20;
    {   // (d, exitfix-20261121) the load-time setup: area 0 at its exact size and the helper thread now, area 1 lazy, the
        // bounce buffers and events made once; the saves after it use what it made (no regrow)
        CkptHelper& h1 = ckpt_helper(1);
        check(!h1.started && h1.cap[0] == 0, "a fresh helper has nothing yet");
        check(ckpt_prepare(h1, 2 * n), "prepare");
        void* const a0 = h1.area[0];
        check(a0 != nullptr && h1.cap[0] == 2 * n && h1.cap[1] == 0 && h1.area[1] == nullptr, "prepare: area 0 exact, area 1 lazy");
        check(h1.started && !h1.busy[0] && !h1.busy[1], "prepare: the helper thread is started, no area taken");
        check(ckpt_prepare(h1, 2 * n) && h1.area[0] == a0, "prepare twice: nothing made again");
        CheckpointBlock a, b;
        save(h1, a, b, n, 41);
        check(h1.area[0] == a0 && h1.cap[0] == 2 * n && h1.cap[1] == 0, "the first save takes the prepared area 0 as it is");
        CheckpointBlock c, d;
        save(h1, c, d, n, 43);   // area 0 is still held (P1) or copying: the second save makes area 1, as before
        check(h1.area[0] == a0 && h1.cap[1] == 2 * n, "a second save while area 0 is busy makes area 1 (lazy)");
        check(all(a, 41) && all(b, 42) && all(c, 43) && all(d, 44), "the saves after prepare carry their bytes");
        BounceSlot* bs = nullptr;
        check(bounce_prepare() == cudaSuccess && bounce_slot(bs) == cudaSuccess, "bounce prepare");
        bool made = true;
        for (size_t i = 0; i < bounce_bufs; ++i) made = made && bs->buf[i] != nullptr && bs->ev[i] != nullptr;
        check(made, "bounce prepare: every buffer and event made");
        void* const b0 = bs->buf[0];
        check(bounce_prepare() == cudaSuccess && bs->buf[0] == b0, "bounce prepare twice: nothing made again");
    }
    for (int round = 0; round < 20; ++round) {
        CheckpointBlock a, b;
        save(h, a, b, n, uint8_t(round));
        CheckpointBlock shared = a;                                     // the park: shares, does not wait
        check(shared.size() == n, "the size is known while pending");
        std::vector<std::thread> readers;
        readers.emplace_back([&] { check(all(a, uint8_t(round)), "data() waits for the bytes"); });
        readers.emplace_back([&] { check(shared == a, "== waits"); });
        readers.emplace_back([&] { check(all(b, uint8_t(round + 1)), "the second block"); });
        for (auto& t : readers) t.join();
        check(!a.unique(), "unique() after the wait: the park still shares");
        shared.clear();
        check(a.unique(), "unique() once the share is gone (the helper holds no reference)");
        // the pool's reuse: save again into the same (unique, same-size) blocks
        save(h, a, b, n, uint8_t(round + 50));
        check(all(a, uint8_t(round + 50)) && all(b, uint8_t(round + 51)), "a save into the reused blocks");
        // a rewrite through a pending block waits, then copies on write
        save(h, a, b, n, uint8_t(round + 90));
        CheckpointBlock keep = a;
        a.mutable_data()[0] = 7;
        check(keep[0] == uint8_t(round + 90) && a[0] == 7, "mutable_data() waits, then copies on write");
        // the last reference dropped while pending: the Store waits in its destructor
        { CheckpointBlock c, d; save(h, c, d, n, 3); }
        // three saves in a row on one device (two areas): the third waits for an area
        CheckpointBlock x[3], y[3];
        for (int k = 0; k < 3; ++k) save(h, x[k], y[k], n, uint8_t(k + 10));
        for (int k = 0; k < 3; ++k) check(all(x[k], uint8_t(k + 10)) && all(y[k], uint8_t(k + 11)), "three saves in a row");
    }
    // P1: a held copy runs only when released (or when a reader reaches its latch)
    setenv("STRATA_TEST_CKPT_HELPER_DELAY_MS", "0", 1);
    {
        CheckpointBlock a, b; save(h, a, b, n, 200);
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        check(a.pending() && b.pending(), "P1: the copy is held until released");
        ckpt_release_held();
        for (int i = 0; i < 200 && (a.pending() || b.pending()); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(5));
        check(!a.pending() && all(a, 200) && all(b, 201), "P1: released, the copy runs");
        CheckpointBlock c, d; save(h, c, d, n, 210);
        check(all(c, 210), "P1: a reader at a held latch releases it and gets the bytes");
        CheckpointBlock x[3], y[3];
        for (int k = 0; k < 3; ++k) save(h, x[k], y[k], n, uint8_t(220 + k));   // the third finds both areas held: releases them
        for (int k = 0; k < 3; ++k) check(all(x[k], uint8_t(220 + k)), "P1: three saves with no release");
        { CheckpointBlock e, f; save(h, e, f, n, 230); }                          // dropped while held: the destructor releases
        check(true, "P1: the last reference dropped while held does not hang");
    }
    std::printf("conversation_latch_test: %d checks, %d failures\n", checks, failures.load());
    return failures ? 1 : 0;
}
