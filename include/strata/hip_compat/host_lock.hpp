#pragma once
// Host memory lock (Linux; results/stall-20261115, Fable review-fable-memlock Q1).  Included by the HIP shim
// (cuda_runtime.h), whose cudaHostAlloc / cudaMallocHost / cudaFreeHost go through host_alloc / malloc_host / host_free.
//
// Every block made here is GPU-mapped host memory (a KFD userptr BO).  When the kernel's reclaim pages one out, the KFD
// stops every queue of the process until the whole block is faulted back, and under swap pressure that restore can
// loop for seconds (cp0; the pressure test's 12.5 s stop).  So each block is mlock'ed as it is made, and reclaim can
// no longer invalidate it.
// - The expert arena is NOT locked: it is mmap'ed and registered (pinned.cu), not made here, and it is 72 GiB.
// - At the first lock the engine raises its own soft RLIMIT_MEMLOCK to the hard limit.  If the hard limit is under
//   1 GiB (the 8 MiB default), nothing is locked and one line says so.
// - A failed mlock never fails an allocation: the block stays unlocked and is counted (the first 8 get one line each).
// - The runtime's own GPU-mapped blocks (never made here) are locked by host_lock_sweep_locked at READY and every 10 s.
// - host_lock_report() prints the one load line (blocks, GiB locked, runtime GiB, failures, the kernel's VmLck, the limit).
// No environment variable and no flag.
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <utility>
#if !defined(_WIN32)
#include <sys/mman.h>
#include <sys/resource.h>
#endif

namespace strata::hip_compat {

struct HostLock {
    std::mutex mu;
    std::unordered_map<void*, size_t> blocks;   // locked block -> bytes (forgotten on free; the unmap drops the lock)
    uint64_t bytes = 0;
    uint64_t fail_bytes = 0;
    int fails = 0;
    int state = -1;                             // -1 not decided, 0 off, 1 on
    unsigned long long hard = 0;
};
inline HostLock& host_lock_state() { static HostLock s; return s; }

inline void host_lock_decide(HostLock& s) {   // under s.mu
    if (s.state >= 0) return;
#if !defined(_WIN32)
    rlimit rl{};
    if (getrlimit(RLIMIT_MEMLOCK, &rl) != 0) { s.state = 0; return; }
    s.hard = rl.rlim_max == RLIM_INFINITY ? ~0ull : (unsigned long long) rl.rlim_max;
    if (s.hard < (1ull << 30)) {
        s.state = 0;
        std::fprintf(stderr, "strata: host memory lock: nothing locked (the RLIMIT_MEMLOCK hard limit is %llu KiB, under 1 GiB)\n",
                     s.hard >> 10);
        return;
    }
    if (rl.rlim_cur != rl.rlim_max) { rl.rlim_cur = rl.rlim_max; (void) setrlimit(RLIMIT_MEMLOCK, &rl); }
    s.state = 1;
#else
    s.state = 0;
#endif
}

inline void host_lock(void* p, size_t bytes) {
#if !defined(_WIN32)
    if (p == nullptr || bytes == 0) return;
    HostLock& s = host_lock_state();
    std::lock_guard<std::mutex> g(s.mu);
    host_lock_decide(s);
    if (s.state != 1) return;
    if (mlock(p, bytes) != 0) {
        const int e = errno;
        if (s.fails++ < 8)
            std::fprintf(stderr, "strata: host memory lock: a %zu KiB block stays unlocked (mlock: %s)\n", bytes >> 10,
                         std::strerror(e));
        s.fail_bytes += bytes;
        return;
    }
    s.blocks[p] = bytes;
    s.bytes += bytes;
#else
    (void) p; (void) bytes;
#endif
}

inline void host_lock_forget(void* p) {
    if (p == nullptr) return;
    HostLock& s = host_lock_state();
    std::lock_guard<std::mutex> g(s.mu);
    auto it = s.blocks.find(p);
    if (it == s.blocks.end()) return;
    s.bytes -= it->second;
    s.blocks.erase(it);
}

/// The HIP runtime makes GPU-mapped host blocks of its own (staging buffers, signal and kernarg pools: 1-16 MiB,
/// about 600 MiB on a 2-GPU engine), which never pass through host_alloc.  They are the same KFD userptr BOs (a shared
/// anonymous mapping, "/dev/zero (deleted)" in /proc/self/maps) and the same stop source (G2 run B-1 of 86de97914: a
/// 3.6 s stop from kcompactd migrating 16 / 4 / 1 MiB unlocked ones).  host_lock_sweep() mlocks every such mapping of
/// 4 GiB or less (an arena on a shared mapping stays out), once per mapping.  It runs at
/// READY and then from a background thread every 10 s (silent unless a lock fails).  Under s.mu.
inline void host_lock_sweep_locked(HostLock& s, std::unordered_map<uintptr_t, std::pair<size_t, unsigned long long>>& seen, uint64_t& swept_bytes) {
#if !defined(_WIN32)
    if (s.state != 1) return;
    std::FILE* f = std::fopen("/proc/self/maps", "r");
    if (f == nullptr) return;
    char line[512];
    while (std::fgets(line, sizeof line, f) != nullptr) {
        unsigned long long a = 0, b = 0, off = 0, inode = 0; char perm[8] = {0}, dev[16] = {0};
        if (std::sscanf(line, "%llx-%llx %7s %llx %15s %llu", &a, &b, perm, &off, dev, &inode) != 6) continue;
        if (std::strcmp(perm, "rw-s") != 0 || std::strstr(line, "/dev/zero (deleted)") == nullptr) continue;
        const size_t bytes = (size_t) (b - a);
        if (bytes == 0 || bytes > ((size_t) 4 << 30)) continue;
        // Seen = the same start, size and shmem inode (a block freed and made again at the same address gets a new
        // inode), so a mapping is locked once; the sweep then costs one /proc/self/maps read and no mlock.
        const std::pair<size_t, unsigned long long> id(bytes, inode);
        auto it = seen.find((uintptr_t) a);
        if (it != seen.end() && it->second == id) continue;
        if (mlock((void*) (uintptr_t) a, bytes) != 0) {
            const int e = errno;
            if (s.fails++ < 8)
                std::fprintf(stderr, "strata: host memory lock: a %zu KiB runtime block stays unlocked (mlock: %s)\n",
                             bytes >> 10, std::strerror(e));
            s.fail_bytes += bytes;
            seen[(uintptr_t) a] = id;   // not retried every sweep
            continue;
        }
        bool ours = false;   // a block host_alloc made (and locked) starts inside this mapping
        for (auto& kv : s.blocks)
            if ((uintptr_t) kv.first >= (uintptr_t) a && (uintptr_t) kv.first < (uintptr_t) b) { ours = true; break; }
        seen[(uintptr_t) a] = id;
        if (!ours) swept_bytes += bytes;
    }
    std::fclose(f);
#else
    (void) s; (void) seen; (void) swept_bytes;
#endif
}
struct HostLockSweep { std::unordered_map<uintptr_t, std::pair<size_t, unsigned long long>> seen; uint64_t bytes = 0; bool thread = false; };
inline HostLockSweep& host_lock_sweep_state() { static HostLockSweep w; return w; }

/// The one load line (fixed form, for the watcher):
/// "strata: host memory lock: <n> blocks, <x.xx> GiB locked, <r.rr> GiB runtime blocks, <f> failures (<y.yy> GiB unlocked),
///  VmLck <z.zz> GiB, limit <l> (<when>)"
/// It sweeps the runtime's blocks first, and starts the 10 s background sweep the first time (a detached thread).
inline void host_lock_report(const char* when) {
#if !defined(_WIN32)
    HostLock& s = host_lock_state();
    HostLockSweep& w = host_lock_sweep_state();
    std::lock_guard<std::mutex> g(s.mu);
    host_lock_decide(s);
    if (s.state != 1) return;   // the "nothing locked" line was printed when it was decided
    host_lock_sweep_locked(s, w.seen, w.bytes);
    unsigned long long vmlck_kib = 0;
    if (std::FILE* f = std::fopen("/proc/self/status", "r")) {
        char line[256];
        while (std::fgets(line, sizeof line, f) != nullptr)
            if (std::strncmp(line, "VmLck:", 6) == 0) { vmlck_kib = std::strtoull(line + 6, nullptr, 10); break; }
        std::fclose(f);
    }
    char lim[32];
    if (s.hard == ~0ull) std::snprintf(lim, sizeof lim, "unlimited");
    else std::snprintf(lim, sizeof lim, "%.2f GiB", s.hard / 1073741824.0);
    std::fprintf(stderr,
                 "strata: host memory lock: %zu blocks, %.2f GiB locked, %.2f GiB runtime blocks, %d failures (%.2f GiB unlocked), "
                 "VmLck %.2f GiB, limit %s (%s)\n",
                 s.blocks.size(), s.bytes / 1073741824.0, w.bytes / 1073741824.0, s.fails, s.fail_bytes / 1073741824.0,
                 vmlck_kib / 1048576.0, lim, when);
    if (!w.thread) {
        w.thread = true;
        std::thread([] {
            for (;;) {
                std::this_thread::sleep_for(std::chrono::seconds(10));
                HostLock& s2 = host_lock_state();
                HostLockSweep& w2 = host_lock_sweep_state();
                std::lock_guard<std::mutex> g2(s2.mu);
                host_lock_sweep_locked(s2, w2.seen, w2.bytes);
            }
        }).detach();
    }
#else
    (void) when;
#endif
}

template <typename T>
inline hipError_t host_alloc(T** p, size_t bytes, unsigned int flags) {
    const hipError_t e = hipHostMalloc(reinterpret_cast<void**>(p), bytes, flags);
    if (e == hipSuccess) host_lock(reinterpret_cast<void*>(*p), bytes);
    return e;
}

inline hipError_t host_free(void* p) {
    host_lock_forget(p);
#if HIP_VERSION_MAJOR < 7
    return hipHostFree(p);
#else
    return hipFreeHost(p);
#endif
}

}  // namespace strata::hip_compat
