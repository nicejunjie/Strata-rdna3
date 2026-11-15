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
// - host_lock_report() prints the one load line (blocks, GiB locked, failures, the kernel's VmLck, the limit).
// No environment variable and no flag.
#include <hip/hip_runtime.h>
#include <hip/hip_version.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <unordered_map>
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

/// The one load line (fixed form, for the watcher):
/// "strata: host memory lock: <n> blocks, <x.xx> GiB locked, <f> failures (<y.yy> GiB unlocked), VmLck <z.zz> GiB, limit <l> (<when>)"
inline void host_lock_report(const char* when) {
#if !defined(_WIN32)
    HostLock& s = host_lock_state();
    std::lock_guard<std::mutex> g(s.mu);
    host_lock_decide(s);
    if (s.state != 1) return;   // the "nothing locked" line was printed when it was decided
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
                 "strata: host memory lock: %zu blocks, %.2f GiB locked, %d failures (%.2f GiB unlocked), VmLck %.2f GiB, limit %s (%s)\n",
                 s.blocks.size(), s.bytes / 1073741824.0, s.fails, s.fail_bytes / 1073741824.0, vmlck_kib / 1048576.0, lim, when);
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
