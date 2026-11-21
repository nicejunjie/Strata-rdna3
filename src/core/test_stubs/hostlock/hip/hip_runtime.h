// Host-only stand-in for the HIP runtime, for exit_teardown_test.cpp only (exitfix-20261121).  A "pinned" block is a
// shared anonymous mapping, which /proc/self/maps shows as "/dev/zero (deleted)" rw-s like the runtime's KFD userptr
// blocks, so host_lock_sweep_locked finds and locks it.  The block's size sits in a header page (no table: a table here
// would be one more static the exit could destroy).
#pragma once
#include <cstddef>
#include <sys/mman.h>

typedef int hipError_t;
constexpr hipError_t hipSuccess = 0, hipErrorOutOfMemory = 2;
inline hipError_t hipHostMalloc(void** p, size_t bytes, unsigned int) {
    const size_t total = bytes + 4096;
    void* m = mmap(nullptr, total, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
    if (m == MAP_FAILED) { *p = nullptr; return hipErrorOutOfMemory; }
    *static_cast<size_t*>(m) = total;
    *p = static_cast<char*>(m) + 4096;
    return hipSuccess;
}
inline hipError_t hipFreeHost(void* p) {
    if (p == nullptr) return hipSuccess;
    char* m = static_cast<char*>(p) - 4096;
    munmap(m, *reinterpret_cast<size_t*>(m));
    return hipSuccess;
}
inline hipError_t hipHostFree(void* p) { return hipFreeHost(p); }
