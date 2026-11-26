// crashfix-20261117: the HIP bounce path of the conversation copies (conversation_copy.hpp), host only.
// 1. The runtime is never handed the pageable side of a save or restore - only the engine's pinned bounce buffers
//    and the device side - so ROCm never locks (and keeps) a userptr over memory the conversation cache later frees.
// 2. The bytes arrive whole for every size around the 4 MiB chunk, in both directions, with copies deferred until an
//    event or the stream is synchronized (a bounce buffer reused too early corrupts the result).
// 3. A copy between states (HostSide::none) is handed through unchanged.
// Build: c++ -std=c++20 -DSTRATA_USE_HIP=1 -I src/core/test_stubs/bounce -I include src/core/conversation_copy_bounce_test.cpp
#include "conversation_copy.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>

using strata::core::conversation_detail::HostSide;
using strata::core::conversation_detail::copy_nonblocking;
using strata::core::conversation_detail::bounce_chunk;

namespace {
int failures = 0, checks = 0;
void check(bool ok, const char* what, size_t n) {
    ++checks;
    if (!ok) { ++failures; std::fprintf(stderr, "FAIL: %s (%zu bytes)\n", what, n); }
}
bool handed_inside(const std::vector<uint8_t>& v) {
    for (const auto& [q, n] : fake::handed()) {
        const auto* a = static_cast<const uint8_t*>(q);
        if (a < v.data() + v.size() && a + n > v.data()) return true;
    }
    return false;
}
}  // namespace

int main() {
    std::mt19937_64 rng(20261117);
    const size_t C = bounce_chunk;
    for (size_t n : {size_t(1), size_t(4095), C - 1, C, C + 1, 2 * C, 2 * C + 7, 3 * C + C / 2, size_t(25) << 20,
                     size_t(37) * 1000 * 1000}) {
        std::vector<uint8_t> device(n), host(n), back(n);
        for (auto& b : device) b = (uint8_t) rng();
        // save: device -> pageable host
        fake::handed().clear();
        check(copy_nonblocking(host.data(), device.data(), n, false, HostSide::dst) == cudaSuccess, "save status", n);
        check(fake::queue().empty(), "save complete on return", n);
        check(host == device, "save bytes", n);
        check(!handed_inside(host), "save never hands the runtime the pageable target", n);
        // restore: pageable host -> device (queued: still complete on return, the staging is reused)
        fake::handed().clear();
        std::fill(back.begin(), back.end(), 0);
        check(copy_nonblocking(back.data(), host.data(), n, true, HostSide::src) == cudaSuccess, "restore status", n);
        check(fake::queue().empty(), "restore complete on return", n);
        check(back == device, "restore bytes", n);
        check(!handed_inside(host), "restore never hands the runtime the pageable source", n);
    }
    {   // between states: the runtime gets both pointers as before
        std::vector<uint8_t> a(3 * C, 7), b(3 * C, 0);
        fake::handed().clear();
        check(copy_nonblocking(b.data(), a.data(), a.size(), false, HostSide::none) == cudaSuccess, "state copy", a.size());
        check(b == a && handed_inside(a) && handed_inside(b), "state copy handed through", a.size());
    }
    std::printf("conversation_copy_bounce_test: %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
