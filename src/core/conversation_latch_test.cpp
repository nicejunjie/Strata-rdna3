// crashfix-20261117 design O: the pending-block latch and the checkpoint helper, host only (fake runtime), meant for TSan.
// A save fills a staging area and hands the blocks to the device's helper; readers (data, ==, unique, mutable_data,
// assign_for_overwrite, a copy then a read), a share without waiting, and the last reference dropped while pending must all
// see the final bytes or wait, never torn ones; three saves in a row reuse the two areas.  (d): the load-time setup.
// Build: c++ -std=c++20 -fsanitize=thread -g -DSTRATA_USE_HIP=1 -I src/core/test_stubs/bounce -I include -I src/core src/core/conversation_latch_test.cpp
#include "conversation_copy.hpp"
#include <atomic>
#include <chrono>
#include <cstdio>
#include <thread>
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
}  // namespace
int main() {
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
