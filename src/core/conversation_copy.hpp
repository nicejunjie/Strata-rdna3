#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/progress.hpp"
#include "strata/platform/aux_cpus.hpp"
#include "strata/core/trace_clock.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <condition_variable>
#include <deque>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

namespace strata::core::conversation_detail {

#if defined(__GLIBCXX__)
// the function-local mutexes below are never destroyed in effect: their destructor does nothing (exitfix-20261121)
static_assert(std::is_trivially_destructible<std::mutex>::value, "a static std::mutex here must survive exit");
#endif

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
/// STRATA_TRACE (exitfix-20261121 (d)): where a checkpoint restore's time goes on this thread - the one-time setup
/// (the bounce buffers' allocation, the bounce events and this thread's private stream) against the copies.  Summed only
/// while tracing; conversation_checkpoint_restore resets and prints it.
struct CopySplit { double alloc_ms = 0.0, create_ms = 0.0; };
inline bool copy_trace_on() { static const bool on = std::getenv("STRATA_TRACE") != nullptr; return on; }
inline CopySplit& copy_split() { thread_local CopySplit c; return c; }
inline double ms_since_(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}
/// private_stream: this thread's stream on the current device, created on first use.
inline cudaError_t private_stream(cudaStream_t& s) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaStream_t> streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);   // never destroyed: process lifetime
    if (!streams[(size_t) dev]) {
        const auto t0 = std::chrono::steady_clock::now();
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
        if (copy_trace_on()) copy_split().create_ms += ms_since_(t0);
        if (e != cudaSuccess) { streams[(size_t) dev] = nullptr; return e; }
    }
    s = streams[(size_t) dev];
    return cudaSuccess;
}

/// Which side of a copy is ordinary pageable host memory owned by the conversation code (a checkpoint's vectors, a
/// ConversationBuffer segment, a session file block): `dst` for a save, `src` for a restore, `none` between states.
enum class HostSide { none, dst, src };

#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
/// crashfix-20261117 (results/planner-20261112/phase3/followup/crash/analysis.md): on ROCm a copy to or from pageable
/// memory of more than GPU_PINNED_MIN_XFER_SIZE (1 MiB) is done by locking the pages in place (a KFD userptr) and
/// ROCclr keeps the last few such locks per stream after the copy.  The conversation cache frees its vectors and
/// segments (drop superseded, evict) and glibc gives the pages back to the kernel (heap trim / munmap): a lock that
/// is still kept then covers unmapped pages, the KFD leaves its GPU mapping invalid, and a later copy that reuses the
/// lock for a new allocation at the same address faults ("Memory access fault ... Page not present", 2026-11-17
/// 22:17, conversation_checkpoint_save).  So the conversation copies never hand ROCm a pageable pointer: they go
/// through two engine-owned pinned 4 MiB buffers per device (one copy in flight while the CPU moves the other).
#ifndef STRATA_CONV_BOUNCE_MIB   // build-time only (the chunk trial of review-fable-crash condition 10); no environment variable
#define STRATA_CONV_BOUNCE_MIB 4
#endif
inline constexpr size_t bounce_chunk = (size_t) STRATA_CONV_BOUNCE_MIB << 20;
#ifndef STRATA_CONV_BOUNCE_BUFS   // build-time only, like the chunk (condition 10: the third buffer)
#define STRATA_CONV_BOUNCE_BUFS 2
#endif
inline constexpr size_t bounce_bufs = STRATA_CONV_BOUNCE_BUFS;
static_assert(bounce_bufs >= 2, "the bounce pipeline needs two buffers at least");
struct BounceSlot {
    std::mutex mu;   // one copy at a time per device (stage threads save their parts on their own devices)
    void* buf[bounce_bufs] = {};
    cudaEvent_t ev[bounce_bufs] = {};
};
/// the current device's slot, its buffers made on first use; never freed (process lifetime, like the streams).  The table
/// and the slots are immortal (exitfix-20261121): a static vector of unique_ptr freed every slot at exit while a stage
/// thread could still be inside copy_bounced with one.
inline cudaError_t bounce_slot(BounceSlot*& out) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    if (dev < 0) dev = 0;
    static std::mutex mu;   // (trivially destructible on libstdc++: asserted at the top of this namespace)
    static std::vector<BounceSlot*>& slots = *new std::vector<BounceSlot*>();
    {
        std::lock_guard<std::mutex> lk(mu);
        if ((size_t) dev >= slots.size()) slots.resize((size_t) dev + 1, nullptr);
        if (!slots[(size_t) dev]) slots[(size_t) dev] = new BounceSlot();
        out = slots[(size_t) dev];
    }
    return cudaSuccess;
}
inline cudaError_t bounce_ready(BounceSlot& b) {   // under b.mu
    for (size_t i = 0; i < bounce_bufs; ++i) {
        if (!b.buf[i]) {
            const auto t0 = std::chrono::steady_clock::now();
            const cudaError_t e = cudaHostAlloc(&b.buf[i], bounce_chunk, cudaHostAllocPortable);
            if (copy_trace_on()) copy_split().alloc_ms += ms_since_(t0);
            if (e != cudaSuccess) { b.buf[i] = nullptr; return e; }
        }
        if (!b.ev[i]) {
            const auto t0 = std::chrono::steady_clock::now();
            const cudaError_t e = cudaEventCreateWithFlags(&b.ev[i], cudaEventDisableTiming);
            if (copy_trace_on()) copy_split().create_ms += ms_since_(t0);
            if (e != cudaSuccess) { b.ev[i] = nullptr; return e; }
        }
    }
    return cudaSuccess;
}
/// (d) at load: the current device's bounce buffers and events, made now instead of at the first restore
inline cudaError_t bounce_prepare() {
    BounceSlot* slot = nullptr;
    cudaError_t e = bounce_slot(slot);
    if (e != cudaSuccess) return e;
    std::lock_guard<std::mutex> lk(slot->mu);
    return bounce_ready(*slot);
}
/// The copy through the slot's pinned buffers on stream `s`, complete on return.
inline cudaError_t copy_bounced(void* dst, const void* src, size_t bytes, HostSide side, cudaStream_t s) {
    BounceSlot* slot = nullptr;
    cudaError_t e = bounce_slot(slot);
    if (e != cudaSuccess) return e;
    std::lock_guard<std::mutex> lk(slot->mu);
    if ((e = bounce_ready(*slot)) != cudaSuccess) return e;
    BounceSlot& b = *slot;
    const size_t chunks = (bytes + bounce_chunk - 1) / bounce_chunk;
    auto len = [&](size_t k) { return std::min(bounce_chunk, bytes - k * bounce_chunk); };
    auto* d = static_cast<uint8_t*>(dst);
    auto* h = static_cast<const uint8_t*>(src);
    if (side == HostSide::dst) {
        // up to bounce_bufs - 1 chunks in flight: chunk k lands in buf[k % N] while the CPU moves chunk k - (N - 1) out;
        // buf[k % N] last held chunk k - N, moved out at step k - 1
        constexpr size_t N = bounce_bufs;
        for (size_t k = 0; k < chunks + N - 1; ++k) {
            if (k < chunks) {
                if ((e = cudaMemcpyAsync(b.buf[k % N], h + k * bounce_chunk, len(k), cudaMemcpyDefault, s)) != cudaSuccess ||
                    (e = cudaEventRecord(b.ev[k % N], s)) != cudaSuccess) {
                    (void) cudaStreamSynchronize(s);   // nothing of ours stays in flight
                    return e;
                }
            }
            if (k >= N - 1) {
                const size_t j = k - (N - 1);
                if ((e = cudaEventSynchronize(b.ev[j % N])) != cudaSuccess) { (void) cudaStreamSynchronize(s); return e; }
                std::memcpy(d + j * bounce_chunk, b.buf[j % N], len(j));
                core::progress_beat();
            }
        }
        return cudaSuccess;
    }
    // HostSide::src: the CPU fills buf[k % N] once the copy that read it (chunk k - N) is done
    constexpr size_t N = bounce_bufs;
    for (size_t k = 0; k < chunks; ++k) {
        if (k >= N && (e = cudaEventSynchronize(b.ev[k % N])) != cudaSuccess) { (void) cudaStreamSynchronize(s); return e; }
        std::memcpy(b.buf[k % N], h + k * bounce_chunk, len(k));
        if ((e = cudaMemcpyAsync(d + k * bounce_chunk, b.buf[k % N], len(k), cudaMemcpyDefault, s)) != cudaSuccess ||
            (e = cudaEventRecord(b.ev[k % N], s)) != cudaSuccess) {
            (void) cudaStreamSynchronize(s);
            return e;
        }
        core::progress_beat();
    }
    return cudaStreamSynchronize(s);
}
#endif

/// `queued`: the copy is only queued on the private stream, and the caller synchronizes the device (or orders a
/// later launch after it with order_after_queued) before anything reads the target or writes the source.
/// `side`: the pageable host side, if any; on HIP such a copy goes through the pinned bounce buffers above and is
/// complete on return even when `queued` (the staging buffers are reused by the next copy).
inline cudaError_t copy_nonblocking(void* dst, const void* src, size_t bytes, bool queued = false,
                                    HostSide side = HostSide::none) {
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
    if (side != HostSide::none && bytes) return copy_bounced(dst, src, bytes, side, s);
#else
    (void) side;
#endif
    e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDefault, s);
    if (e != cudaSuccess || queued) return e;
    e = cudaStreamSynchronize(s);
    core::progress_beat();   // #2049: a parked conversation is many of these; each one that lands is progress
    return e;
}

#if defined(STRATA_USE_HIP) || defined(STRATA_HIP_GFX906)
/// O (crashfix-20261117, Fable "Ruling on the suite short-prompt delta"): a checkpoint save copies the running state straight
/// into one of two engine-owned pinned staging areas of its device (a direct DMA: the runtime's own allocation, no userptr),
/// returns, and a persistent helper thread of that device, off the host's CPU, copies the area into the checkpoint's blocks
/// (marked pending: every byte reader waits, checkpoint_block.hpp) and frees the area.  No free area: the save waits for one;
/// an area that cannot be allocated: the caller uses the bounce path.  STRATA_TEST_CKPT_HELPER_DELAY_MS=N (tests only): the
/// helper sleeps N ms before each job.
struct CkptPart { CheckpointBlock::Store* store; size_t off, bytes; };
struct CkptHelper {
    std::mutex mu;
    std::condition_variable work, freed;
    void* area[2] = {nullptr, nullptr};
    size_t cap[2] = {0, 0};
    bool busy[2] = {false, false};
    struct Job { int slot; std::vector<CkptPart> parts; bool held = true; unsigned id = 0; double submit_ms = 0.0; };   // P1: held until ckpt_release_held(); (d trace) id, submit stamp
    std::deque<Job> q;
    bool started = false;
    int dev = -1;            // (d trace) the device, set by ckpt_helper
    unsigned next_job = 0;   // (d trace) under mu: the job counter of this helper (1, 2, ...)
};
inline long test_ckpt_helper_delay_ms() {
    static const long v = [] { const char* e = std::getenv("STRATA_TEST_CKPT_HELPER_DELAY_MS"); return e ? std::atol(e) : 0L; }();
    return v;
}
inline std::mutex& ckpt_helpers_mu() { static std::mutex mu; return mu; }
inline std::vector<CkptHelper*>& ckpt_helpers() { static auto* v = new std::vector<CkptHelper*>(); return *v; }
/// P1: every held copy on every device may run now (the serve loop after the last late join; a reader at a pending latch; a save
/// that finds both areas busy; the request-scope guard)
inline void ckpt_release_held() {
    std::vector<CkptHelper*> hs;
    { std::lock_guard<std::mutex> lk(ckpt_helpers_mu()); hs = ckpt_helpers(); }
    for (CkptHelper* h : hs) {
        if (!h) continue;
        bool any = false;
        { std::lock_guard<std::mutex> lk(h->mu); for (auto& j : h->q) if (j.held) { j.held = false; any = true; } }
        if (any) h->work.notify_all();
    }
}
/// the device's helper (never destroyed: process lifetime, like the private streams and the bounce slots)
inline CkptHelper& ckpt_helper(int dev) {
    std::lock_guard<std::mutex> lk(ckpt_helpers_mu());
    auto* helpers = &ckpt_helpers();
    if (dev < 0) dev = 0;
    if ((size_t) dev >= helpers->size()) helpers->resize((size_t) dev + 1, nullptr);
    if (!(*helpers)[(size_t) dev]) { (*helpers)[(size_t) dev] = new CkptHelper(); (*helpers)[(size_t) dev]->dev = dev; }
    return *(*helpers)[(size_t) dev];
}
/// STRATA_TEST_CKPT_STAGING_FAIL=1 (tests only): no staging area can be allocated (the bounce path takes the save)
inline bool test_ckpt_staging_fail() {
    static const bool refuse = [] { const char* e = std::getenv("STRATA_TEST_CKPT_STAGING_FAIL"); return e && e[0] == '1'; }();
    return refuse;
}
/// A free staging area of at least `bytes` on device `dev`, waiting for one when both are busy; -1 when none can be allocated.
/// (d): area 0 is normally made at load at its exact size (ckpt_prepare); this grow branch stays as the fallback.
inline int ckpt_acquire(CkptHelper& h, size_t bytes) {
    if (test_ckpt_staging_fail()) return -1;
    std::unique_lock<std::mutex> lk(h.mu);
    if (h.busy[0] && h.busy[1]) {   // both areas taken by held copies: let them run (P1), then wait for one
        lk.unlock(); ckpt_release_held(); lk.lock();
    }
    // STRATA_TRACE (d trace): a save that waits for a staging area (both held by jobs the helper has not finished) prints
    // the wait on the shared trace clock; only when it really waits
    const bool wait_tr = trace_on() && h.busy[0] && h.busy[1];
    const double aw0 = wait_tr ? trace_ms() : 0.0;
    h.freed.wait(lk, [&] { return !h.busy[0] || !h.busy[1]; });
    if (wait_tr) {
        const double aw1 = trace_ms();
        std::fprintf(stderr, "strata trace: ckpt area wait dev %d: begin @%.2f ms end @%.2f ms (%.2f ms)\n", h.dev, aw0, aw1, aw1 - aw0);
    }
    const int slot = !h.busy[0] ? 0 : 1;
    if (h.cap[slot] < bytes) {
        if (h.area[slot]) { (void) cudaFreeHost(h.area[slot]); h.area[slot] = nullptr; h.cap[slot] = 0; }
        void* a = nullptr;
        if (cudaHostAlloc(&a, bytes, cudaHostAllocPortable) != cudaSuccess || !a) { (void) cudaGetLastError(); return -1; }
        h.area[slot] = a; h.cap[slot] = bytes;
    }
    h.busy[slot] = true;
    return slot;
}
inline void ckpt_release(CkptHelper& h, int slot) {
    { std::lock_guard<std::mutex> lk(h.mu); h.busy[slot] = false; }
    h.freed.notify_all();
}
inline void ckpt_start_locked(CkptHelper& h) {   // under h.mu: the device's helper thread, once
    ckpt_release_hook().store(&ckpt_release_held, std::memory_order_release);
    if (!h.started) {
        h.started = true;
        std::thread([&h] {
            (void) strata::aux_cpus::place_off_host();   // never on the host's core: the decode runs there
            for (;;) {
                CkptHelper::Job job;
                {
                    std::unique_lock<std::mutex> l2(h.mu);
                    h.work.wait(l2, [&] { return !h.q.empty() && !h.q.front().held; });
                    job = std::move(h.q.front());
                    h.q.pop_front();
                }
                // STRATA_TRACE (d trace): one line per job on the shared trace clock (trace_clock.hpp) - dequeue, first copy
                // start, copy end, first / last latch open; no sync, only clock reads
                const bool tr = trace_on();
                const double deq = tr ? trace_ms() : 0.0;
                if (const long ms = test_ckpt_helper_delay_ms(); ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(ms));
                const auto* a = static_cast<const uint8_t*>(h.area[job.slot]);
                double c0 = -1.0, c1 = 0.0, o0 = -1.0, o1 = 0.0;
                size_t bytes = 0;
                const void* first_block = job.parts.empty() ? nullptr : (const void*) job.parts.front().store;
                for (const CkptPart& p : job.parts) {
                    if (tr && c0 < 0.0) c0 = trace_ms();
                    if (p.bytes) std::memcpy(p.store->bytes.data(), a + p.off, p.bytes);
                    bytes += p.bytes;
                    if (tr) c1 = trace_ms();
                    p.store->open();   // the block holds the bytes now; its readers go on
                    if (tr) { o1 = trace_ms(); if (o0 < 0.0) o0 = o1; }
                }
                if (tr)
                    std::fprintf(stderr, "strata trace: ckpt copy dev %d job %u block %p: %zu part(s), %.2f MB, submit @%.2f ms, "
                                         "dequeue @%.2f ms, first copy @%.2f ms, copy end @%.2f ms, latch open @%.2f-@%.2f ms\n",
                                 h.dev, job.id, first_block, job.parts.size(), bytes / 1e6, job.submit_ms, deq, c0, c1, o0, o1);
                ckpt_release(h, job.slot);
            }
        }).detach();
    }
}
inline void ckpt_submit(CkptHelper& h, int slot, std::vector<CkptPart> parts) {
    std::lock_guard<std::mutex> lk(h.mu);
    const unsigned id = ++h.next_job;
    for (const CkptPart& p : parts)
        if (p.store) { p.store->dev.store(h.dev, std::memory_order_relaxed); p.store->job.store(id, std::memory_order_relaxed); }
    h.q.push_back({slot, std::move(parts), true, id, trace_on() ? trace_ms() : 0.0});
    ckpt_start_locked(h);
    h.work.notify_one();
}
/// (d, exitfix-20261121) at load: staging area 0 of this helper at exactly `bytes` (what every save of the session asks
/// for) and the helper thread, both made now instead of at the first save; area 1 stays lazy (made only when a save
/// finds area 0 busy).  False when area 0 cannot be allocated (the save's grow branch then tries again, as before).
inline bool ckpt_prepare(CkptHelper& h, size_t bytes) {
    std::lock_guard<std::mutex> lk(h.mu);
    bool ok = true;
    if (bytes && !test_ckpt_staging_fail() && !h.busy[0] && h.cap[0] < bytes) {
        if (h.area[0]) { (void) cudaFreeHost(h.area[0]); h.area[0] = nullptr; h.cap[0] = 0; }
        void* a = nullptr;
        if (cudaHostAlloc(&a, bytes, cudaHostAllocPortable) != cudaSuccess || !a) { (void) cudaGetLastError(); ok = false; }
        else { h.area[0] = a; h.cap[0] = bytes; }
    }
    ckpt_start_locked(h);
    return ok;
}
#endif

/// #2049 #2065 #1971: a host-to-host copy that tells the watchdog it is moving.  Parked state (the running
/// checkpoints, the K/V segments) sits in ordinary pageable memory for the whole session; when the OS has written
/// it to the pagefile, copying it back runs at the drive's page-in speed (measured in the reports: ~190 hard faults
/// a second, under 1 MB/s) and one memcpy of it stays silent past the 60 s limit, so the watchdog ended an engine
/// that was only slow.  One beat per 1 MiB slice keeps a copy that moves alive and still lets a real stall show.
/// STRATA_TEST_SLOW_COPY_US=N (tests only): every 1 MiB slice of a checkpoint copy takes N microseconds more, as when
/// the OS has paged the checkpoints out and each slice is read back from the drive.  Unset: no delay.
inline long test_slow_copy_us() {
    static const long v = [] { const char* e = std::getenv("STRATA_TEST_SLOW_COPY_US"); return e ? std::atol(e) : 0L; }();
    return v;
}

template<class T>
inline std::vector<T> copy_beating(const std::vector<T>& src) {
    std::vector<T> out;
    out.reserve(src.size());
    const size_t step = std::max<size_t>(1, ((size_t) 1 << 20) / sizeof(T));
    for (size_t at = 0; at < src.size(); at += step) {
        out.insert(out.end(), src.begin() + (std::ptrdiff_t) at,
                   src.begin() + (std::ptrdiff_t) std::min(src.size(), at + step));
        if (const long us = test_slow_copy_us(); us > 0) std::this_thread::sleep_for(std::chrono::microseconds(us));
        core::progress_beat();
    }
    return out;
}

inline ConversationCheckpoint copy_checkpoint_beating(const ConversationCheckpoint& c) {
    ConversationCheckpoint o;
    o.ids = copy_beating(c.ids);
    o.imgs = copy_beating(c.imgs);
    // S (crashfix-20261117): the running-state blocks are shared (copy-on-write), not deep-copied
    o.gdn = c.gdn; o.ple = c.ple; o.tails = c.tails; o.dead = c.dead; o.block_pos = c.block_pos;
    core::progress_beat();
    o.used = c.used;
    o.pinned = c.pinned;
    o.stage_parts.reserve(c.stage_parts.size());
    for (const auto& part : c.stage_parts) o.stage_parts.push_back(copy_checkpoint_beating(part));
    return o;
}

inline std::vector<ConversationCheckpoint> copy_checkpoints_beating(const std::vector<ConversationCheckpoint>& cs) {
    std::vector<ConversationCheckpoint> out;
    out.reserve(cs.size());
    for (const auto& c : cs) out.push_back(copy_checkpoint_beating(c));
    return out;
}

/// Work queued on the legacy stream after this call starts after every copy this thread queued on the current
/// device's private stream (an event the legacy stream waits for; no host wait).
inline cudaError_t order_after_queued() {
    cudaStream_t s = nullptr;
    cudaError_t e = private_stream(s);
    if (e != cudaSuccess) return e;
    int dev = 0;
    e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaEvent_t> events;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= events.size()) events.resize((size_t) dev + 1, nullptr);    // never destroyed: process lifetime
    if (!events[(size_t) dev]) {
        e = cudaEventCreateWithFlags(&events[(size_t) dev], cudaEventDisableTiming);
        if (e != cudaSuccess) { events[(size_t) dev] = nullptr; return e; }
    }
    e = cudaEventRecord(events[(size_t) dev], s);
    if (e != cudaSuccess) return e;
    return cudaStreamWaitEvent(nullptr, events[(size_t) dev], 0);
}

} // namespace strata::core::conversation_detail
