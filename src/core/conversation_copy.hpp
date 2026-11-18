#pragma once

#include "strata/core/conversation_cache.hpp"
#include "strata/core/progress.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::core::conversation_detail {

/// Conversation snapshot / checkpoint copies go through a private non-blocking stream of the current device
/// instead of the legacy (null) stream.  A plain cudaMemcpy is ordered against every blocking stream of the
/// device, and while another thread captures a graph on one (a drafter or a stage worker), HIP refuses it with
/// "operation would make the legacy stream depend on a capturing blocking stream".  The caller's contract is
/// unchanged: the device was synchronized before the copy (the copy itself is complete on return).
/// private_stream: this thread's stream on the current device, created on first use.
inline cudaError_t private_stream(cudaStream_t& s) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    thread_local std::vector<cudaStream_t> streams;
    if (dev < 0) dev = 0;
    if ((size_t) dev >= streams.size()) streams.resize((size_t) dev + 1, nullptr);   // never destroyed: process lifetime
    if (!streams[(size_t) dev]) {
        e = cudaStreamCreateWithFlags(&streams[(size_t) dev], cudaStreamNonBlocking);
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
inline constexpr size_t bounce_chunk = (size_t) 4 << 20;
struct BounceSlot {
    std::mutex mu;   // one copy at a time per device (stage threads save their parts on their own devices)
    void* buf[2] = {nullptr, nullptr};
    cudaEvent_t ev[2] = {nullptr, nullptr};
};
/// the current device's slot, its buffers made on first use; never freed (process lifetime, like the streams)
inline cudaError_t bounce_slot(BounceSlot*& out) {
    int dev = 0;
    cudaError_t e = cudaGetDevice(&dev);
    if (e != cudaSuccess) return e;
    if (dev < 0) dev = 0;
    static std::mutex mu;
    static std::vector<std::unique_ptr<BounceSlot>> slots;
    {
        std::lock_guard<std::mutex> lk(mu);
        if ((size_t) dev >= slots.size()) slots.resize((size_t) dev + 1);
        if (!slots[(size_t) dev]) slots[(size_t) dev] = std::make_unique<BounceSlot>();
        out = slots[(size_t) dev].get();
    }
    return cudaSuccess;
}
inline cudaError_t bounce_ready(BounceSlot& b) {   // under b.mu
    for (int i = 0; i < 2; ++i) {
        if (!b.buf[i]) {
            const cudaError_t e = cudaHostAlloc(&b.buf[i], bounce_chunk, cudaHostAllocPortable);
            if (e != cudaSuccess) { b.buf[i] = nullptr; return e; }
        }
        if (!b.ev[i]) {
            const cudaError_t e = cudaEventCreateWithFlags(&b.ev[i], cudaEventDisableTiming);
            if (e != cudaSuccess) { b.ev[i] = nullptr; return e; }
        }
    }
    return cudaSuccess;
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
        // chunk k lands in buf[k & 1] while the CPU moves chunk k - 1 out of the other one
        for (size_t k = 0; k <= chunks; ++k) {
            if (k < chunks) {
                if ((e = cudaMemcpyAsync(b.buf[k & 1], h + k * bounce_chunk, len(k), cudaMemcpyDefault, s)) != cudaSuccess ||
                    (e = cudaEventRecord(b.ev[k & 1], s)) != cudaSuccess) {
                    (void) cudaStreamSynchronize(s);   // nothing of ours stays in flight
                    return e;
                }
            }
            if (k > 0) {
                const size_t j = k - 1;
                if ((e = cudaEventSynchronize(b.ev[j & 1])) != cudaSuccess) { (void) cudaStreamSynchronize(s); return e; }
                std::memcpy(d + j * bounce_chunk, b.buf[j & 1], len(j));
                core::progress_beat();
            }
        }
        return cudaSuccess;
    }
    // HostSide::src: the CPU fills buf[k & 1] once the copy that read it (chunk k - 2) is done
    for (size_t k = 0; k < chunks; ++k) {
        if (k >= 2 && (e = cudaEventSynchronize(b.ev[k & 1])) != cudaSuccess) { (void) cudaStreamSynchronize(s); return e; }
        std::memcpy(b.buf[k & 1], h + k * bounce_chunk, len(k));
        if ((e = cudaMemcpyAsync(d + k * bounce_chunk, b.buf[k & 1], len(k), cudaMemcpyDefault, s)) != cudaSuccess ||
            (e = cudaEventRecord(b.ev[k & 1], s)) != cudaSuccess) {
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
    o.gdn = copy_beating(c.gdn);
    o.ple = copy_beating(c.ple);
    o.tails = copy_beating(c.tails);
    o.dead = copy_beating(c.dead);
    o.block_pos = copy_beating(c.block_pos);
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
