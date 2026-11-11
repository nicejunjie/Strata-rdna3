#pragma once
// [claude] Host-to-device copies issued to a card's SDMA engine through the HSA runtime directly.
//
// On ROCm, hipMemcpyAsync from host memory that the GPU can address (hipHostRegister'ed with the mapped flag: the
// expert arena) is not always a DMA: the runtime may run it as a copy KERNEL, and such a kernel shares the CUs with
// whatever decode kernel runs beside it.  A long prompt's refill of the cache slots it borrowed (~14 GB a card, ~1 s)
// ran beside the first decode windows that way, and every one of those windows' kernels on that card ran 3-4x slower
// (2x RX 7900 XTX, Qwen3.8-Flash IQ3_S, 128K prompt: the first eight windows 85 ms instead of 20).  hsa_amd_memory_async_copy
// goes to the SDMA engine, which leaves the CUs alone.  HIP builds only; elsewhere for_device returns null.
#include <cstddef>
#include <cstdint>
#include <string>

namespace strata::core {

class SdmaCopier {
public:
    /// the copier of a HIP device (made on first use; null when the build has no HSA or the agent was not found - `err` says)
    static SdmaCopier* for_device(int dev, std::string& err);

    /// a completion counter `n` copies decrement; landed() once every one did
    struct Done { uint64_t handle = 0; };
    bool create(Done& d, int64_t n, std::string& err);
    void destroy(Done& d);
    /// `n` copies that will not come (a copy that was issued another way): the counter no longer waits for them
    void skip(Done& d, int64_t n);

    /// an asynchronous copy of `bytes` from pinned host memory the HSA runtime knows (hipHostRegister'ed / hipHostMalloc'ed)
    /// into device memory, decrementing `d` when it lands.  false: not issued (the caller copies another way).
    bool copy(void* dst_dev, const void* src_host, size_t bytes, Done& d, std::string& err);
    /// landed: no copy still in flight (a failed copy - the signal went negative - counts as landed; `wait` reports it)
    bool landed(const Done& d) const;
    /// [claude] review M5: blocks until `d` lands; false when a copy failed (negative signal) or `timeout_ms` passed
    bool wait(const Done& d, int timeout_ms = 30000) const;
    /// `fn(arg)` once `d` lands (the runtime's async thread; at most once per call; the Done must stay alive until then).
    /// false: not registered (the caller polls `landed` instead).
    bool on_landed(Done& d, void (*fn)(void*), void* arg);

    /// the engine the copies are asked for (STRATA_REFILL_SDMA_ENGINE=<n>; -1: the runtime's choice)
    int engine() const { return engine_; }
    int device() const { return dev_; }

private:
    SdmaCopier() = default;
    int dev_ = -1;
    int engine_ = -1;
    uint64_t gpu_agent_ = 0, cpu_agent_ = 0;
};

}  // namespace strata::core
