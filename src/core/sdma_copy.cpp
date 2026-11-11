// [claude] see include/strata/core/sdma_copy.hpp
#include "strata/core/sdma_copy.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <vector>

#if defined(STRATA_USE_HIP) && !defined(STRATA_NO_HSA)
#include <hip/hip_runtime.h>
#include <hsa/hsa.h>
#include <hsa/hsa_ext_amd.h>
#endif

namespace strata::core {

#if defined(STRATA_USE_HIP) && !defined(STRATA_NO_HSA)
namespace {
struct AgentInfo { hsa_agent_t agent; uint32_t bdf = 0, domain = 0; bool gpu = false; };
hsa_status_t collect(hsa_agent_t a, void* data) {
    auto* v = static_cast<std::vector<AgentInfo>*>(data);
    hsa_device_type_t t;
    if (hsa_agent_get_info(a, HSA_AGENT_INFO_DEVICE, &t) != HSA_STATUS_SUCCESS) return HSA_STATUS_SUCCESS;
    AgentInfo ai;
    ai.agent = a;
    ai.gpu = t == HSA_DEVICE_TYPE_GPU;
    if (ai.gpu) {
        (void) hsa_agent_get_info(a, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_BDFID, &ai.bdf);
        (void) hsa_agent_get_info(a, (hsa_agent_info_t) HSA_AMD_AGENT_INFO_DOMAIN, &ai.domain);
    }
    v->push_back(ai);
    return HSA_STATUS_SUCCESS;
}
inline hsa_signal_t sig_of(const SdmaCopier::Done& d) { hsa_signal_t s; s.handle = d.handle; return s; }
}  // namespace

SdmaCopier* SdmaCopier::for_device(int dev, std::string& err) {
    static std::mutex mu;
    static std::map<int, std::unique_ptr<SdmaCopier>> made;
    static std::map<int, std::string> failed;
    std::lock_guard<std::mutex> lk(mu);
    if (auto it = made.find(dev); it != made.end()) return it->second.get();
    if (auto it = failed.find(dev); it != failed.end()) { err = it->second; return nullptr; }
    auto fail = [&](const std::string& why) -> SdmaCopier* {
        failed[dev] = why;
        err = why;
        return nullptr;
    };
    static const bool off = [] { const char* v = std::getenv("STRATA_HSA_COPY"); return v != nullptr && v[0] == '0'; }();
    if (off) return fail("STRATA_HSA_COPY=0");
    // the HIP runtime has HSA up already; this adds a reference (never shut down here)
    if (hsa_init() != HSA_STATUS_SUCCESS) return fail("hsa_init failed");
    std::vector<AgentInfo> agents;
    if (hsa_iterate_agents(collect, &agents) != HSA_STATUS_SUCCESS) return fail("hsa_iterate_agents failed");
    char bus[64] = {0};
    if (hipDeviceGetPCIBusId(bus, (int) sizeof(bus), dev) != hipSuccess) return fail("hipDeviceGetPCIBusId failed");
    unsigned dom = 0, b = 0, d = 0, f = 0;
    if (std::sscanf(bus, "%x:%x:%x.%x", &dom, &b, &d, &f) != 4) return fail(std::string("unparsed PCI bus id ") + bus);
    const uint32_t bdf = (b << 8) | (d << 3) | f;
    std::unique_ptr<SdmaCopier> c(new SdmaCopier);
    c->dev_ = dev;
    for (const AgentInfo& ai : agents) {
        if (!ai.gpu && c->cpu_agent_ == 0) c->cpu_agent_ = ai.agent.handle;
        if (ai.gpu && ai.bdf == bdf && (ai.domain == dom || agents.size() < 2) && c->gpu_agent_ == 0) c->gpu_agent_ = ai.agent.handle;
    }
    if (c->gpu_agent_ == 0 || c->cpu_agent_ == 0) return fail(std::string("no HSA agent for ") + bus);
    static const int eng = [] { const char* v = std::getenv("STRATA_REFILL_SDMA_ENGINE"); return v ? std::atoi(v) : -1; }();
    c->engine_ = eng;
    SdmaCopier* r = c.get();
    made[dev] = std::move(c);
    return r;
}

bool SdmaCopier::create(Done& d, int64_t n, std::string& err) {
    hsa_signal_t s;
    if (hsa_signal_create((hsa_signal_value_t) n, 0, nullptr, &s) != HSA_STATUS_SUCCESS) { err = "hsa_signal_create failed"; return false; }
    d.handle = s.handle;
    return true;
}
void SdmaCopier::destroy(Done& d) {
    if (d.handle != 0) (void) hsa_signal_destroy(sig_of(d));
    d.handle = 0;
}
void SdmaCopier::skip(Done& d, int64_t n) {
    if (d.handle != 0 && n > 0) hsa_signal_subtract_screlease(sig_of(d), (hsa_signal_value_t) n);
}
bool SdmaCopier::copy(void* dst_dev, const void* src_host, size_t bytes, Done& d, std::string& err) {
    if (d.handle == 0) { err = "no completion counter"; return false; }
    hsa_agent_t ga, ca;
    ga.handle = gpu_agent_;
    ca.handle = cpu_agent_;
    hsa_status_t st;
    if (engine_ >= 0)
        st = hsa_amd_memory_async_copy_on_engine(dst_dev, ga, src_host, ca, bytes, 0, nullptr, sig_of(d),
                                                 (hsa_amd_sdma_engine_id_t) (1u << engine_), true);
    else
        st = hsa_amd_memory_async_copy(dst_dev, ga, src_host, ca, bytes, 0, nullptr, sig_of(d));
    if (st != HSA_STATUS_SUCCESS) {
        const char* what = nullptr;
        (void) hsa_status_string(st, &what);
        err = std::string("hsa_amd_memory_async_copy: ") + (what != nullptr ? what : "failed");
        return false;
    }
    return true;
}
bool SdmaCopier::landed(const Done& d) const {
    return d.handle == 0 || hsa_signal_load_scacquire(sig_of(d)) <= 0;
}
bool SdmaCopier::wait(const Done& d, int timeout_ms) const {
    if (d.handle == 0) return true;
    // [claude] review M5: bounded (a lost completion hung the engine in the next request's join) and the error state seen
    static const uint64_t freq = [] {
        uint64_t f = 0;
        return hsa_system_get_info(HSA_SYSTEM_INFO_TIMESTAMP_FREQUENCY, &f) == HSA_STATUS_SUCCESS && f > 0 ? f : (uint64_t) 1000000000;
    }();
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        const hsa_signal_value_t v = hsa_signal_wait_scacquire(sig_of(d), HSA_SIGNAL_CONDITION_LT, 1, freq / 100, HSA_WAIT_STATE_BLOCKED);
        if (v == 0) return true;
        if (v < 0) { std::fprintf(stderr, "strata sdma: a copy failed (signal %lld)\n", (long long) v); return false; }
        if (std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count() > timeout_ms) {
            std::fprintf(stderr, "strata sdma: copies not landed after %d ms (%lld left)\n", timeout_ms, (long long) v);
            return false;
        }
    }
}
namespace {
struct Landed { void (*fn)(void*); void* arg; };
bool landed_handler(hsa_signal_value_t, void* p) {
    std::unique_ptr<Landed> l(static_cast<Landed*>(p));
    l->fn(l->arg);
    return false;   // once
}
}  // namespace
bool SdmaCopier::on_landed(Done& d, void (*fn)(void*), void* arg) {
    if (d.handle == 0) return false;
    auto* l = new Landed{fn, arg};
    if (hsa_amd_signal_async_handler(sig_of(d), HSA_SIGNAL_CONDITION_LT, 1, landed_handler, l) != HSA_STATUS_SUCCESS) {
        delete l;
        return false;
    }
    return true;
}
#else
SdmaCopier* SdmaCopier::for_device(int, std::string& err) { err = "not a HIP build"; return nullptr; }
bool SdmaCopier::create(Done&, int64_t, std::string& err) { err = "not a HIP build"; return false; }
void SdmaCopier::destroy(Done& d) { d.handle = 0; }
void SdmaCopier::skip(Done&, int64_t) {}
bool SdmaCopier::copy(void*, const void*, size_t, Done&, std::string& err) { err = "not a HIP build"; return false; }
bool SdmaCopier::landed(const Done&) const { return true; }
bool SdmaCopier::wait(const Done&, int) const { return true; }
bool SdmaCopier::on_landed(Done&, void (*)(void*), void*) { return false; }
#endif

}  // namespace strata::core
