// include/strata/core/trace_clock.hpp - [claude] exitfix-20261121 (d) trace: the one clock of the STRATA_TRACE "@<ms>" stamps.
// generate.cpp's per-request trace (tr(): "strata trace: <step> a b @X ms", the "part next" / "part done" stamps the plan
// analysis reads) and the checkpoint copy / latch lines (conversation_copy.hpp, checkpoint_block.hpp, the restore line) all
// stamp milliseconds since the same origin: steady_clock at the first trace_ms() call in the process (in practice the
// request trace's first step, as before), so copies can be aligned with decode windows in one engine.log.
// Trace off: nothing here is called on any path (every caller checks trace_on() first).
#pragma once

#include <atomic>
#include <chrono>
#include <cstdlib>

namespace strata::core {

inline bool trace_on() { static const bool on = std::getenv("STRATA_TRACE") != nullptr; return on; }
/// the origin: fixed by the first call (thread-safe static init)
inline std::chrono::steady_clock::time_point trace_t0() { static const auto t0 = std::chrono::steady_clock::now(); return t0; }
/// ms of `t` (default: now) since the origin
inline double trace_ms(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(t - trace_t0()).count();
}
inline double trace_ms() { const auto t0 = trace_t0(); return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count(); }

} // namespace strata::core
