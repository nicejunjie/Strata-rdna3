// include/strata/core/request_switches.hpp - [claude] A/B inside one engine, the cheap part: a few host-side STRATA_*
// switches that a request may override for itself, so one engine load serves both arms of an experiment.
//
// A request names them on its GEN/GENI line as `env=NAME:VALUE,NAME:VALUE` (the server's "strata_tune": {"env": {...}});
// every GEN/GENI starts from the start-up values (the environment) and applies its own list, so a request without the key
// runs exactly as the engine was started.  Only switches that are read on the host, per request or per window, and are
// baked into no captured graph, allocation, stream or thread are listed (see docs/FAST_START.md for the
// audit of the others).  All of them are read and written on the serve loop's thread; the atomics only keep a stray reader
// from tearing a value.  BGEN batch slots do not parse the key (they keep the last GEN's values).
#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace strata::core::reqsw {

enum Id : int {
    REFILL_PACE,     ///< STRATA_REFILL_PACE=<MiB per window>: bytes (MiB << 20)
    REFILL_RATE,     ///< STRATA_REFILL_RATE=<MiB/s>: bytes per second
    REFILL_DEMAND,   ///< STRATA_REFILL_DEMAND=1: 1 / 0
    REFILL_DEFER,    ///< STRATA_REFILL_DEFER=<k windows>
    REFILL_JOIN,     ///< STRATA_REFILL_JOIN=landed: 1 / 0 (due)
    // [claude] planner step 1b: a request's forced plan (calibration arms in one engine load).  REQUEST-ONLY: never read
    // from the environment at start-up (-1 = not forced: the engine's rule), so they add no start-up variable.
    PLAN_STAGED_MAX, ///< plan staged_max=<tokens>: chunks of fewer tokens are staged, the others stream every expert
    PLAN_WALK_MAX,   ///< plan walk_max=<tokens>: the routed walk's limit (one-chunk parts, layer split)
    PLAN_TP_MIN,     ///< plan tp_min=<tokens>: the TP read's lower bound
    PLAN_TP_MAX,     ///< plan tp_max=<tokens>: the TP read's upper bound
    PLAN_DP_MIN,     ///< plan dp_min=<tokens>: the DP read's lower bound (the shape floor still applies)
    PLAN_TAIL_MAX,   ///< plan tail_max=<tokens>: STRATA_PREFILL_EQUAL's short-tail exception limit
    PLAN_PROBE,      ///< [claude] planner rev 5 shadow: 1 = this request is option C's canned start-up read (logged, never acted on)
    kCount
};

inline const char* name(Id id) {
    static const char* const n[kCount] = {"STRATA_REFILL_PACE", "STRATA_REFILL_RATE", "STRATA_REFILL_DEMAND",
                                          "STRATA_REFILL_DEFER", "STRATA_REFILL_JOIN",
                                          "STRATA_PLAN_STAGED_MAX", "STRATA_PLAN_WALK_MAX", "STRATA_PLAN_TP_MIN",
                                          "STRATA_PLAN_TP_MAX", "STRATA_PLAN_DP_MIN", "STRATA_PLAN_TAIL_MAX",
                                          "STRATA_PLAN_PROBE"};
    return n[id];
}

/// [claude] planner step 1b: switches that exist per request only (the start-up environment is not read for them)
inline bool request_only(Id id) { return id >= PLAN_STAGED_MAX; }

/// the value a setting means, exactly as each site parsed its getenv before (nullptr: unset)
inline double parse(Id id, const char* v) {
    switch (id) {
        case REFILL_PACE: return v ? (double) ((int64_t) std::max(0, std::atoi(v)) << 20) : 0.0;
        case REFILL_RATE: return v ? std::max(0.0, std::atof(v)) * 1048576.0 : 0.0;
        case REFILL_DEMAND: return v != nullptr && std::atoi(v) != 0 ? 1.0 : 0.0;
        case REFILL_DEFER: return v ? (double) std::max(0, std::atoi(v)) : 0.0;
        case REFILL_JOIN: return v != nullptr && std::strcmp(v, "landed") == 0 ? 1.0 : 0.0;
        case PLAN_STAGED_MAX: case PLAN_WALK_MAX: case PLAN_TP_MIN: case PLAN_TP_MAX: case PLAN_DP_MIN: case PLAN_TAIL_MAX: case PLAN_PROBE:
            return v != nullptr && *v ? (double) std::max<long long>(0, std::atoll(v)) : -1.0;
        default: return 0.0;
    }
}

struct State {
    double startup[kCount];
    std::atomic<double> cur[kCount];
    State() {
        for (int i = 0; i < kCount; ++i) {
            startup[i] = request_only((Id) i) ? parse((Id) i, nullptr) : parse((Id) i, std::getenv(name((Id) i)));
            cur[i].store(startup[i], std::memory_order_relaxed);
        }
    }
};
inline State& state() {
    static State s;
    return s;
}

/// the value for the request running now
inline double get(Id id) { return state().cur[id].load(std::memory_order_relaxed); }
inline int64_t get_i64(Id id) { return (int64_t) get(id); }
inline bool get_bool(Id id) { return get(id) != 0.0; }

/// A GEN/GENI's `env=` list ("" = none): every listed switch takes its value, the rest their start-up values.  `said`: a
/// line for the log when the request runs with anything other than the start-up values, or names a switch that is not
/// runtime-switchable (ignored).
inline void apply_request(const std::string& spec, std::string& said) {
    State& s = state();
    double want[kCount];
    for (int i = 0; i < kCount; ++i) want[i] = s.startup[i];
    std::string ignored;
    size_t at = 0;
    while (at < spec.size()) {
        size_t end = spec.find(',', at);
        if (end == std::string::npos) end = spec.size();
        const std::string item = spec.substr(at, end - at);
        at = end + 1;
        const size_t colon = item.find(':');
        if (item.empty() || colon == std::string::npos) continue;
        std::string key = item.substr(0, colon);
        const std::string val = item.substr(colon + 1);
        if (key.compare(0, 7, "STRATA_") != 0) key = "STRATA_" + key;
        int hit = -1;
        for (int i = 0; i < kCount; ++i)
            if (key == name((Id) i)) hit = i;
        if (hit < 0) { ignored += (ignored.empty() ? "" : ", ") + key; continue; }
        want[hit] = parse((Id) hit, val.c_str());
    }
    std::string diff;
    for (int i = 0; i < kCount; ++i) {
        s.cur[i].store(want[i], std::memory_order_relaxed);
        if (want[i] != s.startup[i]) {
            char b[96];
            std::snprintf(b, sizeof b, "%s%s=%g", diff.empty() ? "" : " ", name((Id) i), want[i]);
            diff += b;
        }
    }
    said.clear();
    if (!diff.empty()) said = "this request's switches: " + diff;
    if (!ignored.empty()) said += (said.empty() ? "" : "; ") + std::string("not runtime-switchable (ignored): ") + ignored;
}

}  // namespace strata::core::reqsw
