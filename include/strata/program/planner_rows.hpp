// include/strata/program/planner_rows.hpp - [claude] phase 3 (results/planner-20261112/phase3, review-fable-phase3 sections 1
// and 5, "Deploy A"): the two-card threshold rows of the --planner table, read by the engine at start-up.
//
// A row is one line `row2 STRATA_<KEY> <integer>` in the --planner file; every other line (the one-card planner's own lines,
// `#` comments) is ignored here, so the same file serves both engines.  The fill follows the gfx1100 arch defaults'
// rule (setenv(key, value, 0)): an explicit environment value wins, else the table row, else the engine's own default at
// the key's parse site.  The rows are applied before the option loop and before any parse site runs, so every site reads
// exactly the string the config's environment gave it before: a table holding today's values is behaviour-identical by
// construction.  Only the threshold keys below are accepted (switches and anything else: a refused table, one line, and
// the engine runs on its environment alone).
#pragma once

#include <cctype>
#include <cerrno>
#include <cstdlib>
#include <functional>
#include <string>
#include <utility>
#include <vector>

namespace strata::program::planner_rows {

/// the threshold keys a row may set (classes A, B and C of review-fable-phase3 section 5, and the Thresholds fields
/// segment_plan.hpp parses next to them)
inline const std::vector<std::string>& allowed_keys() {
    static const std::vector<std::string> k = {
        "STRATA_TP_PREFILL_MIN", "STRATA_TP_PREFILL_MAX", "STRATA_TP_PREFILL_CHUNK", "STRATA_DP_MIN", "STRATA_DP_PIPE_CHUNK",
        "STRATA_PIPE_HALF_MIN", "STRATA_PIPE_HALF_MAX", "STRATA_PREFILL_EQUAL_TAIL_MAX", "STRATA_PREFILL_AUTO_MAX",
        "STRATA_TP_DEAL_BUDGET", "STRATA_FUSED_ROUTED_MAX", "STRATA_REFILL_HEAD", "STRATA_REFILL_HEAD_MIN", "STRATA_REFILL_HEAD2",
        "STRATA_REFILL_HEAD2_MIN", "STRATA_REFILL_HEAD_BIG", "STRATA_REFILL_HEAD_BIG_CHUNK", "STRATA_REFILL_HEAD_TP_KEEP",
        "STRATA_REFILL_HEAD_TP_MIN"};
    return k;
}

struct Row { std::string key, value; };

/// parse the rows of a table's text; false + err on a malformed or refused row, a duplicate key or a non-integer value
inline bool parse(const std::string& text, std::vector<Row>& rows, std::string& err) {
    rows.clear();
    size_t at = 0; int lineno = 0;
    while (at <= text.size()) {
        size_t nl = text.find('\n', at); if (nl == std::string::npos) nl = text.size();
        std::string s = text.substr(at, nl - at); at = nl + 1; ++lineno;
        if (const size_t h = s.find('#'); h != std::string::npos) s.resize(h);   // a comment, whole-line or trailing
        std::vector<std::string> w; size_t i = 0;
        while (i < s.size()) {
            while (i < s.size() && std::isspace((unsigned char) s[i])) ++i;
            size_t j = i; while (j < s.size() && !std::isspace((unsigned char) s[j])) ++j;
            if (j > i) w.push_back(s.substr(i, j - i));
            i = j;
        }
        if (w.empty() || w[0] != "row2") { if (nl == text.size()) break; continue; }
        const std::string where = "line " + std::to_string(lineno);
        if (w.size() != 3) { err = where + ": a row is `row2 STRATA_<KEY> <integer>`"; return false; }
        bool known = false;
        for (const auto& k : allowed_keys()) known = known || k == w[1];
        if (!known) { err = where + ": " + w[1] + " is not a threshold key a row may set"; return false; }
        const std::string& v = w[2];
        size_t p = (v[0] == '-') ? 1 : 0;
        if (p >= v.size()) { err = where + ": " + w[1] + " has no integer value"; return false; }
        for (; p < v.size(); ++p) if (!std::isdigit((unsigned char) v[p])) { err = where + ": " + w[1] + " value " + v + " is not an integer"; return false; }
        for (const auto& r : rows) if (r.key == w[1]) { err = where + ": " + w[1] + " appears twice"; return false; }
        rows.push_back({w[1], v});
        if (nl == text.size()) break;
    }
    return true;
}

struct Applied { std::string key, value, source; };   // source: "env" (an explicit value kept), "table" or "derived"

/// fill: env, else the row (set with overwrite 0).  get / set are std::getenv / setenv in the engine, fakes in the test.
/// (e, B2 review-fable-structural decision 1) the two derived chunk rows, after the rows are applied (--planner only):
///   STRATA_TP_PREFILL_CHUNK unset => ceil256(STRATA_TP_PREFILL_MAX), when the table sets TP_PREFILL_MAX;
///   STRATA_DP_PIPE_CHUNK unset    => ceil1024(STRATA_DP_MIN),        when the table sets DP_MIN.
/// "Unset" = neither the environment nor a row gave it (an explicit value still wins); the base is the value in force
/// after the fill (the environment's, if it overrode the row).  Without a table nothing is derived, so the parse-site
/// defaults (16384, 8192) are unchanged.  A base that is not a positive integer derives nothing.  Both rules give
/// today's rows exactly (IQ4 20480 / 11264, Q4 28672 / 18432) and 16384 on IQ3.
inline long long ceil_to(long long v, long long q) { return (v + q - 1) / q * q; }
inline std::vector<Applied> derive(const std::vector<Row>& rows, const std::function<const char*(const char*)>& get,
                                   const std::function<void(const char*, const char*)>& set) {
    std::vector<Applied> out;
    auto in_table = [&](const char* k) { for (const auto& r : rows) if (r.key == k) return true; return false; };
    struct Rule { const char* base; const char* key; long long quantum; };
    for (const Rule& u : {Rule{"STRATA_TP_PREFILL_MAX", "STRATA_TP_PREFILL_CHUNK", 256}, Rule{"STRATA_DP_MIN", "STRATA_DP_PIPE_CHUNK", 1024}}) {
        if (!in_table(u.base) || get(u.key) != nullptr) continue;
        const char* bv = get(u.base);
        if (bv == nullptr || *bv == '\0') continue;
        char* end = nullptr;
        errno = 0;
        const long long v = std::strtoll(bv, &end, 10);
        if (errno != 0 || end == bv || *end != '\0' || v <= 0 || v > (1LL << 40)) continue;
        const std::string d = std::to_string(ceil_to(v, u.quantum));
        set(u.key, d.c_str());
        out.push_back({u.key, d, "derived"});
    }
    return out;
}

/// fill: env, else the row (set with overwrite 0), then the derived rows (derive above).  get / set are std::getenv /
/// setenv in the engine, fakes in the test.
inline std::vector<Applied> apply(const std::vector<Row>& rows, const std::function<const char*(const char*)>& get,
                                  const std::function<void(const char*, const char*)>& set) {
    std::vector<Applied> out;
    for (const auto& r : rows) {
        const char* e = get(r.key.c_str());
        if (e != nullptr) { out.push_back({r.key, e, "env"}); continue; }
        set(r.key.c_str(), r.value.c_str());
        out.push_back({r.key, r.value, "table"});
    }
    for (Applied& a : derive(rows, get, set)) out.push_back(std::move(a));
    return out;
}

}  // namespace strata::program::planner_rows
