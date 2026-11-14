// [claude] planner rev 5, option C (frozen rule, localLLM planner branch 67850ea: results/planner-20261112/rev5/rule_c.py +
// model_c.py), ported for the engine's SHADOW path.  Pure arithmetic, no CUDA/HIP: the one-card staged limit X derived from
//   start-up facts : cache slots and bytes, the loan per path (part_slots: the start-up loan table), the link rate BW (M1),
//                    the expert profile's ranking (the cache's initial residency: the first slots - lent pairs)
//   the pack table : per text (english, code, chinese) and builder length T, each staged read's per-layer per-expert routed-token
//                    counts (--dump-prompt-counts) and its part time; the decode windows' ms; the stream compute shape; o_s
//   start reads    : the canned English 2048 staged read's part times, exactly 8 (alternating prompts); the median of reads 4-8
//                    (re-frozen, review 12 C1: localLLM planner branch 66c39be)
// The arithmetic follows model_c.py operation for operation (double precision, the same order), so X matches bit for bit
// (planner_c_test.cpp checks it against rule_c.py on every stored load).  Nothing here changes a plan.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <stdexcept>
#include <vector>

// review 11 S6: no FMA contraction in this arithmetic (with -march=native the ratio's last bit moved; X never did)
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC push_options
#pragma GCC optimize("fp-contract=off")
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#endif

namespace strata::program::planner_c {

struct TextTable {
    std::vector<int64_t> lengths;                          ///< builder part lengths T (sorted)
    std::map<int64_t, std::vector<int>> chunks;            ///< T -> indices into the counts file
    std::map<int64_t, std::vector<double>> builder_ms;     ///< T -> the builder's staged part times
};
struct Pack {
    std::string quant, counts_file, engine_pack;   ///< engine_pack: the --pack directory name the table was built with
    int n_chunks = 0;
    double win = 0, sc_a = 0, sc_b = 0, o_s = 0;            ///< decode windows ms; stream compute a + b c; o_s
    int layers = 48, experts = 512, maxt = 8, grid = 256, margin = 128, max_reads = 8, start = 2048, lo = 3328, hi = 12288, step = 16, start_limit = 16641;
    uint64_t content_hash = 0;                             ///< FNV-1a of the pack file and the counts file (the cache key)
    uint64_t profile_hash = 0;                             ///< review 15 D1: the `profile` line - file_hash of the expert profile it was built under (0: none)
    double ref_start_ms = 0;                               ///< review 16/17 D10: the `ref_start` line - the reference start read in the
                                                           ///< engine's own statistic (ms); 0: none, the builder's English median is used
    double share = 0.6, guard = 1.6, conv = 0.003;
    TextTable text[3];
    std::vector<std::vector<int32_t>> probe_tokens;      ///< the canned English start-read prompts (token ids), by variant
    std::vector<std::vector<int32_t>> counts;              ///< per chunk: layers * experts counts
    std::vector<int64_t> chunk_T;                          ///< per chunk: its chunk tokens
};
struct Facts {
    int64_t slots = 0;
    double cache_bytes = 0;
    std::map<int64_t, double> loan_staged, loan_streamed;  ///< chunk -> slots (part_slots)
    double bw = 0;                                         ///< bytes / s (M1)
    std::vector<int64_t> rank;                             ///< profile rank per (layer, expert); 1e9 when unranked
};
struct Result {
    bool derived = false;
    int64_t X = -1, acted = -1;
    int64_t crossing[3] = {-1, -1, -1};
    double ratio = -1, add_ms = 0, start_ms = 0;
    int reads_used = 0;
    std::string why;
};

/// model_c.interp: linear on the sorted keys, c clamped to their range
inline double interp(const std::map<int64_t, double>& tab, double c) {
    const double k0 = (double) tab.begin()->first, k1 = (double) tab.rbegin()->first;
    c = std::min(std::max(c, k0), k1);
    auto it = tab.lower_bound((int64_t) std::ceil(c));
    // keys are integers; c is integral in every call (the rule's lengths), so lower_bound(c) is numpy's searchsorted-left
    if ((double) it->first == c) return it->second;
    auto prev = std::prev(it);
    const double a = (double) prev->first, b = (double) it->first;
    return prev->second + (it->second - prev->second) * (c - a) / (b - a);
}
inline double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    const size_t n = v.size();
    return n % 2 ? v[n / 2] : (v[n / 2 - 1] + v[n / 2]) / 2.0;
}

/// one text's model at THIS load (model_c.ModelC with text = t, the additive correction)
struct TextModel {
    const Pack& p; const Facts& f; int t;
    double b = 0, add = 0;
    std::map<int64_t, double> ngs, rem;   ///< by builder length: median GPU-streamed experts; staged remainder
    TextModel(const Pack& p_, const Facts& f_, int t_) : p(p_), f(f_), t(t_) {
        b = f.cache_bytes / (double) f.slots;
        for (int64_t T : p.text[t].lengths) ngs[T] = counts_at(T);
        for (int64_t T : p.text[t].lengths) rem[T] = median(p.text[t].builder_ms.at(T)) - n_gs((double) T) * b / f.bw * 1e3;
    }
    double counts_at(int64_t T) const {   // model_c.counts_at: n_gs at this residency, median over the length's reads
        const double lent = interp(f.loan_staged, (double) T);
        const double limit = (double) f.slots - lent;
        std::vector<double> v;
        for (int ci : p.text[t].chunks.at(T)) {
            const std::vector<int32_t>& c = p.counts[(size_t) ci];
            int64_t n_gs = 0;
            for (int l = 0; l < p.layers; ++l) {
                int64_t nstream = 0, cand = 0;
                for (int e = 0; e < p.experts; ++e) {
                    const size_t i = (size_t) l * p.experts + e;
                    const int32_t k = c[i];
                    if (k <= 0 || (double) f.rank[i] < limit) continue;   // not routed, or resident
                    ++nstream;
                    if (k <= p.maxt) ++cand;
                }
                const int64_t take = std::min<int64_t>(cand, (int64_t) std::nearbyint(p.share * (double) nstream));   // numpy rint: half to even
                n_gs += nstream - take;
            }
            v.push_back((double) n_gs);
        }
        return median(v);
    }
    double n_gs(double c) const { return interp(ngs, c); }
    double copy_staged(double c) const { return n_gs(c) * b / f.bw * 1e3; }
    double staged(double c) const {
        const double r = 1.0 * interp(rem, c) + add;
        return copy_staged(c) + r + p.win + interp(f.loan_staged, c + 7) * b / f.bw * 1e3;
    }
    double stream(double c, double* ratio = nullptr) const {
        const double lent = interp(f.loan_streamed, c + 7);
        const double copy = ((double) (p.layers * p.experts) - ((double) f.slots - lent)) * b / f.bw * 1e3;
        const double comp = 1.0 * (p.sc_a + p.sc_b * c);
        if (ratio) *ratio = copy / comp;
        return std::max(copy, comp) + p.o_s + p.win + lent * b / f.bw * 1e3;
    }
    void crossing(int64_t& L, double& ratio) const {
        int64_t last = -1;
        for (int64_t c = p.lo; c < p.hi; c += p.step) {
            if (stream((double) c) / staged((double) c) - 1 > 0) last = c;
            else break;
        }
        L = (last >= 0 ? last : p.lo) + 7;
        stream((double) (L - 7), &ratio);
    }
};

/// rule_c.start_read (RE-FROZEN, review 12 C1): the median of reads 4-8 of exactly 8 counted start reads; false with fewer than 8
inline bool start_read(const std::vector<double>& r, double& out, int& used) {
    used = (int) std::min<size_t>(r.size(), 8);
    if (r.size() < 8) return false;
    std::vector<double> v(r.begin() + 3, r.begin() + 8);
    std::sort(v.begin(), v.end());
    out = v[2];
    return true;
}
/// the rule frozen at 67850ea (first of two successive reads within conv), kept for the record; not used by derive
inline bool converged_c0(const std::vector<double>& r, const Pack& p, double& out, int& used) {
    const int n = (int) std::min<size_t>(r.size(), (size_t) p.max_reads);
    for (int i = 1; i < n; ++i)
        if (std::fabs(r[(size_t) i] / r[(size_t) i - 1] - 1) <= p.conv) { out = r[(size_t) i]; used = i + 1; return true; }
    used = n;
    return false;
}

inline Result derive_from_start(const Pack& p, const Facts& f, double start, int64_t configured);
/// rule_c.derive: the frozen rule
inline Result derive(const Pack& p, const Facts& f, const std::vector<double>& start_reads, int64_t configured) {
    Result res;
    res.acted = configured;
    double start = 0;
    if (!start_read(start_reads, start, res.reads_used)) {
        res.why = "only " + std::to_string(res.reads_used) + " start reads (8 needed)";
        return res;
    }
    Result r = derive_from_start(p, f, start, configured);
    r.reads_used = res.reads_used;
    return r;
}
/// the rule after the start read (review 13 fallback level 3 feeds it the pack's own reference: the builder's English median)
inline Result derive_from_start(const Pack& p, const Facts& f, double start, int64_t configured) {
    Result res;
    res.acted = configured;
    res.start_ms = start;
    TextModel en(p, f, 0);
    const double r = start - en.copy_staged((double) (p.start - 7));
    en.add = r - interp(en.rem, (double) (p.start - 7));
    res.add_ms = en.add;
    int64_t bestL = -1; double bestR = 0;
    for (int t = 0; t < 3; ++t) {
        TextModel m(p, f, t);
        m.add = en.add;
        int64_t L; double ratio;
        m.crossing(L, ratio);
        res.crossing[t] = L;
        if (bestL < 0 || L < bestL || (L == bestL && ratio < bestR)) { bestL = L; bestR = ratio; }
    }
    res.ratio = bestR;
    if (bestR < p.guard) {
        char b[96];
        std::snprintf(b, sizeof b, "regime guard: copy/compute %.2f < %.2f", bestR, p.guard);
        res.why = b;
        return res;
    }
    const int64_t v = bestL - p.margin - 1;
    res.X = (v >= 0 ? v / p.grid : -((-v + p.grid - 1) / p.grid)) * p.grid + 1;   // Python's floor division
    res.derived = true;
    res.acted = std::max(configured, res.X);
    return res;
}

/// the pack's reference start read (level 3's input; review 16 X_ref's input).  Review 16/17 D10: the table's `ref_start` line when it
/// has one - the mean start read (median of reads 4-8) of the validated cold engine derivations, i.e. the engine's own statistic; the
/// builder's English median at start - 7 (a different statistic: the builder's staged reads, 16 / 15 / 4-11 ms off) otherwise
inline bool ref_start(const Pack& p, double& s) {
    if (p.ref_start_ms > 0) { s = p.ref_start_ms; return true; }
    const auto it = p.text[0].builder_ms.find((int64_t) (p.start - 7));
    if (it == p.text[0].builder_ms.end() || it->second.empty()) return false;
    s = median(it->second);
    return true;
}
/// the rule at the pack's reference start on this load's facts (level 3's value); false without the facts, a reference or an X
inline bool ref_X(const Pack& p, const Facts& f, int64_t& X) {
    double s = 0;
    if (!(f.bw > 0) || f.slots <= 0 || !ref_start(p, s)) return false;   // review 14 F2: needs the start-up facts (no M1 -> no reference)
    const Result t = derive_from_start(p, f, s, -1);
    if (!t.derived) return false;
    X = t.X;
    return true;
}
// ---- review 16 D8/D9 (option e): bracket the limit between two statistics of the same 8 reads ----
//   s_med = the median of reads 4-8 (derive's statistic, unchanged)      X_med = rule(s_med)  (= derive's X)
//   s_lo  = the 2nd smallest of reads 4-8                                 X_lo  = rule(s_lo)
//   X_ref = the rule at the pack's reference start on this load's facts (level 3's value)
//   X     = X_ref limited to [min(X_med, X_lo), max(X_med, X_lo)]  (min/max: the IQ3 regime-guard edge cannot invert the range)
// A slow read (swap-in, page fault) can only slow a read, so the fastest reads are the undisturbed ones; the 2nd smallest ignores one
// freak fast read.  The limit moves down only when even the 2nd-fastest settled read says so, up only when the median says so; when
// the two disagree the load's reads straddle a step and the reference decides.  X stays between X_med and X_lo, so a wrong reference
// gives at worst today's X_med.  Edge cases: no X from the median -> not bracketed (today's paths); no X_ref -> X = X_med; no X from
// the 2nd smallest (below IQ3's guard) -> X = X_med.  The cache takes X only when the bracket is CLOSED (X_med = X_lo = X_ref) and
// stable_X holds.  derive, rule_c.py and the stored vectors are unchanged: the bracket sits in apply_limit and the cache condition.
struct Bracket {
    bool on = false;                                  ///< bracketed (an X from the median and 8 reads); off -> apply_limit uses Result::X
    double s_lo = 0, s_ref = 0;                       ///< the 2nd-smallest settled read; the reference start (0: none)
    int64_t X_med = -1, X_lo = -1, X_ref = -1, X = -1;
    std::string note;                                 ///< why X is not the clamp of X_ref (an edge case), empty otherwise
    bool closed() const { return on && X_med > 0 && X_med == X_lo && X_lo == X_ref; }
    /// why the cache refuses this X (empty when closed)
    std::string open_why() const {
        if (closed()) return "";
        if (!on) return "not bracketed";
        return "bracket open: X_med " + std::to_string((long long) X_med) + ", X_lo " + std::to_string((long long) X_lo) + ", X_ref " + std::to_string((long long) X_ref);
    }
};
/// the 2nd smallest of reads 4-8 (false with fewer than 8 reads)
inline bool start_read_lo(const std::vector<double>& r, double& out) {
    if (r.size() < 8) return false;
    std::vector<double> v(r.begin() + 3, r.begin() + 8);
    std::sort(v.begin(), v.end());
    out = v[1];
    return true;
}
/// the combination alone (X_med > 0; X_lo / X_ref -1 when the rule gave none)
inline void bracket_combine(Bracket& b) {
    b.on = true;
    if (b.X_ref < 0) { b.X = b.X_med; b.note = "no reference X: X_med"; return; }
    if (b.X_lo < 0) { b.X = b.X_med; b.note = "no X from the 2nd-smallest read: X_med"; return; }
    const int64_t lo = std::min(b.X_med, b.X_lo), hi = std::max(b.X_med, b.X_lo);
    b.X = std::min(std::max(b.X_ref, lo), hi);
    b.note.clear();
}
/// the bracket of a load: r = derive(p, f, reads, ...).  Not derived (or fewer than 8 reads) -> on = false, today's paths
inline Bracket bracket(const Pack& p, const Facts& f, const std::vector<double>& reads, const Result& r) {
    Bracket b;
    if (!r.derived || reads.size() < 8 || !start_read_lo(reads, b.s_lo)) return b;
    b.X_med = r.X;
    const Result lo = derive_from_start(p, f, b.s_lo, -1);
    if (lo.derived) b.X_lo = lo.X;
    if (!ref_start(p, b.s_ref)) b.s_ref = 0;
    ref_X(p, f, b.X_ref);   // (left -1 when there is none)
    bracket_combine(b);
    return b;
}

/// review 13: the staged limit the engine APPLIES (key present: configured >= 0; absent: -1)
///  1  key present                -> max(configured, X)   (no X: configured)        - also the rollback
///  2  key absent, X derived      -> X
///  3  key absent, no X           -> the rule at the pack's own reference start read (the builder's English median), with a warning
///  4  no valid pack              -> the engine default (a deploy test must make this unreachable)
/// review 16: X is the bracketed X when b is given and on (a derivation from 8 reads), else r's X (a cache hit stores a closed X)
inline int64_t apply_limit(const Pack* p, const Facts& f, const Result* r, int64_t configured, int64_t engine_default, int& level, std::string& note,
                           const Bracket* b = nullptr) {
    const int64_t X = r && r->derived && b && b->on ? b->X : r ? r->X : -1;
    if (configured >= 0) { level = 1; note = r && r->derived ? "max(configured, X)" : "configured (no X: " + (r ? r->why : std::string("no rule")) + ")";
                           return r && r->derived ? std::max(configured, X) : configured; }
    if (r && r->derived) { level = 2; note = "X"; return X; }
    int64_t Xref = -1;
    if (p && ref_X(*p, f, Xref)) { level = 3; note = "WARNING: no X this load (" + (r ? r->why : std::string("no rule")) + "): the pack's reference value"; return Xref; }
    level = 4; note = p ? "WARNING: no start-up facts (link rate): the engine default" : "WARNING: no valid pack: the engine default"; return engine_default;
}

/// FNV-1a 64 over a file's bytes (the cache key's build and pack identity); 0 when unreadable
inline uint64_t file_hash(const std::string& path, uint64_t h = 1469598103934665603ull) {
    std::FILE* fh = std::fopen(path.c_str(), "rb");
    if (!fh) return 0;
    std::vector<unsigned char> buf(1 << 20);
    size_t n;
    while ((n = std::fread(buf.data(), 1, buf.size(), fh)) > 0)
        for (size_t i = 0; i < n; ++i) { h ^= buf[i]; h *= 1099511628211ull; }
    std::fclose(fh);
    return h;
}
// ---- the derived-X cache (coordinator scope addition to review 14): one record per pack file, card and variant,
//      PACK.txt.<pci>.<first 8 hex of the key's args_env>.x-cache (review 16 S8) ----
// A load whose key matches the stored one (host, exe, pack and expert-profile content, card, copy mode, slots/bytes/context/chunk, the
// engine's arguments and STRATA_/HSA_/GPU_ environment, boot id and kernel release, free VRAM after load in 32 MiB buckets) and whose
// M1 is within kCacheBwTol of the stored M1 applies the stored X without the 8 start reads.  Anything else - no file, a short or garbled
// file, another format version, a checksum that does not match, another key, M1 moved - is a MISS with its reason, and the start reads
// run.  Written to PATH.tmp.<pid> and renamed over PATH (a reader sees the old record or the new one, never half); written only when
// stable_X holds (review 15 D5) and the bracket is closed (review 16 D9: X_med = X_lo = X_ref).
constexpr int kCacheVersion = 2;          ///< review 15: 2 = the D1/D2/D3 key fields; a version-1 record reads as a miss
constexpr double kCacheBwTol = 0.0025;    ///< review 15 D5: M1 within 0.25% of the stored M1 on a hit (was 0.5%)
struct CacheEntry { int64_t X = -1; double start_ms = 0, bw = 0; int reads = 0; };
/// review 17c D11 ("M1 settles"): the link probe times passes until the last three agree within tol (the hit tolerance,
/// kCacheBwTol), at most kM1MaxPasses timed passes after the warm-up.  Settled: the median of those three (three quiet passes give
/// exactly the median of 3, as before).  Not settled: the 2nd fastest of all passes (a disturbance only slows a pass), and the
/// engine writes no cache record from that load.  Fewer than 3 passes: 0 (no link rate).
constexpr int kM1MaxPasses = 9;   ///< timed passes after the warm-up; 3 when the link is quiet (a cap, not a threshold)
inline double m1_from_passes(const std::vector<double>& g, double tol, bool& settled) {
    settled = false;
    const size_t n = g.size();
    if (n < 3) return 0.0;
    std::vector<double> v(g.end() - 3, g.end());
    std::sort(v.begin(), v.end());
    if (v[2] <= v[0] * (1.0 + tol)) { settled = true; return v[1]; }
    v = g; std::sort(v.begin(), v.end());
    return v[n - 2];
}
inline uint64_t fnv_str(const std::string& s, uint64_t h = 1469598103934665603ull) {
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}
inline std::string cache_body(const std::string& key, const CacheEntry& e) {
    char b[160];
    std::snprintf(b, sizeof b, "x %lld start %.4f bw %.6f reads %d\n", (long long) e.X, e.start_ms, e.bw, e.reads);
    return "strata-planner-x-cache " + std::to_string(kCacheVersion) + "\nkey " + key + "\n" + b;
}
inline bool cache_write(const std::string& path, const std::string& key, const CacheEntry& e, const std::string& tmp_suffix, std::string& err) {
    const std::string body = cache_body(key, e);
    char ck[64];
    std::snprintf(ck, sizeof ck, "check %016llx\n", (unsigned long long) fnv_str(body));
    const std::string tmp = path + ".tmp." + tmp_suffix;
    std::FILE* fh = std::fopen(tmp.c_str(), "wb");
    if (!fh) { err = "cannot write " + tmp; return false; }
    const std::string all = body + ck;
    const bool ok = std::fwrite(all.data(), 1, all.size(), fh) == all.size();
    if (std::fclose(fh) != 0 || !ok) { std::remove(tmp.c_str()); err = "write of " + tmp + " failed"; return false; }
#if defined(_WIN32)
    std::remove(path.c_str());
#endif
    if (std::rename(tmp.c_str(), path.c_str()) != 0) { std::remove(tmp.c_str()); err = "rename to " + path + " failed"; return false; }
    return true;
}
/// the first key field (name=value) that differs, for the miss reason
inline std::string key_diff(const std::string& a, const std::string& b) {
    auto split = [](const std::string& s) { std::map<std::string, std::string> m; size_t i = 0;
        while (i < s.size()) { while (i < s.size() && s[i] == ' ') ++i; size_t j = s.find(' ', i); if (j == std::string::npos) j = s.size();
            const std::string w = s.substr(i, j - i); const size_t q = w.find('='); if (!w.empty()) m[q == std::string::npos ? w : w.substr(0, q)] = q == std::string::npos ? "" : w.substr(q + 1); i = j; }
        return m; };
    const auto ma = split(a), mb = split(b);
    for (const auto& kv : mb) { auto it = ma.find(kv.first); if (it == ma.end() || it->second != kv.second) return kv.first + " " + (it == ma.end() ? "absent" : it->second) + " -> " + kv.second; }
    for (const auto& kv : ma) if (!mb.count(kv.first)) return kv.first + " no longer in the key";
    return "the key differs";
}
inline bool cache_read(const std::string& path, const std::string& key, double bw_now, double bw_tol, const Pack& p, CacheEntry& e, std::string& why) {
    std::FILE* fh = std::fopen(path.c_str(), "rb");
    if (!fh) { why = "no cache file"; return false; }
    std::string all; char buf[4096]; size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, fh)) > 0 && all.size() < (1u << 16)) all.append(buf, n);
    std::fclose(fh);
    std::vector<std::string> ln; size_t a = 0;
    while (a < all.size()) { size_t b = all.find('\n', a); if (b == std::string::npos) { ln.push_back(all.substr(a)); break; } ln.push_back(all.substr(a, b - a)); a = b + 1; }
    if (ln.size() >= 1 && ln[0].compare(0, 23, "strata-planner-x-cache ") == 0 && ln[0] != "strata-planner-x-cache " + std::to_string(kCacheVersion)) {
        why = "a cache record of format " + ln[0].substr(23) + ", this engine reads " + std::to_string(kCacheVersion); return false;   // review 15: v1 -> miss
    }
    if (ln.size() != 4 || ln[0] != "strata-planner-x-cache " + std::to_string(kCacheVersion) || ln[1].compare(0, 4, "key ") != 0 || ln[3].compare(0, 6, "check ") != 0) { why = "the cache file is not a cache record"; return false; }
    const std::string body = ln[0] + "\n" + ln[1] + "\n" + ln[2] + "\n";
    char ck[32]; std::snprintf(ck, sizeof ck, "%016llx", (unsigned long long) fnv_str(body));
    if (ln[3].substr(6) != ck) { why = "the cache record's checksum does not match"; return false; }
    if (ln[1].substr(4) != key) { why = "another key (" + key_diff(ln[1].substr(4), key) + ")"; return false; }
    long long X = -1; double st = 0, bw = 0; int reads = 0; char tail = 0;
    if (std::sscanf(ln[2].c_str(), "x %lld start %lf bw %lf reads %d%c", &X, &st, &bw, &reads, &tail) != 4) { why = "the cache record's values do not parse"; return false; }
    if (X < 1024 || X > p.hi + 7 || X % p.grid != 1 || reads != 8 || !(bw > 0)) { why = "the cache record's values are out of range"; return false; }
    if (!(bw_now > 0) || std::fabs(bw_now / bw - 1) > bw_tol) {
        char b[96]; std::snprintf(b, sizeof b, "M1 %.4f GB/s now, %.4f GB/s stored", bw_now / 1e9, bw / 1e9); why = b; return false;
    }
    e.X = X; e.start_ms = st; e.bw = bw; e.reads = reads;
    return true;
}
/// review 15 D5: write the record only when the rule gives the same X at start +-10 ms and at M1 +-0.25% (the hit tolerance) - four more
/// derive_from_start calls on this load's facts.  False (why: the first perturbation that moves X) -> not cached, every load reads.
inline bool stable_X(const Pack& p, const Facts& f, const Result& r, int64_t configured, std::string& why) {
    if (!r.derived) { why = "no X"; return false; }
    const struct { double dstart, bwf; const char* what; } v[4] = {
        {+10.0, 1.0, "start +10 ms"}, {-10.0, 1.0, "start -10 ms"}, {0.0, 1.0 + kCacheBwTol, "M1 +0.25%"}, {0.0, 1.0 - kCacheBwTol, "M1 -0.25%"}};
    for (const auto& x : v) {
        Facts g = f; g.bw = f.bw * x.bwf;
        const Result d = derive_from_start(p, g, r.start_ms + x.dstart, configured);
        if (!d.derived || d.X != r.X) {
            why = std::string(x.what) + " gives " + (d.derived ? "X " + std::to_string((long long) d.X) : "no X (" + d.why + ")") + ", not " + std::to_string((long long) r.X);
            return false;
        }
    }
    return true;
}
/// review 16 S9: /proc/vmstat's pswpin and pgmajfault (false when unreadable: not Linux, or a field missing)
inline bool vmstat_read(long long& pswpin, long long& pgmajfault) {
    pswpin = pgmajfault = -1;
    std::FILE* fh = std::fopen("/proc/vmstat", "r");
    if (!fh) return false;
    char name[64]; long long v;
    while (std::fscanf(fh, "%63s %lld", name, &v) == 2) {
        if (std::strcmp(name, "pswpin") == 0) pswpin = v;
        else if (std::strcmp(name, "pgmajfault") == 0) pgmajfault = v;
    }
    std::fclose(fh);
    return pswpin >= 0 && pgmajfault >= 0;
}
/// a NUL-separated /proc file (cmdline, environ) as its strings; empty when unreadable
inline std::vector<std::string> nul_list(const std::string& path) {
    std::vector<std::string> out;
    std::FILE* fh = std::fopen(path.c_str(), "rb");
    if (!fh) return out;
    std::string all; char buf[4096]; size_t n;
    while ((n = std::fread(buf, 1, sizeof buf, fh)) > 0) all.append(buf, n);
    std::fclose(fh);
    size_t a = 0;
    while (a < all.size()) { size_t b = all.find('\0', a); if (b == std::string::npos) b = all.size(); out.push_back(all.substr(a, b - a)); a = b + 1; }
    return out;
}
/// review 15 D2: the cache key's args_env field.  Hashed (FNV-1a 64, fnv_str):
///   - the engine's command line WITHOUT argv[0], in order (an argument's position carries its meaning: `--spec 4 --kv int8`);
///     argv[0] is the binary's path, and its content is already the key's exe= field (the same binary at another path is the same load)
///   - every environment variable whose name starts with STRATA_, HSA_ or GPU_, as NAME=VALUE, sorted (the environment's order does not
///     matter), except kArgsEnvSkip below.  Other variables (PATH, HOME, HIP_VISIBLE_DEVICES - the card is the gpu= field) are not hashed.
/// kArgsEnvSkip: values that legitimately differ between identical loads -
///   STRATA_OWNER_ATTEST_KEY  a fresh secret per launch (serve/launch_identity: the owner's attestation key)
///   STRATA_DS4_SLOT_DIR      a folder the server derives from the log path (DeepSeek only; the cache lives in the log directory anyway)
inline constexpr const char* kArgsEnvSkip[] = {"STRATA_OWNER_ATTEST_KEY", "STRATA_DS4_SLOT_DIR"};
inline uint64_t args_env_hash(const std::vector<std::string>& cmdline, const std::vector<std::string>& environ_kv) {
    std::vector<std::string> env;
    for (const std::string& kv : environ_kv) {
        if (kv.compare(0, 7, "STRATA_") != 0 && kv.compare(0, 4, "HSA_") != 0 && kv.compare(0, 4, "GPU_") != 0) continue;
        const std::string name = kv.substr(0, kv.find('='));
        bool skip = false;
        for (const char* s : kArgsEnvSkip) skip = skip || name == s;
        if (!skip) env.push_back(kv);
    }
    std::sort(env.begin(), env.end());
    uint64_t h = fnv_str("args");
    for (size_t i = 1; i < cmdline.size(); ++i) { h = fnv_str(cmdline[i], h); h = fnv_str(std::string(1, '\0'), h); }
    h = fnv_str(std::string(1, '\x01') + "env", h);
    for (const std::string& kv : env) { h = fnv_str(kv, h); h = fnv_str(std::string(1, '\0'), h); }
    return h;
}
// ---- loading (the pack's line format, pack_export.py) ----
inline bool load_pack(const std::string& path, const std::string& dir, Pack& p, std::string& err) {
    std::FILE* fh = std::fopen(path.c_str(), "r");
    if (!fh) { err = "cannot read " + path; return false; }
    char line[1 << 16];
    std::vector<std::string> lines;   // read and closed first: a parse error below (std::stoi throws) leaks no handle
    while (std::fgets(line, sizeof line, fh)) lines.emplace_back(line);
    std::fclose(fh);
    for (const std::string& s : lines) {
        auto words = [&]() { std::vector<std::string> w; size_t i = 0; while (i < s.size()) { while (i < s.size() && isspace((unsigned char) s[i])) ++i;
                                 size_t j = i; while (j < s.size() && !isspace((unsigned char) s[j])) ++j; if (j > i) w.push_back(s.substr(i, j - i)); i = j; } return w; }();
        if (words.empty()) continue;
        const std::string& k = words.at(0);
        if (k == "quant") p.quant = words.at(1);
        else if (k == "counts_file") p.counts_file = words.at(1);
        else if (k == "engine_pack") p.engine_pack = words.at(1);
        else if (k == "profile") p.profile_hash = std::stoull(words.at(1), nullptr, 16);   // review 15 D1
        else if (k == "ref_start") {   // review 16/17 D10
            p.ref_start_ms = std::stod(words.at(1));
            if (!(p.ref_start_ms > 0) || !std::isfinite(p.ref_start_ms)) { err = "bad ref_start line"; return false; }
        }
        else if (k == "probe_tokens") {   // probe_tokens <variant> id,id,...
            const size_t v = (size_t) std::stoi(words.at(1));
            if (p.probe_tokens.size() <= v) p.probe_tokens.resize(v + 1);
            const std::string& ids = words.at(2); size_t a = 0;
            while (a < ids.size()) { size_t b = ids.find(',', a); if (b == std::string::npos) b = ids.size(); p.probe_tokens[v].push_back((int32_t) std::stol(ids.substr(a, b - a))); a = b + 1; }
        }
        else if (k == "n_chunks") p.n_chunks = std::stoi(words.at(1));
        else if (k == "win") p.win = std::stod(words.at(1));
        else if (k == "stream_compute") { p.sc_a = std::stod(words.at(1)); p.sc_b = std::stod(words.at(2)); }
        else if (k == "o_s") p.o_s = std::stod(words.at(1));
        else if (k == "const") {
            for (size_t i = 1; i + 1 < words.size(); i += 2) {
                const std::string& n = words.at(i); const std::string& v = words.at(i + 1);
                if (n == "layers") p.layers = std::stoi(v); else if (n == "experts") p.experts = std::stoi(v);
                else if (n == "maxt") p.maxt = std::stoi(v); else if (n == "share") p.share = std::stod(v);
                else if (n == "grid") p.grid = std::stoi(v); else if (n == "margin") p.margin = std::stoi(v);
                else if (n == "guard") p.guard = std::stod(v); else if (n == "conv") p.conv = std::stod(v);
                else if (n == "max_reads") p.max_reads = std::stoi(v); else if (n == "start") p.start = std::stoi(v);
                else if (n == "lo") p.lo = std::stoi(v); else if (n == "hi") p.hi = std::stoi(v); else if (n == "step") p.step = std::stoi(v);
                else if (n == "start_limit") p.start_limit = std::stoi(v);
            }
        } else if (k == "text") {
            const int t = std::stoi(words.at(1)); const int64_t T = std::stoll(words.at(3));
            if (t < 0 || t > 2 || words.size() < 5 || words.at(2) != "T" || words.at(4) != "chunks") { err = "bad text line"; return false; }
            TextTable& tt = p.text[t];
            tt.lengths.push_back(T);
            size_t i = 5;   // "chunks" at 4
            for (; i < words.size() && words[i] != "ms"; ++i) tt.chunks[T].push_back(std::stoi(words.at(i)));
            for (++i; i < words.size(); ++i) tt.builder_ms[T].push_back(std::stod(words.at(i)));
        }
    }
    for (auto& tt : p.text) std::sort(tt.lengths.begin(), tt.lengths.end());
    if (p.layers <= 0 || p.experts <= 0 || p.layers > 4096 || p.experts > 65536) { err = "bad geometry"; return false; }
    // the counts: records (int32 layer, int32 T, int32 n_expert, n_expert counts); a chunk = 48 consecutive layer records
    const std::string cp = dir + "/" + p.counts_file;
    std::FILE* cf = std::fopen(cp.c_str(), "rb");
    if (!cf) { err = "cannot read " + cp; return false; }
    int32_t hdr[3];
    std::vector<std::vector<int32_t>> all; std::vector<int64_t> allT;   // model_c.read_counts: a new chunk at layer 0 or a new T
    std::vector<int32_t> row;
    // the shipped compact form (tools/make_planner_table.py): "SPC2", u32 version 1, layers, experts, maxt, chunks; per chunk int32 T and
    // layers*experts 2-bit classes, 4 a byte, low bits first: 0 not routed, 1 routed by 1..maxt tokens, 2 by more.  The rule reads a
    // count only as "> 0" and "<= maxt" (TextModel::counts_at), so the classes give the same X bit for bit; they load as 0 / 1 / maxt+1.
    char magic[4] = {0, 0, 0, 0};
    const bool compact = std::fread(magic, 1, 4, cf) == 4 && std::memcmp(magic, "SPC2", 4) == 0;
    if (compact) {
        uint32_t h[5];
        if (std::fread(h, sizeof h, 1, cf) != 1 || h[0] != 1 || (int) h[1] != p.layers || (int) h[2] != p.experts || (int) h[3] != p.maxt || h[4] > 100000) {
            std::fclose(cf); err = "compact counts header does not fit the pack (" + std::to_string(p.layers) + "x" + std::to_string(p.experts) + ", maxt " + std::to_string(p.maxt) + ")"; return false;
        }
        const size_t n = (size_t) p.layers * p.experts;
        std::vector<uint8_t> buf((n + 3) / 4);
        for (uint32_t c = 0; c < h[4]; ++c) {
            int32_t T = 0;
            if (std::fread(&T, sizeof T, 1, cf) != 1 || std::fread(buf.data(), 1, buf.size(), cf) != buf.size() || T <= 0) {
                std::fclose(cf); err = "compact counts file is short or has a bad length at chunk " + std::to_string(c); return false;
            }
            std::vector<int32_t> v(n);
            for (size_t i = 0; i < n; ++i) {
                const int k = (buf[i >> 2] >> ((i & 3) * 2)) & 3;
                if (k == 3) { std::fclose(cf); err = "compact counts file has an unknown class at chunk " + std::to_string(c); return false; }
                v[i] = k == 0 ? 0 : k == 1 ? 1 : p.maxt + 1;
            }
            all.push_back(std::move(v)); allT.push_back(T);
        }
        if (std::fgetc(cf) != EOF) { std::fclose(cf); err = "compact counts file is longer than its header says"; return false; }
    } else std::rewind(cf);
    while (!compact && std::fread(hdr, sizeof hdr, 1, cf) == 1) {
        if (hdr[2] != p.experts || hdr[0] < 0 || hdr[0] >= p.layers || hdr[1] <= 0) {   // review 11 S3: never index past a chunk
            std::fclose(cf); err = "counts record (layer " + std::to_string(hdr[0]) + ", " + std::to_string(hdr[2]) + " experts) does not fit the pack's " +
                                   std::to_string(p.layers) + "x" + std::to_string(p.experts); return false;
        }
        row.resize((size_t) hdr[2]);
        if (std::fread(row.data(), sizeof(int32_t), row.size(), cf) != row.size()) break;
        if (hdr[0] == 0 || all.empty() || allT.back() != hdr[1]) { all.emplace_back((size_t) p.layers * p.experts, 0); allT.push_back(hdr[1]); }
        std::copy(row.begin(), row.end(), all.back().begin() + (size_t) hdr[0] * p.experts);
    }
    for (size_t i = 0; i < all.size(); ++i) {   // ...kept only when every layer has a routed expert
        bool ok = true;
        for (int l = 0; l < p.layers && ok; ++l)
            ok = std::any_of(all[i].begin() + (size_t) l * p.experts, all[i].begin() + (size_t) (l + 1) * p.experts, [](int32_t v) { return v != 0; });
        if (ok) { p.counts.push_back(std::move(all[i])); p.chunk_T.push_back(allT[i]); }
    }
    std::fclose(cf);
    if ((int) p.counts.size() != p.n_chunks) { err = "counts file has " + std::to_string(p.counts.size()) + " chunks, the pack says " + std::to_string(p.n_chunks); return false; }
    p.content_hash = file_hash(cp, file_hash(path));
    return true;
}
/// the expert profile's ranking (tools/make_profile.py format): rank[layer * experts + expert] = its position; unranked = 1e9
inline bool load_rank(const std::string& path, int layers, int experts, std::vector<int64_t>& rank, std::string& err) {
    std::FILE* fh = std::fopen(path.c_str(), "rb");
    if (!fh) { err = "cannot read " + path; return false; }
    char magic[4]; uint32_t h[5];
    if (std::fread(magic, 1, 4, fh) != 4 || std::fread(h, sizeof h, 1, fh) != 1) { std::fclose(fh); err = "bad profile"; return false; }
    rank.assign((size_t) layers * experts, 1000000000);
    for (uint32_t i = 0; i < h[4]; ++i) {
        uint16_t le[2];
        if (std::fread(le, sizeof le, 1, fh) != 1) break;
        rank[(size_t) le[0] * experts + le[1]] = i;
    }
    std::fclose(fh);
    return true;
}

/// review 11 S3: a pack that does not fit this model, or whose indices point outside its counts, turns the shadow off
/// review 14 F10: the probe prompts (when present) - every variant 0..n-1 exactly `start` tokens, every id in [0, vocab) (vocab <= 0: ids >= 0)
/// review 15 D1: profile_hash != 0 (the engine: file_hash of its --expert-profile) - the table must record that same profile; the rule's
/// residency and the table's staged times both come from the profile's ranking, so a table built under another profile is refused
inline bool validate(const Pack& p, int64_t layers, int64_t experts, std::string& err, int64_t vocab = -1, uint64_t profile_hash = 0) {
    if (profile_hash != 0 && p.profile_hash != profile_hash) {
        char b[160];
        if (p.profile_hash == 0) std::snprintf(b, sizeof b, "the table records no expert profile; --expert-profile is %016llx", (unsigned long long) profile_hash);
        else std::snprintf(b, sizeof b, "built for another expert profile: the table's %016llx, --expert-profile %016llx", (unsigned long long) p.profile_hash, (unsigned long long) profile_hash);
        err = b; return false;
    }
    if (p.layers != layers || p.experts != experts) { err = "the pack is for " + std::to_string(p.layers) + "x" + std::to_string(p.experts) + " experts, the model has " + std::to_string(layers) + "x" + std::to_string(experts); return false; }
    if (p.step <= 0 || p.grid <= 0 || p.max_reads < 2 || p.lo >= p.hi || p.start <= 7) { err = "bad constants"; return false; }
    for (int t = 0; t < 3; ++t) {
        const TextTable& tt = p.text[t];
        if (tt.lengths.size() < 2) { err = "text " + std::to_string(t) + " has fewer than 2 lengths"; return false; }
        for (int64_t T : tt.lengths) {
            auto c = tt.chunks.find(T); auto m = tt.builder_ms.find(T);
            if (c == tt.chunks.end() || c->second.empty() || m == tt.builder_ms.end() || m->second.empty()) { err = "text " + std::to_string(t) + " length " + std::to_string(T) + " is empty"; return false; }
            for (int ci : c->second) if (ci < 0 || ci >= (int) p.counts.size() || p.chunk_T[(size_t) ci] != T) { err = "chunk index out of range or of another length"; return false; }
        }
    }
    for (size_t v = 0; v < p.probe_tokens.size(); ++v) {
        const std::vector<int32_t>& ids = p.probe_tokens[v];
        if ((int64_t) ids.size() != (int64_t) p.start) { err = "probe_tokens variant " + std::to_string(v) + " has " + std::to_string(ids.size()) + " tokens, not start " + std::to_string(p.start); return false; }
        for (int32_t id : ids) if (id < 0 || (vocab > 0 && id >= vocab)) { err = "probe_tokens variant " + std::to_string(v) + " has token id " + std::to_string(id) + " outside the vocabulary (" + std::to_string(vocab) + ")"; return false; }
    }
    return true;
}

}  // namespace strata::program::planner_c

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
