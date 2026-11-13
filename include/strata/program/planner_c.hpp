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
    double share = 0.6, guard = 1.6, conv = 0.003;
    TextTable text[3];
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

/// rule_c.derive: the frozen rule
inline Result derive(const Pack& p, const Facts& f, const std::vector<double>& start_reads, int64_t configured) {
    Result res;
    res.acted = configured;
    double start = 0;
    if (!start_read(start_reads, start, res.reads_used)) {
        res.why = "only " + std::to_string(res.reads_used) + " start reads (8 needed)";
        return res;
    }
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
    while (std::fread(hdr, sizeof hdr, 1, cf) == 1) {
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
inline bool validate(const Pack& p, int64_t layers, int64_t experts, std::string& err) {
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
    return true;
}

}  // namespace strata::program::planner_c

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif
