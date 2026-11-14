// [claude] planner rev 5 shadow: planner_c.hpp against rule_c.py (localLLM 67850ea) on every stored load - X, acted, the three
// per-text crossings and the guard ratio must match.  CPU only.
//   g++ -O2 -std=c++20 -Iinclude src/program/planner_c_test.cpp -o planner_c_test && ./planner_c_test PACKDIR COUNTSDIR PROFILE [VECTORS]
//   (the shipped table: ./planner_c_test data/planner data/planner data/expert-profile.bin VECTORS)
#include "strata/program/planner_c.hpp"
#include <cstdio>
#include <sstream>
#include <fstream>
#include <iterator>
#include <unistd.h>
using namespace strata::program::planner_c;
int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s PACKDIR COUNTSDIR PROFILE [VECTORS]\n", argv[0]); return 2; }
    const std::string pd = argv[1], cd = argv[2], prof = argv[3];
    std::map<std::string, Pack> packs; std::string err;
    for (const char* q : {"iq3", "iq4", "q4"}) {
        Pack p;
        // the rev 5 export (pack-<q>.txt + its int32 counts) or the shipped table (data/planner/qwen-<q>.txt + its compact counts)
        std::string pp = pd + "/pack-" + q + ".txt";
        if (std::FILE* t = std::fopen(pp.c_str(), "r")) std::fclose(t); else pp = pd + "/qwen-" + q + ".txt";
        if (!load_pack(pp, cd, p, err) || !validate(p, 48, 512, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        std::printf("pack %s: %s, %d chunks\n", q, pp.c_str(), (int) p.counts.size());
        packs[q] = std::move(p);
    }
    std::vector<int64_t> rank;
    if (!load_rank(prof, 48, 512, rank, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    // review 14 F7: the configured one-card values in production since deploy14 (= the rule's X); rule_c.py (frozen, not edited) still
    // carries the old CONFIGURED 4097 / 6145 / 8193, so each vector's stored `acted` is checked against THAT map, ours against this one
    const std::map<std::string, int64_t> configured = {{"iq3", 4609}, {"iq4", 6145}, {"q4", 8705}};
    const std::map<std::string, int64_t> rule_c_configured = {{"iq3", 4097}, {"iq4", 6145}, {"q4", 8193}};
    const std::map<std::string, int64_t> table_X = {{"iq3", 4609}, {"iq4", 6145}, {"q4", 8705}};
    std::map<std::string, std::pair<Facts, std::vector<double>>> sample;
    std::map<std::string, std::string> sample_tag;   // per quant: one real 8-read load whose X is the table X
    std::ifstream vf(argc > 4 ? std::string(argv[4]) : pd + "/vectors.txt"); std::string l1, l2, l3, l4, l5;
    int n = 0, bad = 0;
    while (std::getline(vf, l1) && std::getline(vf, l2) && std::getline(vf, l3) && std::getline(vf, l4) && std::getline(vf, l5)) {
        std::istringstream a(l1); std::string kw, tag, q; Facts f; a >> kw >> tag >> q >> f.slots >> f.cache_bytes >> f.bw;
        auto table = [](const std::string& s, std::map<int64_t, double>& m) { std::istringstream x(s); std::string w; x >> w;
            while (x >> w) { const size_t c = w.find(':'); m[std::stoll(w.substr(0, c))] = std::stod(w.substr(c + 1)); } };
        table(l2, f.loan_staged); table(l3, f.loan_streamed);
        std::vector<double> reads; { std::istringstream x(l4); std::string w; x >> w; double v; while (x >> v) reads.push_back(v); }
        int64_t eX, eA, ec[3]; double er; { std::istringstream x(l5); std::string w; x >> w >> eX >> eA >> ec[0] >> ec[1] >> ec[2] >> er; }
        f.rank = rank;
        const Result r = derive(packs.at(q), f, reads, configured.at(q));
        const Result rc = derive(packs.at(q), f, reads, rule_c_configured.at(q));   // rule_c's own acted (its stale constant)
        const int64_t want_acted = r.derived ? std::max(configured.at(q), r.X) : configured.at(q);
        const bool ok = r.X == eX && rc.acted == eA && r.acted == want_acted &&
                        (eX < 0 && ec[0] < 0 ? true : (r.crossing[0] == ec[0] && r.crossing[1] == ec[1] && r.crossing[2] == ec[2] && r.ratio == er));
        // a real 8-read engine series (day 2 / load20, not a truncated or outlier variant) whose X is the table X
        if (!sample.count(q) && reads.size() == 8 && eX == table_X.at(q) && tag.rfind("shadow-", 0) == 0 && tag.find("-7reads") == std::string::npos && tag.find("-outlier") == std::string::npos) { sample[q] = {f, reads}; sample_tag[q] = tag; }
        // derive_from_start at derive's own start read gives the same result (derive = start_read + derive_from_start)
        if (r.derived) {
            double st = 0; int used = 0; start_read(reads, st, used);
            const Result d = derive_from_start(packs.at(q), f, st, configured.at(q));
            if (d.X != r.X || d.ratio != r.ratio || d.crossing[0] != r.crossing[0]) { std::printf("%-26s DIFF derive_from_start X %lld != derive %lld\n", tag.c_str(), (long long) d.X, (long long) r.X); ++bad; }
        }
        ++n; bad += !ok;
        std::printf("%-26s %s X %lld (rule_c %lld) acted %lld (%lld) crossings %lld %lld %lld (%lld %lld %lld) ratio %.17g (%.17g)%s %s\n", tag.c_str(), ok ? "OK  " : "DIFF",
                    (long long) r.X, (long long) eX, (long long) r.acted, (long long) eA, (long long) r.crossing[0], (long long) r.crossing[1], (long long) r.crossing[2],
                    (long long) ec[0], (long long) ec[1], (long long) ec[2], r.ratio, er, r.why.empty() ? "" : " |", r.why.c_str());
    }
    std::printf("%d vectors, %d differ\n", n, bad);
    // review 14 F7: apply_limit's four levels and derive_from_start, per quant, on a real load's facts (expected: the table X)
    int na = 0, abad = 0;
    auto check = [&](const std::string& what, int64_t got, int lvl, int64_t want, int want_lvl, const std::string& note) {
        const bool ok = got == want && lvl == want_lvl; ++na; abad += !ok;
        std::printf("apply %-4s %-44s %s limit %lld (want %lld) level %d (want %d) | %s\n", "", what.c_str(), ok ? "OK  " : "DIFF", (long long) got, (long long) want, lvl, want_lvl, note.c_str());
    };
    for (const char* q : {"iq3", "iq4", "q4"}) {
        if (!sample.count(q)) { std::printf("apply %s: no 8-read vector with X %lld\n", q, (long long) table_X.at(q)); ++abad; continue; }
        const Pack& p = packs.at(q); const Facts& f = sample[q].first; const std::vector<double>& reads = sample[q].second;
        const int64_t cfg = configured.at(q), X = table_X.at(q), dflt = 3072;
        const std::string Q = q;
        std::printf("apply %s: the facts and reads of %s (M1 %.4f GB/s)\n", q, sample_tag[q].c_str(), f.bw / 1e9);
        const Result r8 = derive(p, f, reads, cfg);
        const Result r7 = derive(p, f, std::vector<double>(reads.begin(), reads.begin() + 7), cfg);
        int lvl = 0; std::string note; int64_t lim;
        lim = apply_limit(&p, f, &r8, cfg, dflt, lvl, note);          check(Q + " 1 key, X derived", lim, lvl, std::max(cfg, X), 1, note);
        lim = apply_limit(&p, f, &r8, X - 256, dflt, lvl, note);      check(Q + " 1 key below X: max(configured, X)", lim, lvl, X, 1, note);
        lim = apply_limit(&p, f, &r8, X + 256, dflt, lvl, note);      check(Q + " 1 key above X: configured", lim, lvl, X + 256, 1, note);
        lim = apply_limit(&p, f, &r7, cfg, dflt, lvl, note);          check(Q + " 1 key, 7 reads: configured", lim, lvl, cfg, 1, note);
        lim = apply_limit(&p, f, &r8, -1, dflt, lvl, note);           check(Q + " 2 no key, X derived", lim, lvl, X, 2, note);
        lim = apply_limit(&p, f, &r7, -1, dflt, lvl, note);           check(Q + " 3 no key, 7 reads: reference", lim, lvl, X, 3, note);
        Result none; none.why = "no probes";
        lim = apply_limit(&p, f, &none, -1, dflt, lvl, note);         check(Q + " 3 no key, no probes: reference", lim, lvl, X, 3, note);
        lim = apply_limit(nullptr, f, &none, -1, dflt, lvl, note);    check(Q + " 4 no key, no valid pack", lim, lvl, dflt, 4, note);
        lim = apply_limit(nullptr, f, nullptr, -1, dflt, lvl, note);  check(Q + " 4 no key, no pack, no result", lim, lvl, dflt, 4, note);
        lim = apply_limit(nullptr, f, &none, cfg, dflt, lvl, note);   check(Q + " 1 key, no valid pack", lim, lvl, cfg, 1, note);
        Facts nobw = f; nobw.bw = 0;
        lim = apply_limit(&p, nobw, &none, -1, dflt, lvl, note);      check(Q + " 4 no key, no link rate", lim, lvl, dflt, 4, note);
        // derive_from_start at the pack's reference (the builder's English median at start - 7) = level 3's value
        const Result ref = derive_from_start(p, f, median(p.text[0].builder_ms.at((int64_t) (p.start - 7))), -1);
        check(Q + " derive_from_start(builder median)", ref.X, ref.derived ? 3 : 0, X, 3, "start " + std::to_string(ref.start_ms));
    }
    // review 14 F10: validate's probe_tokens checks
    {
        Pack p = packs.at("iq4");
        int64_t V = 0;   // the smallest vocabulary the pack's ids fit (the test checks the bound, not a model's size)
        for (const auto& ids : p.probe_tokens) for (int32_t id : ids) V = std::max<int64_t>(V, (int64_t) id + 1);
        auto v = [&](const std::string& what, const Pack& pp, int64_t vocab, bool want) {
            std::string e; const bool got = validate(pp, 48, 512, e, vocab); ++na; abad += got != want;
            std::printf("validate %-40s %s %s%s\n", what.c_str(), got == want ? "OK  " : "DIFF", got ? "valid" : "invalid: ", e.c_str());
        };
        v("the pack", p, V, true);
        Pack a = p; a.probe_tokens[1].pop_back();                       v("a variant one token short", a, V, false);
        Pack b = p; b.probe_tokens.insert(b.probe_tokens.begin() + 1, std::vector<int32_t>());   v("a gap in the variant numbers", b, V, false);
        Pack c = p; c.probe_tokens[0][5] = (int32_t) V;                      v("an id at the vocabulary size", c, V, false);
        Pack d = p; d.probe_tokens[2][7] = -3;                          v("a negative id (no vocabulary given)", d, -1, false);
        Pack e2 = p; e2.probe_tokens.clear();                           v("no probe_tokens (a shadow pack)", e2, V, true);
    }
    // the derived-X cache: write, hit, and every miss (no file, garbage, truncated, checksum, another key, M1 moved, out of range)
    {
        const char* td = std::getenv("TMPDIR");
        const std::string path = std::string(td && *td ? td : "/tmp") + "/planner_c_test." + std::to_string((long long) getpid()) + ".x-cache";
        const Pack& p = packs.at("iq4");
        const std::string key = "host=h exe=0123 pack=4567 gpu=0000:03:00.0 sdma=default slots=7492 bytes=1 context=262144 chunk=16384 vram_free_mib32=320";
        CacheEntry e; e.X = 6145; e.start_ms = 1885.4; e.bw = 13.675e9; e.reads = 8;
        auto c = [&](const std::string& what, const std::string& k, double bw, bool want) {
            CacheEntry g; std::string why; const bool hit = cache_read(path, k, bw, 0.005, p, g, why);
            const bool ok = hit == want && (!hit || g.X == 6145); ++na; abad += !ok;
            std::printf("cache %-44s %s %s%s\n", what.c_str(), ok ? "OK  " : "DIFF", (hit ? "hit X " + std::to_string(g.X) : std::string("miss: ")).c_str(), why.c_str());
        };
        auto put = [&](const std::string& text) { std::FILE* f = std::fopen(path.c_str(), "wb"); std::fwrite(text.data(), 1, text.size(), f); std::fclose(f); };
        auto get = [&]() { std::ifstream f(path, std::ios::binary); return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); };
        std::remove(path.c_str());
        c("no file", key, e.bw, false);
        std::string err;
        const bool w = cache_write(path, key, e, "t", err); ++na; abad += !w; std::printf("cache %-44s %s %s\n", "write", w ? "OK  " : "DIFF", err.c_str());
        c("hit", key, e.bw, true);
        c("hit, M1 +0.4%", key, e.bw * 1.004, true);
        c("M1 +0.6%", key, e.bw * 1.006, false);
        c("another free-VRAM bucket", key.substr(0, key.size() - 3) + "352", e.bw, false);
        std::string k2 = key; k2.replace(k2.find("exe=0123"), 8, "exe=0124");
        c("another exe hash", k2, e.bw, false);
        const std::string good = get();
        put(good.substr(0, good.size() / 2));                 c("truncated", key, e.bw, false);
        put("garbage\x01\x02 not a cache\n");               c("garbage", key, e.bw, false);
        put("");                                              c("empty", key, e.bw, false);
        { std::string b = good; b[b.find("x 6145") + 2] = '7'; put(b); c("value edited (checksum)", key, e.bw, false); }
        { std::string b = good; const size_t i = b.find("exe=0123"); b.replace(i, 8, "exe=0124"); put(b); c("stored key edited (checksum)", k2, e.bw, false); }
        { CacheEntry bad = e; bad.X = 6144; cache_write(path, key, bad, "t", err); c("X off the grid", key, e.bw, false); }
        std::remove(path.c_str());
    }
    std::printf("%d apply/validate/cache vectors, %d differ\n", na, abad);
    return bad || abad ? 1 : 0;
}
