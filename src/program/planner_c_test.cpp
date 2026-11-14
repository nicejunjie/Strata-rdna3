// [claude] planner rev 5 shadow: planner_c.hpp against rule_c.py (localLLM 67850ea) on every stored load - X, acted, the three
// per-text crossings and the guard ratio must match.  CPU only.
//   g++ -O2 -std=c++20 -Iinclude src/program/planner_c_test.cpp -o planner_c_test && ./planner_c_test PACKDIR COUNTSDIR PROFILE [VECTORS [READS]]
//   (the shipped table, from the repository root: ./planner_c_test data/planner data/planner data/expert-profile.bin
//    - VECTORS defaults to src/program/planner_c_test_vectors.txt, the rev 5 export's 279 stored loads; review 15 S6
//    - READS defaults to src/program/planner_c_test_reads.txt: review 16's engine read series (Fable's harness reads.txt, every engine
//      log under results/planner-20261112/next/ with 8 counted reads up to 11-14 01:57), for the bracket's checks)
//    - M1 (argv[6]) defaults to src/program/planner_c_test_m1.txt: review 17c's 171 logged loads (tag, quant, the three M1 passes, the
//      ON line's M1), for m1_from_passes (D11)
#include "strata/program/planner_c.hpp"
#include <cstdio>
#include <sstream>
#include <fstream>
#include <iterator>
#include <tuple>
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
    struct Vec { std::string tag, q; Facts f; std::vector<double> reads; int64_t X; };
    std::vector<Vec> allvec;                          // review 16: every stored vector, for the bracket's checks
    std::ifstream vf(argc > 4 ? std::string(argv[4]) : std::string("src/program/planner_c_test_vectors.txt")); std::string l1, l2, l3, l4, l5;
    int n = 0, bad = 0;
    while (std::getline(vf, l1) && std::getline(vf, l2) && std::getline(vf, l3) && std::getline(vf, l4) && std::getline(vf, l5)) {
        std::istringstream a(l1); std::string kw, tag, q; Facts f; a >> kw >> tag >> q >> f.slots >> f.cache_bytes >> f.bw;
        auto table = [](const std::string& s, std::map<int64_t, double>& m) { std::istringstream x(s); std::string w; x >> w;
            while (x >> w) { const size_t c = w.find(':'); m[std::stoll(w.substr(0, c))] = std::stod(w.substr(c + 1)); } };
        table(l2, f.loan_staged); table(l3, f.loan_streamed);
        std::vector<double> reads; { std::istringstream x(l4); std::string w; x >> w; double v; while (x >> v) reads.push_back(v); }
        int64_t eX, eA, ec[3]; double er; { std::istringstream x(l5); std::string w; x >> w >> eX >> eA >> ec[0] >> ec[1] >> ec[2] >> er; }
        f.rank = rank;
        allvec.push_back({tag, q, f, reads, eX});
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
    // review 15 D1: the table's `profile` line against the loaded --expert-profile
    {
        const uint64_t H = file_hash(prof);
        auto v = [&](const std::string& what, const Pack& pp, uint64_t want_profile, bool want, const char* must = nullptr) {
            std::string e; const bool got = validate(pp, 48, 512, e, -1, want_profile);
            const bool ok = got == want && (must == nullptr || e.find(must) != std::string::npos); ++na; abad += !ok;
            std::printf("profile  %-44s %s %s%s\n", what.c_str(), ok ? "OK  " : "DIFF", got ? "valid" : "invalid: ", e.c_str());
        };
        const Pack& p = packs.at("iq4");
        if (p.profile_hash != 0) {   // the shipped table: built under this profile
            ++na; abad += p.profile_hash != H;
            std::printf("profile  %-44s %s table %016llx, PROFILE %016llx\n", "the shipped table's line = file_hash(PROFILE)", p.profile_hash == H ? "OK  " : "DIFF",
                        (unsigned long long) p.profile_hash, (unsigned long long) H);
            v("the shipped table under its profile", p, H, true);
        } else std::printf("profile  (the rev 5 export has no profile line: the shipped-table vector is skipped)\n");
        Pack a = p; a.profile_hash = H;          v("profile line = --expert-profile", a, H, true);
        Pack b = p; b.profile_hash = H ^ 1;      v("another profile -> refused", b, H, false, "built for another expert profile");
        Pack c = p; c.profile_hash = 0;          v("no profile line -> refused", c, H, false, "records no expert profile");
        Pack d = p; d.profile_hash = H ^ 1;      v("no check asked (profile 0: the export path)", d, 0, true);
        // the line as load_pack parses it: a copy of the iq4 table with another profile hash, loaded and refused
        const char* td0 = std::getenv("TMPDIR");
        const std::string tp = std::string(td0 && *td0 ? td0 : "/tmp") + "/planner_c_test." + std::to_string((long long) getpid()) + ".table.txt";
        std::string src = pd + "/qwen-iq4.txt";
        if (std::FILE* t = std::fopen((pd + "/pack-iq4.txt").c_str(), "r")) { std::fclose(t); src = pd + "/pack-iq4.txt"; }
        {
            std::ifstream in(src); std::ofstream out(tp); std::string ln; bool put = false;
            while (std::getline(in, ln)) {
                if (ln.rfind("profile ", 0) == 0) continue;
                out << ln << "\n";
                if (!put && ln.rfind("engine_pack ", 0) == 0) { out << "profile 0123456789abcdef\n"; put = true; }
            }
        }
        Pack e; std::string le;
        const bool loaded = load_pack(tp, cd, e, le);
        ++na; abad += !(loaded && e.profile_hash == 0x0123456789abcdefull);
        std::printf("profile  %-44s %s %s profile %016llx\n", "load_pack parses the line", loaded && e.profile_hash == 0x0123456789abcdefull ? "OK  " : "DIFF",
                    loaded ? "loaded" : le.c_str(), (unsigned long long) e.profile_hash);
        if (loaded) v("that table under PROFILE -> the pack cannot be used", e, H, false, "built for another expert profile");
        std::remove(tp.c_str());
    }
    // review 15 D2: args_env_hash - the command line less argv[0] in order, the STRATA_/HSA_/GPU_ environment sorted, kArgsEnvSkip out
    {
        const std::vector<std::string> cmd = {"/x/strata", "--serve", "--pack", "/p", "--spec", "4"};
        const std::vector<std::string> env = {"PATH=/bin", "STRATA_PREFILL_RING=192", "GPU_MAX_HW_QUEUES=6", "HSA_ENABLE_SDMA=0", "HOME=/h",
                                              "STRATA_OWNER_ATTEST_KEY=aa", "STRATA_DS4_SLOT_DIR=/l/ds4-slots", "AIBOX_STRATA_LOG=/l/x.log"};
        const uint64_t h0 = args_env_hash(cmd, env);
        auto h = [&](const std::string& what, std::vector<std::string> c, std::vector<std::string> e, bool same) {
            const bool got = args_env_hash(c, e) == h0; const bool ok = got == same; ++na; abad += !ok;
            std::printf("args_env %-44s %s %s\n", what.c_str(), ok ? "OK  " : "DIFF", got ? "same hash" : "another hash");
        };
        auto with = [&](std::vector<std::string> e, size_t i, const std::string& kv) { e[i] = kv; return e; };
        std::vector<std::string> perm(env.rbegin(), env.rend());
        h("the environment in another order", cmd, perm, true);
        h("a STRATA_ value changed", cmd, with(env, 1, "STRATA_PREFILL_RING=128"), false);
        h("a GPU_ value changed", cmd, with(env, 2, "GPU_MAX_HW_QUEUES=4"), false);
        h("an HSA_ value changed", cmd, with(env, 3, "HSA_ENABLE_SDMA=1"), false);
        { auto e = env; e.push_back("STRATA_PREFILL_CPU_SHARE_MAX=6145"); h("a STRATA_ variable added (the key)", cmd, e, false); }
        h("PATH changed (not hashed)", cmd, with(env, 0, "PATH=/usr/bin"), true);
        h("AIBOX_STRATA_LOG changed (not a hashed prefix)", cmd, with(env, 7, "AIBOX_STRATA_LOG=/m/y.log"), true);
        h("STRATA_OWNER_ATTEST_KEY changed (skipped)", cmd, with(env, 5, "STRATA_OWNER_ATTEST_KEY=bb"), true);
        h("STRATA_DS4_SLOT_DIR changed (skipped)", cmd, with(env, 6, "STRATA_DS4_SLOT_DIR=/m/ds4-slots"), true);
        { auto c = cmd; c[0] = "/y/strata"; h("argv[0] changed (the exe= field)", c, env, true); }
        { auto c = cmd; c[5] = "3"; h("an argument changed", c, env, false); }
        { auto c = cmd; c.push_back("--planner"); c.push_back("/t.txt"); h("an argument added", c, env, false); }
        { std::vector<std::string> c = {"/x/strata", "--serve", "--spec", "4", "--pack", "/p"}; h("the arguments in another order", c, env, false); }
        { std::vector<std::string> c = {"/x/strata", "--serve", "--pack/p", "--spec", "4"}; h("two arguments joined", c, env, false); }
        const std::vector<std::string> self = nul_list("/proc/self/cmdline");
        ++na; abad += self.empty() || self.size() != (size_t) argc;
        std::printf("args_env %-44s %s %zu strings (argc %d)\n", "nul_list(/proc/self/cmdline)", !self.empty() && self.size() == (size_t) argc ? "OK  " : "DIFF", self.size(), argc);
    }
    // the derived-X cache: write, hit, and every miss (no file, garbage, truncated, checksum, another key, M1 moved, out of range)
    {
        const char* td = std::getenv("TMPDIR");
        const std::string path = std::string(td && *td ? td : "/tmp") + "/planner_c_test." + std::to_string((long long) getpid()) + ".x-cache";
        const Pack& p = packs.at("iq4");
        const std::string key = "host=h exe=0123 pack=4567 profile=caa0c5d52cc0b041 gpu=0000:03:00.0 sdma=default slots=7492 bytes=1 context=262144 chunk=16384 "
                                "args_env=89abcdef01234567 boot=0e5f6c1a-1111-2222-3333-444455556666 kernel=7.0.0-42-generic vram_free_mib32=320";
        CacheEntry e; e.X = 6145; e.start_ms = 1885.4; e.bw = 13.675e9; e.reads = 8;
        auto c = [&](const std::string& what, const std::string& k, double bw, bool want, const std::string& must = "") {
            CacheEntry g; std::string why; const bool hit = cache_read(path, k, bw, kCacheBwTol, p, g, why);
            const bool ok = hit == want && (!hit || g.X == 6145) && (must.empty() || why.find(must) != std::string::npos); ++na; abad += !ok;
            std::printf("cache %-44s %s %s%s\n", what.c_str(), ok ? "OK  " : "DIFF", (hit ? "hit X " + std::to_string(g.X) : std::string("miss: ")).c_str(), why.c_str());
        };
        auto put = [&](const std::string& text) { std::FILE* f = std::fopen(path.c_str(), "wb"); std::fwrite(text.data(), 1, text.size(), f); std::fclose(f); };
        auto get = [&]() { std::ifstream f(path, std::ios::binary); return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); };
        std::remove(path.c_str());
        c("no file", key, e.bw, false);
        std::string err;
        const bool w = cache_write(path, key, e, "t", err); ++na; abad += !w; std::printf("cache %-44s %s %s\n", "write", w ? "OK  " : "DIFF", err.c_str());
        { const std::string g0 = get(); const bool v2 = g0.rfind("strata-planner-x-cache 2\n", 0) == 0; ++na; abad += !v2;
          std::printf("cache %-44s %s\n", "the record is format 2", v2 ? "OK  " : "DIFF"); }
        c("hit", key, e.bw, true);
        // review 15 D5: the hit tolerance is 0.25% (was 0.5%)
        c("hit, M1 +0.24%", key, e.bw * 1.0024, true);
        c("hit, M1 -0.24%", key, e.bw * 0.9976, true);
        c("M1 +0.26%", key, e.bw * 1.0026, false, "GB/s stored");
        c("M1 -0.26%", key, e.bw * 0.9974, false, "GB/s stored");
        c("M1 +0.4% (a hit under the old 0.5%)", key, e.bw * 1.004, false, "GB/s stored");
        c("M1 +0.6%", key, e.bw * 1.006, false);
        // review 15 D1/D2/D3: each new key field, changed alone, is a miss that names it
        for (const char* f : {"profile", "args_env", "boot", "kernel"}) {
            std::string k3 = key; const size_t i = k3.find(std::string(" ") + f + "=") + 1, j = k3.find(' ', i);
            k3.replace(i, j - i, std::string(f) + "=" + (std::string(f) == "kernel" ? "7.0.0-43-generic" : std::string(f) == "boot" ? "ffffffff-0000-0000-0000-000000000000" : "0000000000000001"));
            c(std::string("another ") + f, k3, e.bw, false, std::string("another key (") + f + " ");
        }
        { std::string k4 = key; const size_t i = k4.find(" boot=") + 6, j = k4.find(' ', i); k4.replace(i, j - i, "");
          c("boot id absent (empty field) vs stored", k4, e.bw, false, "another key (boot "); }
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
        {   // review 15: a version-1 record (the 242bf4326 format, its own valid checksum) is a miss, not an error
            const std::string body = "strata-planner-x-cache 1\nkey host=h exe=0123 pack=4567 gpu=0000:03:00.0 sdma=default slots=7492 bytes=1 context=262144 "
                                     "chunk=16384 vram_free_mib32=320\nx 6145 start 1885.4000 bw 13675000000.000000 reads 8\n";
            char ck[64]; std::snprintf(ck, sizeof ck, "check %016llx\n", (unsigned long long) fnv_str(body));
            put(body + ck); c("a version-1 record", key, e.bw, false, "format 1");
        }
        // a record whose key has no boot / kernel (written on a host without them) against one that has them
        { cache_write(path, "host=h boot= kernel=", e, "t", err); c("a record with empty boot/kernel, now set", "host=h boot=b kernel=k", e.bw, false, "another key (boot "); }
        { cache_write(path, "host=h boot= kernel=", e, "t", err); c("empty boot/kernel both times: hit", "host=h boot= kernel=", e.bw, true); }
        std::remove(path.c_str());
    }
    // review 15 D5: stable_X - the record is written only when start +-10 ms and M1 +-0.25% give the same X
    for (const char* q : {"iq3", "iq4", "q4"}) {
        if (!sample.count(q)) continue;
        const Pack& p = packs.at(q); const Facts& f = sample[q].first;
        const Result r = derive(p, f, sample[q].second, configured.at(q));
        std::string why; const bool st = stable_X(p, f, r, configured.at(q), why);
        // the margin to the nearest step of X in start (1 ms resolution): stable needs more than 10 ms both ways
        double up = 0, dn = 0;
        while (up < 200 && derive_from_start(p, f, r.start_ms + up + 1, configured.at(q)).X == r.X) up += 1;
        while (dn < 200 && derive_from_start(p, f, r.start_ms - dn - 1, configured.at(q)).X == r.X) dn += 1;
        std::printf("stable   %-4s %s start %.1f ms X %lld: %s%s (X holds for +%.0f / -%.0f ms)\n", q, sample_tag[q].c_str(), r.start_ms, (long long) r.X,
                    st ? "stable" : "unstable: ", why.c_str(), up, dn);
        if (std::string(q) == "iq4") { ++na; abad += !st; std::printf("stable   %-44s %s\n", "iq4's sample is stable (written)", st ? "OK  " : "DIFF"); }
        // a start 5 ms below X's next step up: +10 ms moves X, so the record is NOT written
        if (up < 200) {
            Result u = derive_from_start(p, f, r.start_ms + up + 1 - 5, configured.at(q));
            std::string w; const bool su = stable_X(p, f, u, configured.at(q), w);
            const bool ok = !su && w.find("start +10 ms") != std::string::npos; ++na; abad += !ok;
            std::printf("stable   %-44s %s %s%s\n", (std::string(q) + " 5 ms below a step: not cached").c_str(), ok ? "OK  " : "DIFF", su ? "stable" : "unstable: ", w.c_str());
        }
        if (dn < 200) {
            Result u = derive_from_start(p, f, r.start_ms - dn - 1 + 5, configured.at(q));
            std::string w; const bool su = stable_X(p, f, u, configured.at(q), w);
            const bool ok = !su && w.find("start -10 ms") != std::string::npos; ++na; abad += !ok;
            std::printf("stable   %-44s %s %s%s\n", (std::string(q) + " 5 ms above a step: not cached").c_str(), ok ? "OK  " : "DIFF", su ? "stable" : "unstable: ", w.c_str());
        }
        // M1: the factor that moves X (scan in 0.05% steps up to 2%); within 0.25% of it the record is not written
        for (int sgn : {+1, -1}) {
            int k = 1; Facts g = f;
            for (; k <= 40; ++k) { g.bw = f.bw * (1 + sgn * 0.0005 * k); if (derive_from_start(p, g, r.start_ms, configured.at(q)).X != r.X) break; }
            if (k > 40) continue;
            Facts h = f; h.bw = f.bw * (1 + sgn * (0.0005 * k - 0.0015));   // 0.15% short of the step: M1 +-0.25% crosses it
            Result u = derive_from_start(p, h, r.start_ms, configured.at(q));
            if (u.X != r.X) continue;
            // M1 * (1 +- 0.25%) alone moves X here; start +-10 ms is checked first and is usually the wider move, so the reason may name it
            Facts m = h; m.bw = h.bw * (1 + sgn * kCacheBwTol);
            const bool m1_moves = derive_from_start(p, m, u.start_ms, configured.at(q)).X != u.X;
            std::string w; const bool su = stable_X(p, h, u, configured.at(q), w);
            const bool ok = !su && m1_moves; ++na; abad += !ok;
            std::printf("stable   %-44s %s %s%s\n", (std::string(q) + (sgn > 0 ? " M1 0.15% below a step" : " M1 0.15% above a step")).c_str(), ok ? "OK  " : "DIFF",
                        su ? "stable" : "unstable: ", w.c_str());
        }
    }
    {   // no X: never written
        Result none; none.why = "no probes"; std::string w; const bool su = stable_X(packs.at("iq4"), sample.count("iq4") ? sample["iq4"].first : Facts(), none, 6145, w);
        ++na; abad += su; std::printf("stable   %-44s %s %s\n", "no X: not cached", !su ? "OK  " : "DIFF", w.c_str());
    }
    // ---- review 16 D8/D9 (option e): the bracket, on Fable's harness cases (results/planner-20261112/next/buildC/fable16: fix.cpp, fix.out,
    // reads.txt).  The same selection and facts as fix.cpp: the engine loads with 8 counted reads and an X (apply1-* left out: the
    // superseded first build), each on its quant's builder-vector facts with the load's own slots, bytes and M1; plus the stored vectors
    // with 8 reads at production slots.  Every number checked here is fix.out's.
    int nb = 0, bbad = 0;
    auto bchk = [&](const std::string& what, bool ok, const std::string& detail) {
        ++nb; bbad += !ok;
        std::printf("bracket  %-52s %s %s\n", what.c_str(), ok ? "OK  " : "DIFF", detail.c_str());
    };
    {
        std::map<std::string, Facts> base;
        for (const Vec& v : allvec) if (v.tag == "builder-" + v.q) base[v.q] = v.f;
        struct Load { std::string tag, q, build; int64_t slots = 0; double bytes = 0, m1 = 0; int64_t X = -1; int spec = 4; std::vector<double> r; bool vec = false; };
        std::vector<Load> L;
        {
            std::ifstream in(argc > 5 ? std::string(argv[5]) : std::string("src/program/planner_c_test_reads.txt")); std::string ln;
            int nlines = 0;
            while (std::getline(in, ln)) {
                if (ln.empty() || ln[0] == '#') continue;
                ++nlines;
                std::istringstream a(ln); Load l; int64_t conf; double st; std::string spec;
                a >> l.tag >> l.q >> l.slots >> l.bytes >> l.m1 >> l.X >> conf >> st >> spec >> l.build;
                l.spec = spec == "?" ? 4 : std::stoi(spec);
                double v; while (a >> v) l.r.push_back(v);
                if (l.r.size() != 8 || l.X <= 0) continue;
                if (l.build == "build-hip-apply" && l.tag.rfind("apply1-", 0) == 0) continue;   // fix.cpp: the first apply build's single loads
                L.push_back(l);
            }
            bchk("the read series file", nlines == 131 && L.size() == 127, std::to_string(nlines) + " lines, " + std::to_string(L.size()) + " engine loads (want 131, 127)");
        }
        for (const Vec& v : allvec)
            if (base.count(v.q) && v.reads.size() == 8 && v.f.slots == base[v.q].slots && v.tag.find("outlier") == std::string::npos && v.tag.find("7reads") == std::string::npos) {
                Load l; l.tag = "vec:" + v.tag; l.q = v.q; l.slots = v.f.slots; l.bytes = v.f.cache_bytes; l.m1 = v.f.bw / 1e9; l.X = v.X; l.r = v.reads; l.vec = true; L.push_back(l);
            }
        auto facts_of = [&](const Load& l) { Facts f = base.at(l.q); f.slots = l.slots; f.cache_bytes = l.bytes; f.bw = l.m1 * 1e9; return f; };
        // fix.out's totals, per group: n, today's median X = table X (S0), bracket X = table X (S2), X_med != X_lo, cached by B (stable_X), cached now
        struct Tot { int n = 0, par = 0, s0 = 0, s2 = 0, open = 0, cacheB = 0, cacheNew = 0; };
        std::map<std::string, Tot> T;
        int parity = 0, nengine = 0, clean = 0, clean_same = 0, ref_moved = 0;
        std::string ref_moved_tags;
        for (const Load& l : L) {
            const Pack& p = packs.at(l.q); const Facts f = facts_of(l); const int64_t tx = table_X.at(l.q);
            const Result r = derive(p, f, l.r, -1);
            const Bracket b = bracket(p, f, l.r, r);
            {   // review 17 F1: a load whose X_ref differs between the table's ref_start and the builder median (C's reference)
                const Result ob = derive_from_start(p, f, median(p.text[0].builder_ms.at((int64_t) (p.start - 7))), -1);
                if ((ob.derived ? ob.X : -1) != b.X_ref) { ++ref_moved; ref_moved_tags += " " + l.tag + " (" + std::to_string(ob.derived ? ob.X : -1) + " -> " + std::to_string(b.X_ref) + ")"; }
            }
            std::string why; const bool st = stable_X(p, f, r, -1, why);
            const bool cache_new = st && b.closed();   // the engine's write condition (generate.cpp: closed, then stable_X)
            const bool prod = !l.vec && l.spec != 2 && l.slots == base.at(l.q).slots;
            const std::string grp = std::string(l.vec ? "vector:" : "engine:") + l.q + (l.spec == 2 ? ":spec2" : "") + (l.slots != base.at(l.q).slots ? ":otherslots" : "");
            Tot& t = T[grp];
            ++t.n; t.par += r.X == l.X; t.s0 += r.X == tx; t.s2 += b.X == tx; t.open += b.X_med != b.X_lo; t.cacheB += st; t.cacheNew += cache_new;
            if (!l.vec) { ++nengine; parity += r.derived && r.X == l.X; }
            // clean: X_med is the table X (the 124 engine loads at production slots with MTP, and the 60 vectors): nothing changes
            if ((prod || l.vec) && r.X == tx) { ++clean; clean_same += b.on && b.X == r.X && cache_new == st; }
            if (!b.on || b.X != tx || !b.closed() || !st)
                std::printf("bracket  %-44s %s med %.1f lo2 %.1f | X_med %lld X_lo %lld X_ref %lld -> X %lld %s | B %s | now %s\n", l.tag.c_str(), l.q.c_str(), r.start_ms, b.s_lo,
                            (long long) b.X_med, (long long) b.X_lo, (long long) b.X_ref, (long long) b.X, b.closed() ? "closed" : "OPEN",
                            st ? "cached" : ("not cached: " + why).c_str(), cache_new ? "cached" : ("not cached: " + (b.closed() ? "unstable: " + why : b.open_why())).c_str());
            // the named cases of review 16
            if (l.tag == "pilot/keyabsent/runs/q4-03-absent") {
                bchk("q4-03: X_med 8449 X_lo 8705 X_ref 8705 -> 8705, open", b.on && b.X_med == 8449 && b.X_lo == 8705 && b.X_ref == 8705 && b.X == 8705 && !b.closed(),
                     "X " + std::to_string(b.X) + " " + b.open_why());
                bchk("q4-03: not cached", !cache_new, b.open_why());
                int lvl = 0; std::string note;
                int64_t lim = apply_limit(&p, f, &r, -1, 3072, lvl, note, &b);
                bchk("q4-03 key absent: level 2, 8705", lim == 8705 && lvl == 2, std::to_string(lim) + " level " + std::to_string(lvl));
                lim = apply_limit(&p, f, &r, -1, 3072, lvl, note);
                bchk("q4-03 key absent, no bracket (build B): 8449", lim == 8449 && lvl == 2, std::to_string(lim) + " level " + std::to_string(lvl));
                lim = apply_limit(&p, f, &r, 8705, 3072, lvl, note, &b);
                bchk("q4-03 key 8705: max(configured, X) 8705", lim == 8705 && lvl == 1, std::to_string(lim) + " level " + std::to_string(lvl));
                lim = apply_limit(&p, f, &r, 6145, 3072, lvl, note, &b);
                bchk("q4-03 key 6145: max(configured, X) 8705", lim == 8705 && lvl == 1, std::to_string(lim) + " level " + std::to_string(lvl));
                lim = apply_limit(&p, f, &r, 8961, 3072, lvl, note, &b);
                bchk("q4-03 key 8961: max(configured, X) 8961", lim == 8961 && lvl == 1, std::to_string(lim) + " level " + std::to_string(lvl));
            }
            if (l.tag == "pilot/item5/dec/min-cand-4")
                bchk("min-cand-4 (--spec 2): 5889 / 6145 / 6145 -> 6145, open", b.X_med == 5889 && b.X_lo == 6145 && b.X_ref == 6145 && b.X == 6145 && !cache_new, "X " + std::to_string(b.X));
            if (l.tag == "pilot/item5/dec/min-cand-1")
                bchk("min-cand-1 (--spec 2): 5889 / 5889 / 6145 -> 5889, open", b.X_med == 5889 && b.X_lo == 5889 && b.X_ref == 6145 && b.X == 5889 && !cache_new, "X " + std::to_string(b.X));
        }
        bchk("the logged X = X_med (derive) on every engine load", parity == 127 && nengine == 127, std::to_string(parity) + " of " + std::to_string(nengine) + " (want 127 of 127)");
        std::printf("bracket  X_ref differs from C's builder-median reference on %d of %zu loads:%s\n", ref_moved, L.size(), ref_moved_tags.c_str());
        bchk("clean loads: the limit and the cache decision unchanged", clean == 184 && clean_same == 184, std::to_string(clean_same) + " of " + std::to_string(clean) + " (want 184 of 184)");
        // fix.out's totals table (S0 / S2 / bracket open / cache B / cache new)
        const std::map<std::string, std::vector<int>> want = {
            {"engine:iq3", {34, 34, 34, 34, 0, 30, 30}}, {"engine:iq4", {57, 57, 57, 57, 0, 57, 57}}, {"engine:iq4:spec2:otherslots", {2, 2, 0, 1, 1, 0, 0}},
            {"engine:q4", {34, 34, 33, 34, 1, 33, 33}}, {"vector:iq3", {20, 20, 20, 20, 0, 20, 20}}, {"vector:iq4", {20, 20, 20, 20, 0, 20, 20}},
            {"vector:q4", {20, 20, 20, 20, 0, 18, 18}}};
        for (const auto& [g, w] : want) {
            const Tot t = T.count(g) ? T.at(g) : Tot();
            const std::vector<int> got = {t.n, t.par, t.s0, t.s2, t.open, t.cacheB, t.cacheNew};
            char d[200]; std::snprintf(d, sizeof d, "n %d parity %d | S0 %d S2 %d | open %d | cache B %d new %d", t.n, t.par, t.s0, t.s2, t.open, t.cacheB, t.cacheNew);
            bchk("totals " + g, got == w, d);
        }
        bchk("no other group", T.size() == want.size(), std::to_string(T.size()) + " groups");
        // contamination (fix.cpp sections 2 and 3): every clean engine load, every subset of k of its 5 settled reads slowed (or made faster)
        // by D ms.  X_med and X_lo through a per-load memo of derive_from_start (the starts are reads or reads +- D); the combination is the
        // engine's bracket_combine; bracket() itself is checked on the k = 3 subset {4,5,6} of every load against the memo + combination.
        const char* want_lines[] = {
            "D    20 ms iq3 (34 clean loads):  k=1 0|0|0 of 170  k=2 0|0|0 of 340  k=3 49|0|0 of 340  k=4 83|5|5 of 170  k=5 34|5|5 of 34",
            "D    20 ms iq4 (57 clean loads):  k=1 0|0|0 of 285  k=2 0|0|0 of 570  k=3 9|0|0 of 570  k=4 18|0|0 of 285  k=5 9|0|0 of 57",
            "D    20 ms q4 (33 clean loads):  k=1 0|0|0 of 165  k=2 0|0|0 of 330  k=3 0|0|0 of 330  k=4 0|0|0 of 165  k=5 0|0|0 of 33",
            "D    40 ms iq3 (34 clean loads):  k=1 0|0|0 of 170  k=2 0|0|0 of 340  k=3 340|0|0 of 340  k=4 170|170|170 of 170  k=5 34|34|34 of 34",
            "D    40 ms iq4 (57 clean loads):  k=1 0|0|0 of 285  k=2 0|0|0 of 570  k=3 396|0|0 of 570  k=4 285|169|169 of 285  k=5 57|57|57 of 57",
            "D    40 ms q4 (33 clean loads):  k=1 0|0|0 of 165  k=2 11|0|0 of 330  k=3 109|5|5 of 330  k=4 101|36|36 of 165  k=5 27|15|15 of 33",
            "D    80 ms iq3 (34 clean loads):  k=1 0|0|0 of 170  k=2 0|0|0 of 340  k=3 340|0|0 of 340  k=4 170|170|170 of 170  k=5 34|34|34 of 34",
            "D    80 ms iq4 (57 clean loads):  k=1 0|0|0 of 285  k=2 0|0|0 of 570  k=3 570|0|0 of 570  k=4 285|285|285 of 285  k=5 57|57|57 of 57",
            "D    80 ms q4 (33 clean loads):  k=1 0|0|0 of 165  k=2 24|0|0 of 330  k=3 330|16|16 of 330  k=4 165|165|165 of 165  k=5 33|33|33 of 33",
            "D   150 ms iq3 (34 clean loads):  k=1 0|0|0 of 170  k=2 0|0|0 of 340  k=3 340|0|0 of 340  k=4 170|170|170 of 170  k=5 34|34|34 of 34",
            "D   150 ms iq4 (57 clean loads):  k=1 0|0|0 of 285  k=2 0|0|0 of 570  k=3 570|0|0 of 570  k=4 285|285|285 of 285  k=5 57|57|57 of 57",
            "D   150 ms q4 (33 clean loads):  k=1 0|0|0 of 165  k=2 24|0|0 of 330  k=3 330|16|16 of 330  k=4 165|165|165 of 165  k=5 33|33|33 of 33",
            "fast D    10 ms iq3 (34 clean loads):  k=1 0|0|0 of 170  k=2 0|0|0 of 340  k=3 0|0|0 of 340  k=4 0|0|0 of 170  k=5 0|0|0 of 34",
            "fast D    10 ms iq4 (57 clean loads):  k=1 0|0|0 of 285  k=2 0|0|0 of 570  k=3 0|0|0 of 570  k=4 0|0|0 of 285  k=5 0|0|0 of 57",
            "fast D    10 ms q4 (33 clean loads):  k=1 0|0|0 of 165  k=2 0|0|0 of 330  k=3 0|0|0 of 330  k=4 0|0|0 of 165  k=5 0|0|0 of 33",
            "fast D    20 ms iq3 (34 clean loads):  k=1 0|0|0 of 170  k=2 0|0|0 of 340  k=3 0|0|0 of 340  k=4 0|0|0 of 170  k=5 0|0|0 of 34",
            "fast D    20 ms iq4 (57 clean loads):  k=1 0|0|0 of 285  k=2 0|0|0 of 570  k=3 0|0|0 of 570  k=4 0|0|0 of 285  k=5 0|0|0 of 57",
            "fast D    20 ms q4 (33 clean loads):  k=1 0|0|0 of 165  k=2 0|10|0 of 330  k=3 2|26|2 of 330  k=4 4|22|4 of 165  k=5 2|6|2 of 33",
            "fast D    30 ms iq3 (34 clean loads):  k=1 0|0|0 of 170  k=2 0|0|0 of 340  k=3 0|0|0 of 340  k=4 0|0|0 of 170  k=5 0|0|0 of 34",
            "fast D    30 ms iq4 (57 clean loads):  k=1 0|0|0 of 285  k=2 0|0|0 of 570  k=3 0|0|0 of 570  k=4 0|0|0 of 285  k=5 0|0|0 of 57",
            "fast D    30 ms q4 (33 clean loads):  k=1 0|0|0 of 165  k=2 0|94|0 of 330  k=3 36|210|36 of 330  k=4 63|147|63 of 165  k=5 27|31|27 of 33"};
        size_t wi = 0; int full_ok = 0, full_n = 0;
        for (int fast = 0; fast < 2; ++fast)
            for (double D : fast ? std::vector<double>{10, 20, 30} : std::vector<double>{20, 40, 80, 150})
                for (const char* q : {"iq3", "iq4", "q4"}) {
                    int n = 0, f0[6] = {}, f1[6] = {}, f2[6] = {}, tot[6] = {};
                    const Pack& p = packs.at(q); const int64_t tx = table_X.at(q);
                    for (const Load& l : L) {
                        if (l.vec || l.q != q || l.spec == 2 || l.slots != base.at(q).slots) continue;
                        const Facts f = facts_of(l);
                        std::map<double, int64_t> memo;
                        auto X_at = [&](double s) { auto it = memo.find(s); if (it != memo.end()) return it->second;
                                                    const Result r = derive_from_start(p, f, s, -1); return memo[s] = r.derived ? r.X : -1; };
                        double sm = 0; int u = 0; start_read(l.r, sm, u);
                        if (X_at(sm) != tx) continue;
                        ++n; int64_t Xr = -1; ref_X(p, f, Xr);
                        for (int m = 1; m < 32; ++m) {
                            std::vector<double> r = l.r; int k = 0;
                            for (int i = 0; i < 5; ++i) if (m >> i & 1) { r[(size_t) 3 + i] += fast ? -D : D; ++k; }
                            double s_med = 0, s_lo = 0; start_read(r, s_med, u); start_read_lo(r, s_lo);
                            Bracket b; b.X_med = X_at(s_med); b.X_lo = X_at(s_lo); b.X_ref = Xr;
                            if (b.X_med < 0) { ++tot[k]; ++f0[k]; ++f2[k]; f1[k] += b.X_lo != tx; continue; }   // (no X from the median: never here)
                            bracket_combine(b);
                            ++tot[k]; f0[k] += b.X_med != tx; f1[k] += b.X_lo != tx; f2[k] += b.X != tx;
                            if (m == 7) {   // bracket() itself, end to end, on this subset
                                const Result rr = derive(p, f, r, -1); const Bracket bb = bracket(p, f, r, rr);
                                ++full_n; full_ok += bb.on && bb.X == b.X && bb.X_med == b.X_med && bb.X_lo == b.X_lo && bb.X_ref == b.X_ref && bb.closed() == b.closed();
                            }
                        }
                    }
                    char ln[400]; int o = std::snprintf(ln, sizeof ln, "%sD %5.0f ms %s (%d clean loads):", fast ? "fast " : "", D, q, n);
                    for (int k = 1; k <= 5; ++k) o += std::snprintf(ln + o, sizeof ln - (size_t) o, "  k=%d %d|%d|%d of %d", k, f0[k], f1[k], f2[k], tot[k]);
                    const bool ok = wi < sizeof want_lines / sizeof *want_lines && std::string(ln) == want_lines[wi]; ++wi;
                    ++nb; bbad += !ok;
                    std::printf("contam   %s %s\n", ok ? "OK  " : "DIFF", ln);
                    if (!ok && wi <= sizeof want_lines / sizeof *want_lines) std::printf("contam   want %s\n", want_lines[wi - 1]);
                }
        bchk("bracket() = memo + bracket_combine on the k=3 subsets", full_n > 0 && full_ok == full_n, std::to_string(full_ok) + " of " + std::to_string(full_n));
    }
    // the edge cases (review 16 section 4) on iq4's / iq3's sample facts
    if (sample.count("iq4") && sample.count("iq3")) {
        const Pack& p = packs.at("iq4"); const Facts& f = sample["iq4"].first; const std::vector<double>& reads = sample["iq4"].second;
        int lvl = 0; std::string note; int64_t lim;
        {   // no X from the median (7 reads): not bracketed, today's paths (level 1 configured, level 3 the reference)
            const Result r7 = derive(p, f, std::vector<double>(reads.begin(), reads.begin() + 7), 6145);
            const Bracket b = bracket(p, f, std::vector<double>(reads.begin(), reads.begin() + 7), r7);
            bchk("edge: 7 reads -> not bracketed", !b.on && !b.closed(), b.open_why());
            lim = apply_limit(&p, f, &r7, 6145, 3072, lvl, note, &b); bchk("edge: 7 reads, key: configured (level 1)", lim == 6145 && lvl == 1, std::to_string(lim) + " " + note);
            lim = apply_limit(&p, f, &r7, -1, 3072, lvl, note, &b);   bchk("edge: 7 reads, no key: the reference (level 3)", lim == 6145 && lvl == 3, std::to_string(lim) + " " + note);
            Result rg = derive_from_start(packs.at("iq3"), sample["iq3"].first, 1500.0, -1);   // below IQ3's regime guard: no X
            std::vector<double> rr(8, 1500.0);
            const Bracket bg = bracket(packs.at("iq3"), sample["iq3"].first, rr, rg);
            bchk("edge: regime guard (no X from the median) -> not bracketed", !rg.derived && !bg.on, rg.why);
        }
        {   // no X_ref: X = X_med, not cached.  (A pack without the reference read cannot derive at all - TextModel reads the same builder
            // median - so on a real load X_ref is missing only when the rule gives no X at the reference: the combination is checked alone)
            const Result r = derive(p, f, reads, -1);
            Bracket b; b.X_med = r.X; b.X_lo = r.X; b.X_ref = -1; bracket_combine(b);
            bchk("edge: no X_ref -> X = X_med, open", b.on && b.X == r.X && !b.closed() && !b.note.empty(), "X " + std::to_string(b.X) + " (" + b.note + ") " + b.open_why());
            b.X_lo = r.X + 256; bracket_combine(b);
            bchk("edge: no X_ref, X_lo above -> X = X_med", b.X == r.X && !b.closed(), "X " + std::to_string(b.X));
            lim = apply_limit(&p, f, &r, -1, 3072, lvl, note, &b);   bchk("edge: no X_ref, no key: X_med (level 2)", lim == r.X && lvl == 2, std::to_string(lim));
            Facts nobw = f; nobw.bw = 0; int64_t xr = -1;
            bchk("edge: no link rate -> no X_ref", !ref_X(p, nobw, xr) && xr == -1, "");
        }
        {   // no X from the 2nd-smallest (IQ3, two settled reads below the regime guard): X = X_med, not cached
            const Pack& p3 = packs.at("iq3"); const Facts& f3 = sample["iq3"].first; std::vector<double> r3 = sample["iq3"].second;
            r3[3] = 1500.0; r3[4] = 1500.0;
            const Result r = derive(p3, f3, r3, -1);
            const Bracket b = bracket(p3, f3, r3, r);
            bchk("edge: no X_lo (2 reads below IQ3's guard) -> X = X_med, open", r.derived && b.on && b.X_lo == -1 && b.X == r.X && !b.closed(),
                 "X_med " + std::to_string(b.X_med) + " X " + std::to_string(b.X) + " (" + b.note + ")");
        }
        {   // the combination: X_ref inside, below and above the range; an inverted pair (X_lo < X_med) takes min/max
            auto comb = [&](int64_t m, int64_t lo, int64_t ref, int64_t want, const char* what) {
                Bracket b; b.X_med = m; b.X_lo = lo; b.X_ref = ref; bracket_combine(b);
                bchk(std::string("combine: ") + what, b.X == want, std::to_string(m) + " " + std::to_string(lo) + " ref " + std::to_string(ref) + " -> " + std::to_string(b.X));
            };
            comb(8449, 8705, 8705, 8705, "ref at the top of the range");
            comb(8449, 8961, 8705, 8705, "ref inside the range");
            comb(8449, 8705, 9217, 8705, "ref above the range: X_lo");
            comb(8449, 8705, 8193, 8449, "ref below the range: X_med");
            comb(6145, 6145, 5889, 6145, "closed pair, ref lower: X_med");
            comb(4865, 4609, 4097, 4609, "inverted pair (X_lo < X_med): min/max");
            comb(4865, 4609, 5121, 4865, "inverted pair, ref above: max");
            Bracket c; c.X_med = c.X_lo = c.X_ref = 6145; bracket_combine(c);
            bchk("combine: all three equal -> closed", c.closed() && c.open_why().empty(), "");
        }
        {   // a cache hit (no bracket: off) applies the stored X; bracket off falls back to r.X
            Result hit; hit.derived = true; hit.X = 6145;
            const Bracket off;
            lim = apply_limit(&p, f, &hit, -1, 3072, lvl, note, &off); bchk("cache hit, bracket off: the stored X (level 2)", lim == 6145 && lvl == 2, std::to_string(lim));
            lim = apply_limit(&p, f, &hit, -1, 3072, lvl, note);       bchk("cache hit, no bracket: the stored X (level 2)", lim == 6145 && lvl == 2, std::to_string(lim));
        }
        {   // vmstat_read (S9)
            long long a = -1, b = -1; const bool ok = vmstat_read(a, b);
            bchk("vmstat_read: pswpin and pgmajfault", ok && a >= 0 && b >= 0, std::to_string(a) + " " + std::to_string(b));
        }
    } else bchk("edge cases: the iq3/iq4 samples", false, "missing");
    // ---- review 16 D10 / review 17 F1: the table's ref_start line (the engine's own statistic) is level 3's input and X_ref's ----
    {
        std::map<std::string, Facts> base;
        for (const Vec& v : allvec) if (v.tag == "builder-" + v.q) base[v.q] = v.f;
        const std::map<std::string, double> want_ref = {{"iq3", 1657.1}, {"iq4", 1885.7}, {"q4", 2426.3}};   // apply20 means (tally.txt)
        const std::map<std::string, double> m1mean = {{"iq3", 13.5870}, {"iq4", 13.6721}, {"q4", 13.7771}};   // the engine loads' mean M1 (fix.cpp)
        for (const char* q : {"iq3", "iq4", "q4"}) {
            const Pack& p = packs.at(q); const int64_t tx = table_X.at(q); const std::string Q = q;
            double s = 0; const bool has = ref_start(p, s);
            const bool line = p.ref_start_ms > 0;
            if (!line) { bchk(Q + " the table has no ref_start line (an older table): skipped", true, ""); continue; }
            bchk(Q + " ref_start() = the table's line", has && s == want_ref.at(q), std::to_string(s));
            Pack nb = p; nb.ref_start_ms = 0; double sb = 0; ref_start(nb, sb);
            bchk(Q + " no line: ref_start() = the builder median", sb == median(p.text[0].builder_ms.at((int64_t) (p.start - 7))), std::to_string(sb));
            for (double fct : {1.0, 1.0025, 0.9975}) {
                if (!base.count(q)) { bchk(Q + " base facts", false, "missing"); break; }
                Facts f = base.at(q); f.bw = m1mean.at(q) * 1e9 * fct;
                char w[96]; std::snprintf(w, sizeof w, "%s level 3 at M1 %+.2f%% (ref_start)", q, (fct - 1) * 100);
                Result none; none.why = "no probes"; int lvl = 0; std::string note;
                const int64_t lim = apply_limit(&p, f, &none, -1, 3072, lvl, note);
                const int64_t old = derive_from_start(p, f, sb, -1).X;   // what C's builder-median reference gives here (reported)
                bchk(w, lim == tx && lvl == 3, std::to_string(lim) + " level " + std::to_string(lvl) + " (builder median: " + std::to_string(old) + ")");
                int64_t xr = -1; ref_X(p, f, xr);
                bchk(std::string(w) + ": X_ref", xr == tx, std::to_string(xr));
            }
            {   // the margins from ref_start to X's steps at the mean M1 (0.1 ms scan), reported
                Facts f = base.at(q); f.bw = m1mean.at(q) * 1e9;
                double dn = 0, up = 0;
                while (dn < 200 && derive_from_start(p, f, s - dn - 0.1, -1).X == tx) dn += 0.1;
                while (up < 200 && derive_from_start(p, f, s + up + 0.1, -1).X == tx) up += 0.1;
                char d[160]; std::snprintf(d, sizeof d, "ref %.1f ms: X %lld holds for -%.1f / +%.1f ms at M1 %.4f", s, (long long) tx, dn, up, f.bw / 1e9);
                bchk(Q + " margins of the reference (> 10 ms both ways)", dn > 10 && up > 10, d);
            }
        }
        // load_pack: the line parses, a bad one is refused
        {
            const char* td0 = std::getenv("TMPDIR");
            const std::string tp = std::string(td0 && *td0 ? td0 : "/tmp") + "/planner_c_test." + std::to_string((long long) getpid()) + ".ref.txt";
            auto with_line = [&](const std::string& line) {
                std::ifstream in(pd + "/qwen-iq4.txt"); std::ofstream out(tp); std::string ln; bool put = false;
                while (std::getline(in, ln)) { if (ln.rfind("ref_start ", 0) == 0) continue; out << ln << "\n"; if (!put && ln.rfind("o_s ", 0) == 0) { if (!line.empty()) out << line << "\n"; put = true; } }
            };
            for (const auto& [line, ok, val] : std::vector<std::tuple<std::string, bool, double>>{{"ref_start 1885.7", true, 1885.7}, {"", true, 0.0},
                     {"ref_start -5", false, 0.0}, {"ref_start 0", false, 0.0}, {"ref_start nan", false, 0.0}, {"ref_start inf", false, 0.0}}) {
                with_line(line);
                Pack e; std::string le; bool got = false;
                try { got = load_pack(tp, cd, e, le); } catch (const std::exception& ex) { got = false; le = ex.what(); }
                const bool good = got == ok && (!got || e.ref_start_ms == val);
                bchk("load_pack \"" + (line.empty() ? std::string("(no line)") : line) + "\"", good, got ? "loaded, ref_start_ms " + std::to_string(e.ref_start_ms) : "refused: " + le);
            }
            std::remove(tp.c_str());
        }
    }
    std::printf("%d bracket vectors, %d differ\n", nb, bbad);
    abad += bbad; na += nb;
    std::printf("%d apply/validate/cache vectors, %d differ\n", na, abad);
    // ---- review 17c D11 ("M1 settles"): m1_from_passes, the port-test vectors of review 17c section 3 (values in GB/s) ----
    int nm = 0, mbad = 0;
    auto mchk = [&](const std::string& what, bool ok, const std::string& detail) {
        ++nm; mbad += !ok;
        std::printf("m1       %-52s %s %s\n", what.c_str(), ok ? "OK  " : "DIFF", detail.c_str());
    };
    {
        auto m1 = [](const std::vector<double>& g, bool& st) { return m1_from_passes(g, kCacheBwTol, st); };
        auto pr = [](const std::vector<double>& g) { std::string s; char b[32]; for (double x : g) { std::snprintf(b, sizeof b, " %.4f", x); s += b; } return s; };
        auto v = [&](const std::string& what, const std::vector<double>& g, bool want_st, double want) {
            bool st = true; const double x = m1(g, st);
            char d[96]; std::snprintf(d, sizeof d, "-> %s %.4f |", st ? "settled" : "not settled", x);
            mchk(what, st == want_st && x == want, d + pr(g));
        };
        const std::vector<double> P3 = {13.6697, 13.1692, 13.5972};
        auto P3p = [&](std::vector<double> more) { std::vector<double> g = P3; g.insert(g.end(), more.begin(), more.end()); return g; };
        v(" 1 P4: three quiet passes", {13.6714, 13.6728, 13.6725}, true, 13.6725);
        v(" 2 P3: not settled (today's median)", P3, false, 13.5972);
        v(" 3 P3 + 1", P3p({13.6720}), false, 13.6697);
        v(" 4 P3 + 2", P3p({13.6720, 13.6731}), false, 13.6720);
        v(" 5 P3 + 3: settled", P3p({13.6720, 13.6731, 13.6725}), true, 13.6725);
        v(" 6 iq3-07 (0.27%)", {13.5536, 13.5723, 13.5903}, false, 13.5723);
        v(" 7 one fast pass (0.21%)", {13.6702, 13.6755, 13.6991}, true, 13.6755);
        v(" 8 q4-05 (0.22%)", {13.7492, 13.7796, 13.7779}, true, 13.7779);
        v(" 9 100 100.24 100.1 (0.24%)", {100, 100.24, 100.1}, true, 100.1);
        v(" 9 100 100.26 100.1 (0.26%)", {100, 100.26, 100.1}, false, 100.1);
        { std::vector<double> g; for (int i = 0; i < kM1MaxPasses; ++i) g.push_back(i % 2 ? 13.20 : 13.67);
          v("10 13.67 / 13.20 alternating, 9 passes", g, false, 13.67); }
        v("11 uniform shift", {13.60, 13.60, 13.61}, true, 13.60);
        v("12 no pass", {}, false, 0.0);
        v("12 one pass", {13.67}, false, 0.0);
        v("12 two passes", {13.67, 13.67}, false, 0.0);
        {   // the engine's loop (generate.cpp): a pass at a time, stopping at the first settled call - P3's sequence settles at pass 6
            const std::vector<double> seq = P3p({13.6720, 13.6731, 13.6725, 13.10, 13.10});
            std::vector<double> g; bool st = false; double x = 0;
            for (size_t i = 0; i < seq.size() && (int) g.size() < kM1MaxPasses && !st; ++i) { g.push_back(seq[i]); x = m1(g, st); }
            char d[64]; std::snprintf(d, sizeof d, "%zu passes, %s %.4f", g.size(), st ? "settled" : "not settled", x);
            mchk(" 5 the engine's loop stops at pass 6", st && g.size() == 6 && x == 13.6725, d);
            std::vector<double> h; st = false;
            for (int i = 0; i < 20 && (int) h.size() < kM1MaxPasses && !st; ++i) { h.push_back(i % 2 ? 13.20 : 13.67); x = m1(h, st); }
            std::snprintf(d, sizeof d, "%zu passes, %s %.4f", h.size(), st ? "settled" : "not settled", x);
            mchk("10 the engine's loop stops at the cap (9)", !st && h.size() == 9 && x == 13.67, d);
        }
        {   // 13: every logged load (results/planner-20261112/next/buildC3/r17c/loads.txt: 171 loads with logged passes up to 11-14 05:10:52),
            // its three passes and the ON line's `M1 x.xxxx`: settled on 169, M1 = the logged M1 on all 171
            std::ifstream in(argc > 6 ? std::string(argv[6]) : std::string("src/program/planner_c_test_m1.txt")); std::string ln;
            int nl = 0, nst = 0, same = 0; std::string unsettled, differ;
            while (std::getline(in, ln)) {
                if (ln.empty() || ln[0] == '#') continue;
                std::istringstream a(ln); std::string tag, q; double p0, p1, p2; std::string logged;
                if (!(a >> tag >> q >> p0 >> p1 >> p2 >> logged)) continue;
                ++nl; bool st = false; const double x = m1({p0, p1, p2}, st);
                char b[32]; std::snprintf(b, sizeof b, "%.4f", x);
                nst += st; if (!st) unsettled += " " + tag;
                if (logged == b) ++same; else differ += " " + tag + " (" + b + " != " + logged + ")";
            }
            mchk("13 logged loads: settled", nl == 171 && nst == 169, std::to_string(nst) + " of " + std::to_string(nl) + " (want 169 of 171); not settled:" + unsettled);
            mchk("13 logged loads: M1 = the logged M1", nl == 171 && same == 171, std::to_string(same) + " of " + std::to_string(nl) + differ);
        }
        {   // 14: check 2 replayed - a record written at P1's M1; P5-like (vector 5) hits, P3 (vector 2) and a uniform shift (vector 11) miss
            const char* td = std::getenv("TMPDIR");
            const std::string path = std::string(td && *td ? td : "/tmp") + "/planner_c_test." + std::to_string((long long) getpid()) + ".m1.x-cache";
            const std::string key = "host=h exe=0123 pack=4567 gpu=0000:07:00.0 boot=b kernel=k vram_free_mib32=320";
            CacheEntry e; e.X = 6145; e.start_ms = 1885.6; e.bw = 13.6726e9; e.reads = 8;
            std::string err; std::remove(path.c_str());
            const bool w = cache_write(path, key, e, "t", err);
            auto get = [&]() { std::ifstream f(path, std::ios::binary); return std::string((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>()); };
            const std::string before = get();
            bool st = false;
            const double m5 = m1(P3p({13.6720, 13.6731, 13.6725}), st), m2 = m1(P3, st), m11 = m1({13.60, 13.60, 13.61}, st);
            auto c = [&](const std::string& what, double bw, bool want, const std::string& must) {
                CacheEntry g; std::string why; const bool hit = cache_read(path, key, bw * 1e9, kCacheBwTol, packs.at("iq4"), g, why);
                const bool ok = w && hit == want && (!hit || g.X == 6145) && (must.empty() || why.find(must) != std::string::npos);
                mchk(what, ok, hit ? "hit X " + std::to_string(g.X) : "miss: " + why);
            };
            c("14 record at 13.6726: vector 5's M1 hits", m5, true, "");
            mchk("14 the hit leaves the file's bytes unchanged", get() == before && !before.empty(), std::to_string(before.size()) + " bytes");
            c("14 vector 2's M1 (P3) misses", m2, false, "GB/s stored");
            c("14 vector 11's M1 (uniform shift) misses", m11, false, "GB/s stored");
            std::remove(path.c_str());
        }
        // 15: why the write guard exists - the rule on a disturbed M1 (each quant's sample load: facts and reads, M1 scaled)
        auto at = [&](const char* q, double fct, int64_t& xm, int64_t& xl, int64_t& xr, int64_t& xa, bool& cl, bool& st, std::string& tag) {
            Facts f = sample.at(q).first; f.bw *= fct; const std::vector<double>& rd = sample.at(q).second; tag = sample_tag.at(q);
            const Result r = derive(packs.at(q), f, rd, -1); const Bracket b = bracket(packs.at(q), f, rd, r);
            xm = b.X_med; xl = b.X_lo; xr = b.X_ref; xa = b.on ? b.X : r.X; cl = b.closed(); std::string why; st = stable_X(packs.at(q), f, r, -1, why);
        };
        for (const auto& [q, fct, want, want_closed, want_stable] : std::vector<std::tuple<const char*, double, int64_t, int, int>>{
                 {"q4", 0.99, 8961, 1, 1}, {"iq4", 0.963, 6401, -1, -1}, {"iq4", 0.9945, 6145, 1, -1}}) {
            if (!sample.count(q)) { mchk(std::string("15 ") + q + " sample", false, "missing"); continue; }
            int64_t xm, xl, xr, xa; bool cl, st; std::string tag; at(q, fct, xm, xl, xr, xa, cl, st, tag);
            const bool ok = xm == want && xl == want && xr == want && xa == want && (want_closed < 0 || cl == (bool) want_closed) && (want_stable < 0 || st == (bool) want_stable);
            char w[96], d[200];
            std::snprintf(w, sizeof w, "15 %s at M1 %+.2f%%: X %lld%s%s", q, (fct - 1) * 100, (long long) want, want_closed > 0 ? ", closed" : "", want_stable > 0 ? ", stable_X" : "");
            std::snprintf(d, sizeof d, "%s: X_med %lld X_lo %lld X_ref %lld applied %lld, %s, stable_X %s", tag.c_str(), (long long) xm, (long long) xl, (long long) xr,
                          (long long) xa, cl ? "closed" : "open", st ? "true" : "false");
            // review 17c: "if the sample load is one of the exceptions, report its values; do not move the factor".  The samples are stored
            // vectors (shadow-<q>-1), not engine loads: IQ4's M1 is 13.6225, 0.36% below the engine loads' mean 13.6721, so -3.70% from it
            // is -4.05% from the mean, past IQ4's next step (6657).  Fable's 64 of 65 engine loads at 6401 is reproduced offline (h.out).
            if (!ok && fct == 0.963) { std::printf("m1       %-52s REPORT %s (the sample is an exception; not counted)\n", w, d); continue; }
            mchk(w, ok, d);
        }
    }
    std::printf("%d M1-settle vectors, %d differ\n", nm, mbad);
    return bad || abad || mbad ? 1 : 0;
}
