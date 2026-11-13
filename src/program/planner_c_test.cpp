// [claude] planner rev 5 shadow: planner_c.hpp against rule_c.py (localLLM 67850ea) on every stored load - X, acted, the three
// per-text crossings and the guard ratio must match.  CPU only.
//   g++ -O2 -std=c++17 -Iinclude src/program/planner_c_test.cpp -o planner_c_test && ./planner_c_test PACKDIR COUNTSDIR PROFILE
#include "strata/program/planner_c.hpp"
#include <cstdio>
#include <sstream>
#include <fstream>
using namespace strata::program::planner_c;
int main(int argc, char** argv) {
    if (argc < 4) { std::fprintf(stderr, "usage: %s PACKDIR COUNTSDIR PROFILE [VECTORS]\n", argv[0]); return 2; }
    const std::string pd = argv[1], cd = argv[2], prof = argv[3];
    std::map<std::string, Pack> packs; std::string err;
    for (const char* q : {"iq3", "iq4", "q4"}) {
        Pack p;
        if (!load_pack(pd + "/pack-" + q + ".txt", cd, p, err) || !validate(p, 48, 512, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
        packs[q] = std::move(p);
    }
    std::vector<int64_t> rank;
    if (!load_rank(prof, 48, 512, rank, err)) { std::fprintf(stderr, "%s\n", err.c_str()); return 1; }
    const std::map<std::string, int64_t> configured = {{"iq3", 4097}, {"iq4", 6145}, {"q4", 8193}};
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
        const bool ok = r.X == eX && r.acted == eA && (eX < 0 && ec[0] < 0 ? true : (r.crossing[0] == ec[0] && r.crossing[1] == ec[1] && r.crossing[2] == ec[2] && r.ratio == er));
        ++n; bad += !ok;
        std::printf("%-26s %s X %lld (rule_c %lld) acted %lld (%lld) crossings %lld %lld %lld (%lld %lld %lld) ratio %.17g (%.17g)%s %s\n", tag.c_str(), ok ? "OK  " : "DIFF",
                    (long long) r.X, (long long) eX, (long long) r.acted, (long long) eA, (long long) r.crossing[0], (long long) r.crossing[1], (long long) r.crossing[2],
                    (long long) ec[0], (long long) ec[1], (long long) ec[2], r.ratio, er, r.why.empty() ? "" : " |", r.why.c_str());
    }
    std::printf("%d vectors, %d differ\n", n, bad);
    return bad ? 1 : 0;
}
