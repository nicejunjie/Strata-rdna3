// src/program/planner_rows_test.cpp - [claude] phase 3 Deploy A: CPU test of planner_rows.hpp.
// Build: c++ -std=c++17 -Iinclude src/program/planner_rows_test.cpp -o planner_rows_test
// Run:   ./planner_rows_test                         (unit cases)
//        ./planner_rows_test TABLE KEY=VALUE ...      (parity: the rows applied to an empty environment give exactly the
//                                                      threshold keys of KEY=VALUE ..., today's config env; nothing more)
#include "strata/program/planner_rows.hpp"

#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>

namespace PR = strata::program::planner_rows;
static int fails = 0;
#define CHECK(c) do { if (!(c)) { std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #c); ++fails; } } while (0)

static std::map<std::string, std::string> run(const std::vector<PR::Row>& rows, std::map<std::string, std::string> env,
                                              std::vector<PR::Applied>* ap = nullptr) {
    auto a = PR::apply(rows, [&](const char* k) -> const char* { auto it = env.find(k); return it == env.end() ? nullptr : it->second.c_str(); },
                       [&](const char* k, const char* v) { env.emplace(k, v); });
    if (ap) *ap = a;
    return env;
}

int main(int argc, char** argv) {
    if (argc == 1) {
        std::vector<PR::Row> r; std::string err;
        CHECK(PR::parse("quant q4\n# comment\nrow2 STRATA_TP_PREFILL_MIN 8192\nconst layers 48\nrow2 STRATA_REFILL_HEAD2 2688   # class C policy", r, err));
        CHECK(r.size() == 2 && r[0].key == "STRATA_TP_PREFILL_MIN" && r[1].value == "2688");
        CHECK(!PR::parse("row2 STRATA_TP_PREFILL 1\n", r, err));                 // a switch, not a threshold
        CHECK(!PR::parse("row2 STRATA_DP_MIN 18000x\n", r, err));                // not an integer
        CHECK(!PR::parse("row2 STRATA_DP_MIN\n", r, err));                       // no value
        CHECK(!PR::parse("row2 STRATA_DP_MIN 1\nrow2 STRATA_DP_MIN 2\n", r, err));  // duplicate
        CHECK(PR::parse("", r, err) && r.empty());
        CHECK(PR::parse("row2 STRATA_REFILL_HEAD2 -1\n", r, err) && r[0].value == "-1");
        // precedence: an explicit env value wins, else the row
        CHECK(PR::parse("row2 STRATA_DP_MIN 18000\nrow2 STRATA_TP_PREFILL_MAX 28672\n", r, err));
        std::vector<PR::Applied> ap;
        auto e = run(r, {{"STRATA_DP_MIN", "11000"}}, &ap);
        CHECK(e["STRATA_DP_MIN"] == "11000" && e["STRATA_TP_PREFILL_MAX"] == "28672");
        CHECK(ap[0].source == "env" && ap[1].source == "table");
        // (e) the derived chunk rows: TP_PREFILL_CHUNK = ceil256(TP_PREFILL_MAX), DP_PIPE_CHUNK = ceil1024(max(8448, DP_MIN) + 2)
        {   // table-only: the table sets the bases and not the chunks -> both derived (today's IQ4 / Q4 values)
            CHECK(PR::parse("row2 STRATA_TP_PREFILL_MAX 20480\nrow2 STRATA_DP_MIN 10500\n", r, err));
            auto d = run(r, {}, &ap);
            CHECK(d["STRATA_TP_PREFILL_CHUNK"] == "20480" && d["STRATA_DP_PIPE_CHUNK"] == "11264");
            CHECK(ap.size() == 4 && ap[2].source == "derived" && ap[3].source == "derived");
            CHECK(PR::parse("row2 STRATA_TP_PREFILL_MAX 28500\nrow2 STRATA_DP_MIN 18000\n", r, err));
            d = run(r, {});
            CHECK(d["STRATA_TP_PREFILL_CHUNK"] == "28672" && d["STRATA_DP_PIPE_CHUNK"] == "18432");
            CHECK(PR::parse("row2 STRATA_TP_PREFILL_MAX 16384\n", r, err));   // IQ3-like: the rule gives the default
            d = run(r, {});
            CHECK(d["STRATA_TP_PREFILL_CHUNK"] == "16384" && !d.count("STRATA_DP_PIPE_CHUNK"));
            CHECK(PR::parse("row2 STRATA_TP_PREFILL_MAX 20481\n", r, err));
            CHECK(run(r, {})["STRATA_TP_PREFILL_CHUNK"] == "20736");                // rounds up to 256
        }
        {   // the DP rule's bound (review-fable-impl required change 2): chunk >= ceil256(max(8448, DP_MIN) + 2)
            const std::pair<const char*, const char*> dp[] = {{"11264", "12288"}, {"11263", "12288"}, {"5000", "9216"},
                                                              {"11000", "11264"}, {"18000", "18432"}, {"8448", "9216"},
                                                              {"10238", "10240"}, {"10239", "11264"}};
            for (const auto& c : dp) {
                CHECK(PR::parse(std::string("row2 STRATA_DP_MIN ") + c.first + "\n", r, err));
                const auto d = run(r, {});
                CHECK(d.count("STRATA_DP_PIPE_CHUNK") && d.at("STRATA_DP_PIPE_CHUNK") == c.second);
                if (d.count("STRATA_DP_PIPE_CHUNK") && d.at("STRATA_DP_PIPE_CHUNK") != c.second)
                    std::fprintf(stderr, "  DP_MIN %s: derived %s, expected %s\n", c.first, d.at("STRATA_DP_PIPE_CHUNK").c_str(), c.second);
            }
        }
        {   // the table's own chunk row: kept, nothing derived
            CHECK(PR::parse("row2 STRATA_TP_PREFILL_MAX 20480\nrow2 STRATA_TP_PREFILL_CHUNK 4096\nrow2 STRATA_DP_MIN 10500\nrow2 STRATA_DP_PIPE_CHUNK 8448\n", r, err));
            auto d = run(r, {}, &ap);
            CHECK(d["STRATA_TP_PREFILL_CHUNK"] == "4096" && d["STRATA_DP_PIPE_CHUNK"] == "8448" && ap.size() == 4);
        }
        {   // env wins: an explicit chunk is kept; an explicit base is the base the rule uses
            CHECK(PR::parse("row2 STRATA_TP_PREFILL_MAX 20480\nrow2 STRATA_DP_MIN 10500\n", r, err));
            auto d = run(r, {{"STRATA_TP_PREFILL_CHUNK", "12000"}, {"STRATA_DP_PIPE_CHUNK", "9000"}}, &ap);
            CHECK(d["STRATA_TP_PREFILL_CHUNK"] == "12000" && d["STRATA_DP_PIPE_CHUNK"] == "9000" && ap.size() == 2);
            d = run(r, {{"STRATA_TP_PREFILL_MAX", "30000"}, {"STRATA_DP_MIN", "9000"}}, &ap);
            CHECK(d["STRATA_TP_PREFILL_CHUNK"] == "30208" && d["STRATA_DP_PIPE_CHUNK"] == "9216");
            CHECK(ap[0].source == "env" && ap[1].source == "env" && ap[2].source == "derived");
        }
        {   // no table rows for the bases (and no table at all): nothing derived, the parse-site defaults stay
            CHECK(PR::parse("row2 STRATA_REFILL_HEAD2 2688\n", r, err));
            auto d = run(r, {{"STRATA_TP_PREFILL_MAX", "30000"}, {"STRATA_DP_MIN", "9000"}});
            CHECK(!d.count("STRATA_TP_PREFILL_CHUNK") && !d.count("STRATA_DP_PIPE_CHUNK"));
            CHECK(PR::parse("", r, err));
            d = run(r, {{"STRATA_TP_PREFILL_MAX", "30000"}});
            CHECK(d.size() == 1 && !d.count("STRATA_TP_PREFILL_CHUNK"));
            CHECK(PR::parse("row2 STRATA_TP_PREFILL_MAX 0\nrow2 STRATA_DP_MIN -5\n", r, err));   // a base that is not positive
            d = run(r, {});
            CHECK(!d.count("STRATA_TP_PREFILL_CHUNK") && !d.count("STRATA_DP_PIPE_CHUNK"));
        }
        std::printf("planner_rows_test: unit cases %s\n", fails ? "FAILED" : "passed");
        return fails ? 1 : 0;
    }
    std::ifstream f(argv[1]); std::stringstream ss; ss << f.rdbuf();
    std::vector<PR::Row> r; std::string err;
    if (!PR::parse(ss.str(), r, err)) { std::printf("parity %s: table refused: %s\n", argv[1], err.c_str()); return 1; }
    std::map<std::string, std::string> want;
    for (int i = 2; i < argc; ++i) {
        std::string kv = argv[i]; size_t eq = kv.find('='); std::string k = kv.substr(0, eq);
        for (const auto& a : PR::allowed_keys()) if (a == k) want[k] = kv.substr(eq + 1);
    }
    auto got = run(r, {});
    CHECK(got == want);
    for (auto& [k, v] : want) if (got[k] != v) std::printf("  %s: config %s, table %s\n", k.c_str(), v.c_str(), got.count(k) ? got[k].c_str() : "(absent)");
    for (auto& [k, v] : got) if (!want.count(k)) std::printf("  %s: table %s, not in config\n", k.c_str(), v.c_str());
    std::printf("parity %s: %zu rows, %zu config threshold keys: %s\n", argv[1], r.size(), want.size(), fails ? "DIFFERENT" : "identical");
    return fails ? 1 : 0;
}
