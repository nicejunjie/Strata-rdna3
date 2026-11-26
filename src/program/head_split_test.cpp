// src/program/head_split_test.cpp - [claude] head split (results/planner-20261112/phase3/followup, review-fable-round4 condition 4):
// a part [a0, b0) whose read side is TP or DP with a head h > 0 is read as the windows head [a0, a0 + h) and the batched
// sub-part [a0 + h, b0).  Checks on random facts and thresholds:
//   read side: the sub-part's path, read chunk and TP / DP length equal today's for [a0, b0), and its head is 0;
//   lend side: every difference in want / want_full / lend path between [a0 + h, b0) and [a0, b0) is listed (an
//   unexplained difference stops the build).  The 2-GPU table rows of the three packs are among the threshold sets.
// Build: c++ -O2 -std=c++17 -Iinclude src/program/head_split_test.cpp -o head_split_test && ./head_split_test [N]
#include "strata/program/segment_plan.hpp"
#include <cstdio>
#include <cstdlib>
#include <map>
#include <random>
#include <string>
namespace sp = strata::program::segment_plan;
int main(int argc, char** argv) {
    const long N = argc > 1 ? std::atol(argv[1]) : 2000000;
    std::mt19937_64 rng(20261117);
    auto U = [&](int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(rng); };
    std::vector<sp::Thresholds> rows;
    auto pack = [&](int64_t lo, int64_t hi, int64_t ch, int64_t dmin, int64_t pipe) {
        sp::Thresholds t; t.tp_on = true; t.tp_lo = lo; t.tp_hi = hi; t.tp_chunk = ch; t.dp_min = dmin; t.pipe_chunk = pipe; return t; };
    rows.push_back(pack(2817, 16384, 16384, 8000, 8448));      // IQ3 deploy-A rows
    rows.push_back(pack(4865, 20480, 20480, 11000, 11264));    // IQ4
    rows.push_back(pack(8192, 28672, 28672, 18000, 18432));    // Q4
    long bad_read = 0, split = 0; std::map<std::string, long> lend_diff;
    for (long i = 0; i < N; ++i) {
        sp::Thresholds t = (i % 4 < 3) ? rows[(size_t) (i % 4)] : [&] {
            sp::Thresholds r; r.tp_on = U(0, 1); r.tp_lo = std::max<int64_t>(1024, U(0, 40000)); r.tp_hi = U(0, 70000);
            r.tp_chunk = std::max<int64_t>(1024, U(0, 40000)); r.dp_min = U(0, 1) ? 8000 : std::max<int64_t>(8448, U(0, 40000));
            r.pipe_chunk = std::max<int64_t>(512, U(0, 32768)); r.pipe_half_min = std::max<int64_t>(2048, U(0, 9000)); r.pipe_half_max = U(0, 1) ? 0 : U(0, 20000);
            r.equal = U(0, 1); return r; }();
        sp::PartFacts f; f.a0 = U(0, 300000); f.b0 = f.a0 + U(1, 140000); f.cap = U(0, 3) ? 32768 : (int64_t) (1024 * U(4, 48));
        f.idx_block = U(0, 7) ? 4 : U(1, 16); f.dp_on = U(0, 7) != 0; f.layer_split = true; f.planned = U(0, 15) == 0;
        f.stream_all_min = U(0, 1) ? 1024 : U(256, 4096); f.host_plan_chunk = U(0, 3) ? 0 : U(1024, 40000);
        f.split_small = U(0, 3) ? 0 : U(256, 4096); f.split_small_max = f.split_small ? U(1024, 16384) : 0;
        f.head_alias = U(0, 15) == 0; f.tpp_pictures = U(0, 15) == 0; f.exact = U(0, 7) ? 0 : U(256, 32768);
        sp::SegmentPlan a; sp::read_side(a, f, t);
        if (!((a.path == sp::Path::TP || a.path == sp::Path::DP) && a.head > 0 && f.a0 + a.head < f.b0)) continue;
        ++split;
        sp::PartFacts g = f; g.a0 = f.a0 + a.head;
        sp::SegmentPlan b; sp::read_side(b, g, t);
        if (b.path != a.path || b.read_chunk != a.read_chunk || b.dp_n != a.dp_n || b.head != 0) {
            if (++bad_read <= 5) std::printf("READ MISMATCH a0 %lld b0 %lld h %lld: path %d/%d chunk %lld/%lld dp_n %lld/%lld head' %lld\n",
                (long long) f.a0, (long long) f.b0, (long long) a.head, (int) a.path, (int) b.path, (long long) a.read_chunk, (long long) b.read_chunk,
                (long long) a.dp_n, (long long) b.dp_n, (long long) b.head);
        }
        sp::SegmentPlan la, lb; sp::lend_side(la, f, t); sp::lend_side(lb, g, t);
        if (la.lend_path != lb.lend_path) ++lend_diff["lend path"];
        if (la.want_full != lb.want_full) ++lend_diff[la.lend_path == sp::Path::TP || la.lend_path == sp::Path::DP ? "want_full (TP/DP lend)"
                                                       : f.tpp_pictures ? "want_full (pipe lend: pictures, no split in the engine)" : "want_full (pipe lend: other)"];
        if (la.want != lb.want) {
            const int64_t tok = f.b0 - f.a0;
            std::string why = f.exact > 0 ? "want (exact)" : f.head_alias ? "want (head_alias)" :
                (f.split_small > 0 && ((tok <= f.split_small_max) != (tok - a.head <= f.split_small_max))) ? "want (split_small threshold crossed)" :
                (la.want_full != lb.want_full) ? (f.tpp_pictures ? "want (follows want_full: pictures, no split in the engine)" : "want (follows want_full: other)") : "want (other)";
            ++lend_diff[why];
        }
    }
    std::printf("head_split_test: %ld cases, %ld split parts, read-side mismatches %ld\n", N, split, bad_read);
    for (auto& [k, v] : lend_diff) std::printf("  lend difference: %-40s %ld (%.4f%% of split parts)\n", k.c_str(), v, 100.0 * v / std::max(1L, split));
    return bad_read ? 1 : 0;
}
