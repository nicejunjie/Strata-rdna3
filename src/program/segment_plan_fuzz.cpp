// src/program/segment_plan_fuzz.cpp - [claude] planner (review 4 D8): the re-merge gate's rule fuzz.  The pre-refactor
// rules, transcribed from fa5c621d7's generate.cpp (dp_step_for, dp_takes, tpp_takes, tpp_chunk_for, pipe_chunk_for,
// dp_head, request_chunk, equal_chunk, the serve lend's chunk choice, read_part's path choice, late_issue's head rule;
// statics turned into the Thresholds fields) against segment_plan.hpp on random inputs.
// Build: c++ -O2 -std=c++17 -Iinclude src/program/segment_plan_fuzz.cpp -o segment_plan_fuzz && ./segment_plan_fuzz [N]
#include "strata/program/segment_plan.hpp"

#include <cstdio>
#include <cstdlib>
#include <random>

namespace sp = strata::program::segment_plan;
using T = sp::Thresholds;

namespace old {   // fa5c621d7, verbatim apart from the statics
int64_t dp_step_for(int64_t n, int64_t cap) { int64_t k = std::max<int64_t>(2, (n + cap - 1) / cap); if (k & 1) ++k; return ((n + k - 1) / k + 255) / 256 * 256; }
bool dp_takes(int64_t n, int64_t cap, const T& t) {
    if (n < t.dp_min) return false;
    const int64_t step = old::dp_step_for(n, cap); const int64_t full = (n + step - 1) / step;
    return full >= 2 && n - (full - 1) * step >= 4224;
}
bool tpp_takes(int64_t n, const T& t) { return t.tp_on && n >= t.tp_lo && n <= t.tp_hi; }
int64_t tpp_chunk_for(int64_t n, int64_t cap, const T& t) {
    const int64_t C = std::min(cap, t.tp_chunk); const int64_t k = std::max<int64_t>(1, (n + C - 1) / C);
    return std::min(C, ((n + k - 1) / k + 255) / 256 * 256);
}
int64_t pipe_chunk_for(int64_t n, const T& t) { if (n >= t.pipe_half_min && n <= t.pipe_half_max) return ((n + 1) / 2 + 255) / 256 * 256; return t.pipe_chunk; }
int64_t dp_head(int64_t a0, int64_t blk) { return (blk - a0 % blk) % blk; }
int64_t request_chunk(int64_t tokens, int64_t max_chunk) {
    if (tokens <= 0 || max_chunk <= 0) return 0;
    const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255 ? tokens : ((tokens + 255) / 256) * 256;
    return std::min(max_chunk, rounded);
}
int64_t equal_chunk(int64_t tokens, int64_t max_chunk, const T& t, int64_t sam) {
    const int64_t plain = old::request_chunk(tokens, max_chunk);
    if (!t.equal || tokens <= 0 || max_chunk <= 0) return plain;
    const int64_t n = tokens / max_chunk + (tokens % max_chunk != 0); const int64_t last = tokens - (n - 1) * max_chunk;
    if (n <= 1 || (last < sam && max_chunk <= t.equal_tail_max)) return plain;
    const int64_t per = tokens / n + (tokens % n != 0);
    const int64_t rounded = per > std::numeric_limits<int64_t>::max() - 255 ? per : ((per + 255) / 256) * 256;
    return std::min(max_chunk, rounded);
}
int64_t planned_chunk(int64_t tokens, int64_t cap, int64_t plan, const T& t, int64_t sam) {
    if (plan <= 0 || plan >= cap || tokens <= 0) return old::equal_chunk(tokens, cap, t, sam);
    return strata::program::prefill_plan::balanced_chunk(tokens, cap, plan);
}
void lend(const sp::PartFacts& f, const T& t, int64_t& want, bool& tp_read) {
    const int64_t tokens = f.b0 - f.a0;
    int64_t want_full = old::planned_chunk(tokens, f.cap, f.host_plan_chunk, t, f.stream_all_min);
    tp_read = false;
    if (f.dp_on) {
        const int64_t dp_n = tokens - old::dp_head(f.a0, f.idx_block);
        if (old::tpp_takes(dp_n, t) && !f.tpp_pictures && !f.lend_no_tp) { want_full = std::min(f.cap, old::tpp_chunk_for(dp_n, f.cap, t)); tp_read = true; }
        else if (old::dp_takes(dp_n, f.cap, t)) want_full = std::min(f.cap, old::dp_step_for(dp_n, f.cap));
        else want_full = std::min(want_full, old::pipe_chunk_for(tokens, t));
    }
    const int64_t want_seg = f.head_alias ? f.cap : f.split_small > 0 && tokens <= f.split_small_max ? std::min(want_full, f.split_small) : want_full;
    want = f.exact > 0 ? std::min(f.exact, f.cap) : want_seg;
}
int read_path(const sp::PartFacts& f, const T& t) {   // 3 TP, 4 DP, 0 other
    const int64_t n = f.b0 - f.a0 - old::dp_head(f.a0, f.idx_block);
    if (!f.planned && f.dp_on && old::tpp_takes(n, t)) return 3;
    if (!f.planned && f.dp_on && old::dp_takes(n, f.cap, t)) return 4;
    return 0;
}
size_t head_rows(const T& t, size_t rows, int64_t lent_chunk, int64_t req_prompt_n, bool paced, bool dp_read_req, bool tpp_read_req) {
    int64_t head_want = t.head2 >= 0 && req_prompt_n >= t.head2_min ? t.head2 : t.head;
    if (t.head_big > 0 && lent_chunk > t.head_big_chunk) head_want = std::max(head_want, t.head_big);
    size_t head_n = paced && (!t.head_dp || dp_read_req) && req_prompt_n >= t.head_min ? std::min(rows, (size_t) head_want) : 0;
    if (paced && tpp_read_req && t.head_tp_keep >= 0 && req_prompt_n >= t.head_tp_min && rows > (size_t) t.head_tp_keep)
        head_n = std::max(head_n, rows - (size_t) t.head_tp_keep);
    if (paced && t.head_big > 0 && lent_chunk > t.head_big_chunk) head_n = std::max(head_n, std::min(rows, (size_t) t.head_big));
    return head_n;
}
}  // namespace old

int main(int argc, char** argv) {
    const long N = argc > 1 ? std::atol(argv[1]) : 2000000;
    std::mt19937_64 g(12345);
    auto u = [&](int64_t lo, int64_t hi) { return std::uniform_int_distribution<int64_t>(lo, hi)(g); };
    auto len = [&]() -> int64_t {   // weighted to chunk multiples and the DP floor
        switch (u(0, 4)) {
            case 0: return u(1, 300000);
            case 1: return u(1, 64) * 256 + u(-8, 8);
            case 2: return u(8000, 9000);
            case 3: return u(1, 8) * 32768 + u(-300, 300);
            default: return u(1, 70000);
        }
    };
    long bad = 0;
    for (long i = 0; i < N; ++i) {
        T t; t.tp_on = u(0, 1); t.tp_lo = std::max<int64_t>(1024, u(0, 12000)); t.tp_hi = u(1000, 40000); t.tp_chunk = std::max<int64_t>(1024, u(0, 40000));
        t.dp_min = u(0, 1) ? 8000 : std::max<int64_t>(8448, u(0, 30000)); t.pipe_chunk = std::max<int64_t>(512, u(0, 20000));
        t.pipe_half_min = std::max<int64_t>(2048, u(0, 8000)); t.pipe_half_max = u(0, 2) ? 0 : u(0, 16000);
        t.equal = u(0, 1); t.equal_tail_max = u(0, 2) ? 32768 : u(0, 1 << 20);
        t.head = u(0, 4000); t.head_min = u(0, 300000); t.head2 = u(0, 1) ? -1 : u(0, 4000); t.head2_min = u(0, 300000); t.head_dp = u(0, 1);
        t.head_big = u(0, 1) ? 0 : u(0, 4000); t.head_big_chunk = u(0, 50000); t.head_tp_keep = u(0, 1) ? -1 : u(0, 3000); t.head_tp_min = u(0, 20000);
        sp::PartFacts f; f.a0 = u(0, 3) ? 0 : u(0, 100000); f.b0 = f.a0 + std::max<int64_t>(1, len()); f.cap = u(0, 1) ? 32768 : u(1, 192) * 256;
        f.host_plan_chunk = u(0, 3) ? 0 : u(1, 128) * 256; f.stream_all_min = u(0, 1) ? 480 : u(1024, 12000); f.idx_block = u(0, 4) ? 4 : u(1, 16);
        f.dp_on = u(0, 1); f.layer_split = f.dp_on || u(0, 1); f.tpp_pictures = !u(0, 5); f.lend_no_tp = !u(0, 5); f.head_alias = !u(0, 10);
        f.split_small = u(0, 4) ? 0 : u(1, 32) * 256; f.split_small_max = f.split_small ? f.split_small + u(0, 8192) : 0; f.exact = u(0, 4) ? 0 : u(1, 128) * 256;
        f.planned = !u(0, 5);
        const sp::SegmentPlan s = sp::legacy_plan(f, t);
        int64_t want; bool tp_read; old::lend(f, t, want, tp_read);
        const int rp = old::read_path(f, t);
        const int np = s.path == sp::Path::TP ? 3 : s.path == sp::Path::DP ? 4 : 0;
        const size_t rows = (size_t) u(0, 8000); const int64_t lc = u(0, 50000), rpn = u(0, 300000); const bool pa = u(0, 1), dr = u(0, 1), tr = u(0, 1);
        const bool ok = want == s.want && tp_read == s.tp_read && rp == np &&
                        old::head_rows(t, rows, lc, rpn, pa, dr, tr) == sp::head_rows(t, rows, lc, rpn, pa, dr, tr) &&
                        old::equal_chunk(f.b0 - f.a0, f.cap, t, f.stream_all_min) == sp::equal_chunk(f.b0 - f.a0, f.cap, t, f.stream_all_min);
        if (!ok && ++bad <= 5) std::fprintf(stderr, "MISMATCH case %ld: a0 %lld b0 %lld cap %lld\n", i, (long long) f.a0, (long long) f.b0, (long long) f.cap);
    }
    std::printf("segment_plan_fuzz: %ld cases, %ld mismatches\n", N, bad);
    return bad ? 1 : 0;
}
