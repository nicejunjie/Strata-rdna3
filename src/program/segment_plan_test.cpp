// src/program/segment_plan_test.cpp - [claude] planner step 1: CPU tests of include/strata/program/segment_plan.hpp.
// The expected values are the engine's documented behaviour (the comments the rules were moved from) and the production
// thresholds of 2026-11-12 (results/planner-20261112/env.json).  Build: c++ -std=c++17 -Iinclude src/program/segment_plan_test.cpp
#include "strata/program/segment_plan.hpp"

#include <cstdio>
#include <cstdlib>

namespace sp = strata::program::segment_plan;
static int fails = 0;
#define CHECK(x) do { if (!(x)) { std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); ++fails; } } while (0)

static sp::Thresholds iq3_2gpu() {   // env.json "qwen-flash-iq3 2gpu" (+ the gfx1100 table)
    sp::Thresholds t; t.tp_on = true; t.tp_lo = 3072; t.tp_hi = 16384; t.tp_chunk = 16384; t.dp_min = 8000; t.pipe_chunk = 8448;
    t.head = 2688; t.head_min = 131072; t.head_tp_keep = 900; t.head_tp_min = 8449; return t;
}
static sp::Thresholds q4_2gpu() {
    sp::Thresholds t; t.tp_on = true; t.tp_lo = 8192; t.tp_hi = 28672; t.tp_chunk = 28672; t.dp_min = 18000; t.pipe_chunk = 18432;
    t.head = 1344; t.head_min = 49152; t.head2 = 2688; t.head2_min = 245760; t.head_big = 2688; t.head_tp_keep = 1400;
    t.head_tp_min = 8449; return t;
}
static sp::Thresholds one_card(int64_t share_max) {   // 1-GPU: EQUAL on, tail max 32768 (rg23-defaults); share max = stream_all_min
    sp::Thresholds t; t.equal = true; t.equal_tail_max = 32768; (void) share_max; return t;
}

int main() {
    // dp_takes / dp_step_for: "8441 tokens plan 4352 + 4089, refused"; 8448 is the shortest
    sp::Thresholds t = iq3_2gpu();
    CHECK(sp::dp_step_for(8441, 32768) == 4352);
    CHECK(!sp::dp_takes(8441, 32768, t));
    CHECK(!sp::dp_takes(8448, 32768, t));   // (8448 -> 4352 + 4096: last < 4224, refused too; the floor comment is loose)
    CHECK(sp::dp_takes(8704, 32768, t));    // 4352 + 4352: the shortest the DP plan takes
    CHECK(sp::dp_step_for(73728, 32768) == 18432);   // 4 chunks (k even)
    CHECK(sp::dp_step_for(40960, 40960) == 20480);   // at least 2
    // tpp
    CHECK(sp::tpp_takes(3072, t) && sp::tpp_takes(16384, t) && !sp::tpp_takes(16385, t) && !sp::tpp_takes(3071, t));
    CHECK(sp::tpp_chunk_for(16377, 32768, t) == 16384);
    sp::Thresholds q = q4_2gpu();
    CHECK(sp::tpp_chunk_for(28665, 40960, q) == 28672);
    // pipe chunk, PIPE_HALF off
    CHECK(sp::pipe_chunk_for(4096, t) == 8448);
    sp::Thresholds h = t; h.pipe_half_max = 8192; CHECK(sp::pipe_chunk_for(4096, h) == 2048);
    // dp_head
    CHECK(sp::dp_head(0, 4) == 0 && sp::dp_head(6, 4) == 2 && sp::dp_head(8, 4) == 0);
    // chunk geometry, 1 GPU: the 57K Q4 case (EQUAL, tail max): 57337 tokens at cap 49152, stream_all_min 8193
    sp::Thresholds o = one_card(8193);
    CHECK(sp::equal_chunk(57337, 49152, o, 8193) == 28672);         // tail 8185 < 8193 but cap 49152 > tail max: equal
    sp::Thresholds o2 = o; o2.equal_tail_max = (int64_t) 1 << 40;   // the rg23-ceil default: no limit
    CHECK(sp::equal_chunk(57337, 49152, o2, 8193) == 49152);        // short tail kept: 49152 + 8185 (the stall)
    CHECK(sp::equal_chunk(8185, 49152, o, 8193) == 8192);           // one chunk, rounded to 256
    CHECK(sp::request_chunk(0, 8192) == 0 && sp::request_chunk(1, 8192) == 256);
    // lend side + read side, 2 GPU, the production IQ3 pair
    sp::PartFacts f; f.a0 = 0; f.b0 = 16377; f.cap = 32768; f.dp_on = true; f.layer_split = true; f.stream_all_min = 480;
    sp::SegmentPlan s = sp::legacy_plan(f, t);
    CHECK(s.path == sp::Path::TP && s.lend_path == sp::Path::TP && s.tp_read && s.want == 16384 && s.read_chunk == 16384);
    f.b0 = 18425; s = sp::legacy_plan(f, t);
    CHECK(s.path == sp::Path::DP && s.lend_path == sp::Path::DP && s.want == 9216 && s.read_chunk == 32768);
    f.b0 = 2041; s = sp::legacy_plan(f, t);
    CHECK(s.path == sp::Path::Pipe && s.want == 2048);
    f.b0 = 16377; f.planned = true; s = sp::legacy_plan(f, t);   // a --prefill-pipe planned segment: no TP/DP read
    CHECK(s.path == sp::Path::Pipe && s.lend_path == sp::Path::TP);  // (the lend side does not see `planned`: today's behaviour)
    f.planned = false; f.tpp_pictures = true; s = sp::legacy_plan(f, t);   // picture rows: lend lays out DP's step, read tries TP
    CHECK(s.lend_path == sp::Path::DP && s.path == sp::Path::TP);
    f.tpp_pictures = false; f.exact = 8192; s = sp::legacy_plan(f, t);
    CHECK(s.want == 8192);
    // a part after a reused prefix starting inside a K/V block: the head goes first on the pipeline
    f.exact = 0; f.a0 = 6; f.b0 = 6 + 16377; s = sp::legacy_plan(f, t);
    CHECK(s.head == 2 && s.dp_n == 16375 && s.path == sp::Path::TP);
    // 1 GPU: no DP link
    sp::PartFacts g1; g1.a0 = 0; g1.b0 = 57337; g1.cap = 49152; g1.stream_all_min = 8193;
    s = sp::legacy_plan(g1, o);
    CHECK(s.path == sp::Path::Single && s.want == 28672);
    // heads, IQ3 2-GPU TP read of 16K: all rows but the 900 coldest join (TP_KEEP, prompt >= 8449)
    CHECK(sp::head_rows(t, 2462, 16384, 16384, true, true, true) == 2462 - 900);
    CHECK(sp::head_rows(t, 2462, 16384, 8192, true, true, true) == 0);     // below HEAD_TP_MIN: none
    CHECK(sp::head_rows(t, 800, 16384, 16384, true, true, true) == 0);     // a part of <= 900 rows keeps the late refill
    CHECK(sp::head_rows(t, 4000, 16384, 131072, true, true, false) == 2688); // HEAD from 131072
    // Q4: HEAD_BIG for a loan laid out above 32768, on its own
    CHECK(sp::head_rows(q, 3000, 40960, 40960, true, true, false) == 2688);
    CHECK(sp::head_rows(q, 3000, 40960, 40960, false, true, false) == 0);    // the early refill is not paced: no head
    CHECK(sp::head_rows(q, 3000, 32768, 250000, true, true, false) == 2688); // HEAD2 past 245760
    if (fails) { std::fprintf(stderr, "%d failed\n", fails); return 1; }
    std::printf("segment_plan_test: all passed\n");
    return 0;
}
