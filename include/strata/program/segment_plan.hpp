// include/strata/program/segment_plan.hpp - [claude] planner step 1 (results/planner-20261112, design rev 3 section 2):
// every per-length decision the prompt path makes for one part of a request, in ONE place, as pure functions of the
// part and the thresholds (no engine, no GPU: CPU unit-tested in segment_plan_test.cpp).
//
// This header holds TODAY's rules ("legacy"), moved verbatim from generate.cpp (dp_step_for / dp_takes / tpp_takes /
// tpp_chunk_for / pipe_chunk_for / dp_head, the serve lend()'s chunk choice, read_part()'s path choice, the chunk
// lambdas request_chunk / equal_chunk / planned_chunk, late_issue()'s refill-head rule) so that the engine computes
// each part's SegmentPlan once and every site reads it.  Behaviour is unchanged by construction: the same arithmetic,
// the same inputs; the thresholds are parsed from the same environment variables, once, by the engine.
//
// The decisions made inside the prompt kernels (prefill.cpp: the big ring vs the 8-slot stage, stream-all vs staged
// per chunk, the routed walk, the fused help share) depend on the pack and the loan; the engine fills them into the
// plan from Prefill's own functions and logs the ones the read really took next to the planned ones.
#pragma once

#include <algorithm>
#include <cstdint>
#include <limits>
#include <string>

#include "strata/program/prefill_plan.hpp"

namespace strata::program::segment_plan {

/// the shortest last chunk the DP read takes: the drafter's window (--mtp-window 4096) and 64 positions, on the 256 grid
constexpr int64_t kDpMinLast = 4224;

/// today's thresholds, as the engine parses them from the environment (defaults = the engine's)
struct Thresholds {
    bool tp_on = false;                 ///< STRATA_TP_PREFILL (with STRATA_DP_PREFILL)
    int64_t tp_lo = 4096;               ///< STRATA_TP_PREFILL_MIN (floored at 1024)
    int64_t tp_hi = 32768;              ///< STRATA_TP_PREFILL_MAX
    int64_t tp_chunk = 16384;           ///< STRATA_TP_PREFILL_CHUNK (floored at 1024)
    int64_t dp_min = 8000;              ///< STRATA_DP_MIN (unset 8000; an explicit value is floored at 2 * kDpMinLast)
    int64_t pipe_chunk = 8192;          ///< STRATA_DP_PIPE_CHUNK (floored at 512)
    int64_t pipe_half_min = 2048;       ///< STRATA_PIPE_HALF_MIN (floored at 2048)
    int64_t pipe_half_max = 0;          ///< STRATA_PIPE_HALF_MAX (0: off)
    bool equal = false;                 ///< STRATA_PREFILL_EQUAL=1
    int64_t equal_tail_max = 32768;     ///< STRATA_PREFILL_EQUAL_TAIL_MAX (engine-line default)
    // the refill head (late_issue)
    int64_t head = 0;                   ///< STRATA_REFILL_HEAD
    int64_t head_min = 0;               ///< STRATA_REFILL_HEAD_MIN
    int64_t head2 = -1;                 ///< STRATA_REFILL_HEAD2 (-1: unset)
    int64_t head2_min = std::numeric_limits<int64_t>::max();   ///< STRATA_REFILL_HEAD2_MIN
    bool head_dp = false;               ///< STRATA_REFILL_HEAD_DP=1
    int64_t head_big = 0;               ///< STRATA_REFILL_HEAD_BIG
    int64_t head_big_chunk = 32768;     ///< STRATA_REFILL_HEAD_BIG_CHUNK
    int64_t head_tp_keep = -1;          ///< STRATA_REFILL_HEAD_TP_KEEP (-1: unset)
    int64_t head_tp_min = 0;            ///< STRATA_REFILL_HEAD_TP_MIN
};

// ---- chunk geometry (verbatim from generate.cpp's lambdas) ---------------------------------------------------------
inline int64_t request_chunk(int64_t tokens, int64_t max_chunk) {
    if (tokens <= 0 || max_chunk <= 0) return 0;
    const int64_t rounded = tokens > std::numeric_limits<int64_t>::max() - 255 ? tokens : ((tokens + 255) / 256) * 256;
    return std::min(max_chunk, rounded);
}
/// STRATA_PREFILL_EQUAL: n = ceil(tokens / max_chunk) equal chunks, except a short last chunk (below stream_all_min)
/// when max_chunk <= EQUAL_TAIL_MAX
inline int64_t equal_chunk(int64_t tokens, int64_t max_chunk, const Thresholds& t, int64_t stream_all_min) {
    const int64_t plain = request_chunk(tokens, max_chunk);
    if (!t.equal || tokens <= 0 || max_chunk <= 0) return plain;
    const int64_t n = tokens / max_chunk + (tokens % max_chunk != 0);
    const int64_t last = tokens - (n - 1) * max_chunk;
    if (n <= 1 || (last < stream_all_min && max_chunk <= t.equal_tail_max)) return plain;
    const int64_t per = tokens / n + (tokens % n != 0);
    const int64_t rounded = per > std::numeric_limits<int64_t>::max() - 255 ? per : ((per + 255) / 256) * 256;
    return std::min(max_chunk, rounded);
}
/// #2824: the chunk with the host-aware plan's `plan` (< cap), else equal_chunk at cap
inline int64_t planned_chunk(int64_t tokens, int64_t cap, int64_t plan, const Thresholds& t, int64_t stream_all_min) {
    if (plan <= 0 || plan >= cap || tokens <= 0) return equal_chunk(tokens, cap, t, stream_all_min);
    return strata::program::prefill_plan::balanced_chunk(tokens, cap, plan);
}

// ---- the 2-card paths (verbatim from generate.cpp) -------------------------------------------------------------------
/// the DP plan's chunk for a segment of n tokens: an even number of balanced chunks, multiples of 256
inline int64_t dp_step_for(int64_t n, int64_t cap) {
    int64_t k = std::max<int64_t>(2, (n + cap - 1) / cap);
    if (k & 1) ++k;
    return ((n + k - 1) / k + 255) / 256 * 256;
}
/// a segment the DP plan takes: two or more balanced chunks, the last of at least kDpMinLast tokens
inline bool dp_takes(int64_t n, int64_t cap, const Thresholds& t) {
    if (n < t.dp_min) return false;
    const int64_t step = dp_step_for(n, cap);
    const int64_t full = (n + step - 1) / step;
    return full >= 2 && n - (full - 1) * step >= kDpMinLast;
}
inline bool tpp_takes(int64_t n, const Thresholds& t) { return t.tp_on && n >= t.tp_lo && n <= t.tp_hi; }
inline int64_t tpp_chunk_for(int64_t n, int64_t cap, const Thresholds& t) {
    const int64_t C = std::min(cap, t.tp_chunk);
    const int64_t k = std::max<int64_t>(1, (n + C - 1) / C);
    return std::min(C, ((n + k - 1) / k + 255) / 256 * 256);
}
inline int64_t pipe_chunk_for(int64_t n, const Thresholds& t) {
    if (n >= t.pipe_half_min && n <= t.pipe_half_max) return ((n + 1) / 2 + 255) / 256 * 256;
    return t.pipe_chunk;
}
/// the head read_part reads on the layer pipeline before TP/DP: up to the next K/V block boundary from a0
inline int64_t dp_head(int64_t a0, int64_t idx_block) { return idx_block > 0 ? (idx_block - a0 % idx_block) % idx_block : 0; }

enum class Path : int { Windows = 0, Single = 1, Pipe = 2, TP = 3, DP = 4 };
inline const char* path_name(Path p) {
    switch (p) {
        case Path::Windows: return "windows";
        case Path::Single: return "single";
        case Path::Pipe: return "pipe";
        case Path::TP: return "tp";
        case Path::DP: return "dp";
    }
    return "?";
}

/// the facts of one part that the rules read
struct PartFacts {
    int64_t a0 = 0, b0 = 0;             ///< [a0, b0): the part
    int64_t cap = 0;                    ///< o.prefill_chunk (the buffers' chunk the VRAM scan picked)
    int64_t host_plan_chunk = 0;        ///< pf_plan_chunk (#2824; 0: none)
    int64_t stream_all_min = 1024;      ///< Prefill::stream_all_min_tokens()
    int64_t idx_block = 4;              ///< qsa_real_shapes().idx_block
    bool dp_on = false;                 ///< the DP path is linked (two stages)
    bool layer_split = false;           ///< the read runs on a layer split's stages (the pipeline), not one card
    bool tpp_pictures = false;          ///< TP_PREFILL_OWNROWS and the prompt has picture rows (run_tp declines them)
    bool lend_no_tp = false;            ///< review M1: lay this part out for the usual read after a TP refusal
    bool head_alias = false;            ///< --prefill-head-alias
    int64_t split_small = 0, split_small_max = 0;   ///< #340 STRATA_SPLIT_SMALL_OWN
    int64_t exact = 0;                  ///< a planned segment read on: the loan it was read with (0: none)
    bool planned = false;               ///< --prefill-pipe planned this segment (read_part's `planned`)
};

/// what the serve lend() lays out for the part, and the path read_part takes (computed once per part)
struct SegmentPlan {
    Path path = Path::Single;
    int64_t dp_n = 0;                   ///< tokens after the head (b0 - a0 - dp_head(a0))
    int64_t head = 0;                   ///< tokens read on the pipeline before TP/DP
    int64_t want_full = 0;              ///< the chunk the path asks for
    int64_t want = 0;                   ///< the chunk lend() lays the loan out for (before #2090 halving per card)
    bool tp_read = false;               ///< the loan carves TP's exchange buffers (Prefill::set_tp_layout)
    int64_t read_chunk = 0;             ///< TP: run_tp's chunk; DP: run_dp's cap; else 0 (the buffers' chunk)
    // lend-side and read-side decisions as the engine computes them today: lend() decides on dp_n = tokens -
    // dp_head(a0) and the lend's own flags; read_part() decides again on (b0 - a0 - dp_head(a0)) with `planned`
    // instead of the lend flags.  Both are kept so the log shows any difference between the two sites.
    Path lend_path = Path::Single;
    // filled by the engine as the part runs (facts, for the log; never inputs to a rule here)
    static constexpr int kMaxCards = 4;
    int64_t lent_chunk[kMaxCards] = {0, 0, 0, 0};   ///< per participant: the chunk its loan was laid out for (after #2090)
    int64_t lent_rows[kMaxCards] = {0, 0, 0, 0};    ///< per participant: the slots lent for this part
    int cards = 0;                                  ///< participants that lent
};

/// today's serve lend() chunk choice (generate.cpp lend(): want_full, want_seg, want)
inline void lend_side(SegmentPlan& s, const PartFacts& f, const Thresholds& t) {
    const int64_t tokens = f.b0 - f.a0;
    int64_t want_full = planned_chunk(tokens, f.cap, f.host_plan_chunk, t, f.stream_all_min);
    bool tp_read = false;
    Path lp = f.layer_split ? Path::Pipe : Path::Single;
    if (f.dp_on) {
        lp = Path::Pipe;
        const int64_t dp_n = tokens - dp_head(f.a0, f.idx_block);
        if (tpp_takes(dp_n, t) && !f.tpp_pictures && !f.lend_no_tp) {
            want_full = std::min(f.cap, tpp_chunk_for(dp_n, f.cap, t));
            tp_read = true;
            lp = Path::TP;
        } else if (dp_takes(dp_n, f.cap, t)) {
            want_full = std::min(f.cap, dp_step_for(dp_n, f.cap));
            lp = Path::DP;
        } else {
            want_full = std::min(want_full, pipe_chunk_for(tokens, t));
        }
    }
    const int64_t want_seg = f.head_alias ? f.cap
                           : f.split_small > 0 && tokens <= f.split_small_max ? std::min(want_full, f.split_small)
                                                                               : want_full;
    s.want_full = want_full;
    s.want = f.exact > 0 ? std::min(f.exact, f.cap) : want_seg;
    s.tp_read = tp_read;
    s.lend_path = lp;
}
/// today's read_part() path choice (generate.cpp read_part(): TP, then DP, then the pipeline / one stage)
inline void read_side(SegmentPlan& s, const PartFacts& f, const Thresholds& t) {
    s.head = dp_head(f.a0, f.idx_block);
    s.dp_n = f.b0 - f.a0 - s.head;
    s.read_chunk = 0;
    if (!f.planned && f.dp_on && tpp_takes(s.dp_n, t)) {
        s.path = Path::TP;
        s.read_chunk = std::min(f.cap, tpp_chunk_for(s.dp_n, f.cap, t));
    } else if (!f.planned && f.dp_on && dp_takes(s.dp_n, f.cap, t)) {
        s.path = Path::DP;
        s.read_chunk = f.cap;
    } else {
        s.path = f.layer_split ? Path::Pipe : Path::Single;
        s.head = 0;
    }
}
inline SegmentPlan legacy_plan(const PartFacts& f, const Thresholds& t) {
    SegmentPlan s;
    lend_side(s, f, t);
    read_side(s, f, t);
    return s;
}

/// today's refill head rule (generate.cpp late_issue): how many of a part's `rows` lent rows (in refill order) join
/// before the first token.  `paced`: the refill at the prompt's end (not the DP early refill); `lent_chunk`: the
/// chunk the part's loan was laid out for; `req_prompt_n`: the whole prompt's tokens.
inline size_t head_rows(const Thresholds& t, size_t rows, int64_t lent_chunk, int64_t req_prompt_n, bool paced,
                        bool dp_read_req, bool tpp_read_req) {
    int64_t head_want = t.head2 >= 0 && req_prompt_n >= t.head2_min ? t.head2 : t.head;
    if (t.head_big > 0 && lent_chunk > t.head_big_chunk) head_want = std::max(head_want, t.head_big);
    size_t head_n = paced && (!t.head_dp || dp_read_req) && req_prompt_n >= t.head_min ? std::min(rows, (size_t) head_want) : 0;
    if (paced && tpp_read_req && t.head_tp_keep >= 0 && req_prompt_n >= t.head_tp_min && rows > (size_t) t.head_tp_keep)
        head_n = std::max(head_n, rows - (size_t) t.head_tp_keep);
    if (paced && t.head_big > 0 && lent_chunk > t.head_big_chunk) head_n = std::max(head_n, std::min(rows, (size_t) t.head_big));
    return head_n;
}

}  // namespace strata::program::segment_plan
