// src/kernels/cpu/iq_avx512.cpp - plan v0.3 P6: the i-quant expert rows in 512-bit lanes, several tokens at once.
//
// ggml-cpu's x86 dot products for these formats are AVX2 and single-token: every token re-decodes the weights
// and a sign is applied with five instructions per 32 values.  Here 64 values are decoded once (8 or 16 grid
// lookups, one 64-bit sign mask, one vector of scales) and every token applies them with a masked subtract, a
// `maddubs` and a `madd`.  The arithmetic is ggml's (ggml-cpu/quants.c, the `_generic` references): integer sums
// per block, times d_x * d_y * the format's constant - only the order of the float additions differs.
//
// Formats: IQ2_XXS (16), IQ2_XS (17), IQ3_XXS (18), IQ3_S (21), IQ2_S (22).  IQ1_M stays on ggml-cpu (no shipped
// model has IQ1_M expert rows: the 'Coder IQ1_M' pack's gate/up are IQ2_S / IQ3_XXS / IQ3_S).
#include "strata/kernels/cpu/iq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "ggml-common.h"

#include <immintrin.h>
#if !defined(_MSC_VER)
#include <cpuid.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstring>

namespace strata::kernels::cpu {
namespace {

inline float h2f(uint16_t h) { return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) h))); }
inline uint32_t u32(const uint8_t* p) { uint32_t v; std::memcpy(&v, p, 4); return v; }
inline uint16_t u16(const uint8_t* p) { uint16_t v; std::memcpy(&v, p, 2); return v; }
inline uint64_t u64(const uint8_t* p) { uint64_t v; std::memcpy(&v, p, 8); return v; }

// E-2: the IQ3 grids by one AVX-512 gather (the same words, the same results) - measured 3-5% SLOWER than the scalar
// lookups on a Ryzen 5 7600 (Zen 4 gathers are microcoded), so opt-in: STRATA_IQ_GATHER=1
const bool gather = std::getenv("STRATA_IQ_GATHER") != nullptr;
// E-2: software prefetch distance in bytes (STRATA_IQ_PREFETCH; 0 = off): -2-3% gate/up time on IQ3_S decode
// (the rows are read at ~30 GB/s by 6 cores, near what DDR5 with 4 KB pages gives: the pool is memory-bound)
const int prefetch_ahead = [] {
    const char* v = std::getenv("STRATA_IQ_PREFETCH");
    return v ? std::atoi(v) : 2048;
}();

// lanes 0-7 -> s0, 8-15 -> s1, 16-23 -> s2, 24-31 -> s3 (int16 lanes of a maddubs result: two values each)
inline __m512i scales4(int s0, int s1, int s2, int s3) {
    const __m512i idx = _mm512_set_epi16(3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1,
                                         0, 0, 0, 0, 0, 0, 0, 0);
    const uint64_t packed = (uint64_t) (uint16_t) s0 | ((uint64_t) (uint16_t) s1 << 16) |
                            ((uint64_t) (uint16_t) s2 << 32) | ((uint64_t) (uint16_t) s3 << 48);
    return _mm512_permutexvar_epi16(idx, _mm512_set1_epi64((long long) packed));
}

// ---- per format: 64 values (chunk j of a 256-value block) -> grid magnitudes, sign mask, scales
template <int TY> struct Fmt;

template <> struct Fmt<16> {   // IQ2_XXS: d, qs[32] u16
    static constexpr int bytes = 66;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t a0 = u32(q), a1 = u32(q + 4), b0 = u32(q + 8), b1 = u32(q + 12);
        g = _mm512_set_epi64((long long) iq2xxs_grid[b0 >> 24], (long long) iq2xxs_grid[(b0 >> 16) & 255],
                             (long long) iq2xxs_grid[(b0 >> 8) & 255], (long long) iq2xxs_grid[b0 & 255],
                             (long long) iq2xxs_grid[a0 >> 24], (long long) iq2xxs_grid[(a0 >> 16) & 255],
                             (long long) iq2xxs_grid[(a0 >> 8) & 255], (long long) iq2xxs_grid[a0 & 255]);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a1 & 127] | ((uint64_t) ksigns_iq2xs[(a1 >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a1 >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a1 >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[b1 & 127] << 32) | ((uint64_t) ksigns_iq2xs[(b1 >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(b1 >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(b1 >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a1 >> 28) + 1, sb = 2 * (int) (b1 >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<17> {   // IQ2_XS: d, qs[32] u16 (9-bit grid index + 7-bit sign index), scales[8]
    static constexpr int bytes = 74;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        uint16_t v[8];
        std::memcpy(v, q, 16);
        g = _mm512_set_epi64((long long) iq2xs_grid[v[7] & 511], (long long) iq2xs_grid[v[6] & 511],
                             (long long) iq2xs_grid[v[5] & 511], (long long) iq2xs_grid[v[4] & 511],
                             (long long) iq2xs_grid[v[3] & 511], (long long) iq2xs_grid[v[2] & 511],
                             (long long) iq2xs_grid[v[1] & 511], (long long) iq2xs_grid[v[0] & 511]);
        uint64_t s = 0;
        for (int l = 0; l < 8; ++l) s |= (uint64_t) ksigns_iq2xs[v[l] >> 9] << (8 * l);
        m = _cvtu64_mask64(s);
        const uint8_t s0 = b[66 + 2 * j], s1 = b[66 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<22> {   // IQ2_S: d, qs[64] (32 grid bytes, 32 sign bytes), qh[8], scales[8]
    static constexpr int bytes = 82;
    static constexpr float K = 0.125f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* qs = b + 2 + 8 * j;
        const uint8_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        g = _mm512_set_epi64((long long) iq2s_grid[qs[7] | ((h1 << 2) & 0x300)], (long long) iq2s_grid[qs[6] | ((h1 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[5] | ((h1 << 6) & 0x300)], (long long) iq2s_grid[qs[4] | ((h1 << 8) & 0x300)],
                             (long long) iq2s_grid[qs[3] | ((h0 << 2) & 0x300)], (long long) iq2s_grid[qs[2] | ((h0 << 4) & 0x300)],
                             (long long) iq2s_grid[qs[1] | ((h0 << 6) & 0x300)], (long long) iq2s_grid[qs[0] | ((h0 << 8) & 0x300)]);
        m = _cvtu64_mask64(u64(b + 2 + 32 + 8 * j));
        const uint8_t s0 = b[74 + 2 * j], s1 = b[74 + 2 * j + 1];
        sc = scales4(2 * (s0 & 15) + 1, 2 * (s0 >> 4) + 1, 2 * (s1 & 15) + 1, 2 * (s1 >> 4) + 1);
    }
};

template <> struct Fmt<18> {   // IQ3_XXS: d, qs[64] grid bytes, 8 x u32 (4 x 7-bit sign index + 4-bit scale)
    static constexpr int bytes = 98;
    static constexpr float K = 0.25f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        // E-2: one gather of the 16 grid words (the scalar lookups assembled with set_epi32 cost ~30 uops)
        g = gather ? _mm512_i32gather_epi32(_mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q)), iq3xxs_grid, 4)
                   : _mm512_set_epi32((int) iq3xxs_grid[q[15]], (int) iq3xxs_grid[q[14]], (int) iq3xxs_grid[q[13]], (int) iq3xxs_grid[q[12]],
                             (int) iq3xxs_grid[q[11]], (int) iq3xxs_grid[q[10]], (int) iq3xxs_grid[q[9]], (int) iq3xxs_grid[q[8]],
                             (int) iq3xxs_grid[q[7]], (int) iq3xxs_grid[q[6]], (int) iq3xxs_grid[q[5]], (int) iq3xxs_grid[q[4]],
                             (int) iq3xxs_grid[q[3]], (int) iq3xxs_grid[q[2]], (int) iq3xxs_grid[q[1]], (int) iq3xxs_grid[q[0]]);
        const uint32_t a = u32(b + 2 + 64 + 8 * j), c = u32(b + 2 + 64 + 8 * j + 4);
        const uint64_t s = (uint64_t) ksigns_iq2xs[a & 127] | ((uint64_t) ksigns_iq2xs[(a >> 7) & 127] << 8) |
                           ((uint64_t) ksigns_iq2xs[(a >> 14) & 127] << 16) | ((uint64_t) ksigns_iq2xs[(a >> 21) & 127] << 24) |
                           ((uint64_t) ksigns_iq2xs[c & 127] << 32) | ((uint64_t) ksigns_iq2xs[(c >> 7) & 127] << 40) |
                           ((uint64_t) ksigns_iq2xs[(c >> 14) & 127] << 48) | ((uint64_t) ksigns_iq2xs[(c >> 21) & 127] << 56);
        m = _cvtu64_mask64(s);
        const int sa = 2 * (int) (a >> 28) + 1, sb = 2 * (int) (c >> 28) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <> struct Fmt<21> {   // IQ3_S: d, qs[64], qh[8], signs[32], scales[4]
    static constexpr int bytes = 110;
    static constexpr float K = 1.0f;
    static inline void decode(const uint8_t* b, int j, __m512i& g, __mmask64& m, __m512i& sc) {
        const uint8_t* q = b + 2 + 16 * j;
        const uint32_t h0 = b[66 + 2 * j], h1 = b[66 + 2 * j + 1];
        if (gather) {   // E-2: the 9-bit indices (the high bit from qh under a mask), one gather
            __m512i idx = _mm512_cvtepu8_epi32(_mm_loadu_si128((const __m128i*) q));
            idx = _mm512_mask_add_epi32(idx, (__mmask16) (h0 | (h1 << 8)), idx, _mm512_set1_epi32(256));
            g = _mm512_i32gather_epi32(idx, iq3s_grid, 4);
        } else {
#define G3(k, h, kk) (int) iq3s_grid[q[k] | (((h >> kk) & 1) << 8)]
        g = _mm512_set_epi32(G3(15, h1, 7), G3(14, h1, 6), G3(13, h1, 5), G3(12, h1, 4),
                             G3(11, h1, 3), G3(10, h1, 2), G3(9, h1, 1), G3(8, h1, 0),
                             G3(7, h0, 7), G3(6, h0, 6), G3(5, h0, 5), G3(4, h0, 4),
                             G3(3, h0, 3), G3(2, h0, 2), G3(1, h0, 1), G3(0, h0, 0));
#undef G3
        }
        m = _cvtu64_mask64(u64(b + 74 + 8 * j));
        const uint8_t s = b[106 + j];
        const int sa = 2 * (s & 15) + 1, sb = 2 * (s >> 4) + 1;
        sc = scales4(sa, sa, sb, sb);
    }
};

template <int TY, int NT>
inline void row_dot(const uint8_t* row, int nblocks, const block_q8_K* const* y, float* res) {
    __m512 accf[NT];
    for (int t = 0; t < NT; ++t) accf[t] = _mm512_setzero_ps();
    const __m512i zero = _mm512_setzero_si512();
    for (int i = 0; i < nblocks; ++i) {
        const uint8_t* blk = row + (size_t) i * Fmt<TY>::bytes;
        // E-2: the weight rows stream from DRAM (4 KB pages when large pages are refused): ask for the bytes a few
        // blocks ahead (the next row's start included) before the decode needs them
        if (prefetch_ahead > 0) {
            const uint8_t* ahead = blk + (size_t) prefetch_ahead;
            _mm_prefetch((const char*) ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) ahead + 64, _MM_HINT_T0);
        }
        __m512i acci[NT];
        for (int t = 0; t < NT; ++t) acci[t] = _mm512_setzero_si512();
        for (int j = 0; j < 4; ++j) {
            __m512i g, sc;
            __mmask64 m;
            Fmt<TY>::decode(blk, j, g, m, sc);
            for (int t = 0; t < NT; ++t) {
                const __m512i yv = _mm512_loadu_si512((const void*) (y[t][i].qs + 64 * j));
                const __m512i ys = _mm512_mask_sub_epi8(yv, m, zero, yv);
                acci[t] = _mm512_add_epi32(acci[t], _mm512_madd_epi16(_mm512_maddubs_epi16(g, ys), sc));
            }
        }
        const float dx = h2f(u16(blk)) * Fmt<TY>::K;
        for (int t = 0; t < NT; ++t)
            accf[t] = _mm512_fmadd_ps(_mm512_set1_ps(dx * y[t][i].d), _mm512_cvtepi32_ps(acci[t]), accf[t]);
    }
    for (int t = 0; t < NT; ++t) res[t] = _mm512_reduce_add_ps(accf[t]);
}

template <int TY, int NT>
void gu_rows(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, float* const* ff,
             int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    const int nb = n / QK_K;
    float g[NT], u[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(blob + (size_t) r * gu_row, nb, y, g);
        row_dot<TY, NT>(blob + up_off + (size_t) r * gu_row, nb, y, u);
        for (int t = 0; t < NT; ++t) ff[t][r] = (g[t] / (1.f + std::exp(-g[t]))) * u[t];
    }
}

template <int TY, int NT>
void dot_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0, int r1) {
    const block_q8_K* y[NT];
    for (int t = 0; t < NT; ++t) y[t] = (const block_q8_K*) act[t];
    float res[NT];
    for (int r = r0; r < r1; ++r) {
        row_dot<TY, NT>(w + (size_t) r * row_bytes, n / QK_K, y, res);
        for (int t = 0; t < NT; ++t) out[t][r] = res[t];
    }
}

template <int TY>
void gu_rows_nt(int nt, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act,
                float* const* ff, int r0, int r1) {
    switch (nt) {
        case 1: gu_rows<TY, 1>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 2: gu_rows<TY, 2>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 3: gu_rows<TY, 3>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 4: gu_rows<TY, 4>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 5: gu_rows<TY, 5>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 6: gu_rows<TY, 6>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 7: gu_rows<TY, 7>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: gu_rows<TY, 8>(blob, gu_row, up_off, n, act, ff, r0, r1); break;
    }
}

template <int TY>
void dot_rows_nt(int nt, const uint8_t* w, size_t row_bytes, int n, const void* const* act, float* const* out, int r0,
                 int r1) {
    switch (nt) {
        case 1: dot_rows<TY, 1>(w, row_bytes, n, act, out, r0, r1); break;
        case 2: dot_rows<TY, 2>(w, row_bytes, n, act, out, r0, r1); break;
        case 3: dot_rows<TY, 3>(w, row_bytes, n, act, out, r0, r1); break;
        case 4: dot_rows<TY, 4>(w, row_bytes, n, act, out, r0, r1); break;
        default: for (int t0 = 0; t0 < nt; t0 += 4) {
            const int k = nt - t0 < 4 ? nt - t0 : 4;
            dot_rows_nt<TY>(k, w, row_bytes, n, act + t0, out + t0, r0, r1);
        }
    }
}

// ---- IQ3_S, one token, bitwise ggml-cpu (2026-10-15)
// ggml's AVX2 dot builds each 8-entry grid vector from 8 scalar loads (~2 uops an entry) and keeps 8 int32 lanes:
// lane k = sum over the block's 8 sub-blocks of scale_sb * (entry k of sub-block sb) . (its 4 activations).  Here the
// grid is split into 4 byte planes (byte b of every word), each looked up for the block's 64 indices with 4 vpermt2b
// and 3 masked blends (index bit 7 from the qs byte, bit 8 from qh).  Two unpack rounds turn the planes into vectors
// u_m whose dword (lane L, q) is entry 16L + 4m + q; the activation is permuted to match once per call and the
// signs by one word transpose.  The integer sums land in exactly ggml's 8 lanes (integer adds are exact in any
// order) and the float FMA and horizontal sum are ggml's, so every result is ggml's to the bit.
alignas(64) uint8_t iq3s_plane[4][512];
#if defined(__GNUC__) && !defined(__clang__)
// This translation unit is compiled with AVX-512 enabled.  Keep support probes and table builders AVX2-safe:
// they can be called before dispatch, and a global table initializer is vectorized with ZMM moves on GCC.
#define STRATA_AVX512_SAFE_INIT \
    __attribute__((target("no-avx512f,no-avx512bw,no-avx512vl,no-avx512dq,no-avx512vnni,no-avx512vbmi"), noinline))
#else
#define STRATA_AVX512_SAFE_INIT
#endif
STRATA_AVX512_SAFE_INIT static bool iq3s_planes_ready() noexcept {
    static const bool ready = [] {
        for (int e = 0; e < 512; ++e)
            for (int b = 0; b < 4; ++b) iq3s_plane[b][e] = (uint8_t) (iq3s_grid[e] >> (8 * b));
        return true;
    }();
    return ready;
}

inline float hsum_float_8_ggml(const __m256 x) {   // ggml-cpu/arch/x86/quants.c hsum_float_8, verbatim
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

// yp[256 i + 64 m + 16 L + j] = y[i].qs[64 L + 16 m + j]: the activations in u_m's order
inline void iq3s_permute_act(const block_q8_K* y, int nb, int8_t* yp) {
    for (int i = 0; i < nb; ++i)
        for (int m = 0; m < 4; ++m)
            for (int L = 0; L < 4; ++L) std::memcpy(yp + 256 * i + 64 * m + 16 * L, y[i].qs + 64 * L + 16 * m, 16);
}

inline float iq3s_row_1t(const uint8_t* row, int nb, const block_q8_K* y, const int8_t* yp) {
    const __m512i zero = _mm512_setzero_si512();
    const __m256i widx = _mm256_setr_epi16(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    const __m512i* T0 = (const __m512i*) iq3s_plane[0];
    const __m512i* T1 = (const __m512i*) iq3s_plane[1];
    const __m512i* T2 = (const __m512i*) iq3s_plane[2];
    const __m512i* T3 = (const __m512i*) iq3s_plane[3];
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = row + (size_t) i * 110;   // d, qs[64], qh[8], signs[32], scales[4]
        if (prefetch_ahead > 0) {
            _mm_prefetch((const char*) b + prefetch_ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) b + prefetch_ahead + 64, _MM_HINT_T0);
        }
        const __m512i qs = _mm512_loadu_si512((const void*) (b + 2));
        const __mmask64 m7 = _mm512_movepi8_mask(qs);
        const __mmask64 m8 = _cvtu64_mask64(u64(b + 66));
        auto plane = [&](const __m512i* T) {
            const __m512i r0 = _mm512_permutex2var_epi8(_mm512_load_si512(T + 0), qs, _mm512_load_si512(T + 1));
            const __m512i r1 = _mm512_permutex2var_epi8(_mm512_load_si512(T + 2), qs, _mm512_load_si512(T + 3));
            const __m512i r2 = _mm512_permutex2var_epi8(_mm512_load_si512(T + 4), qs, _mm512_load_si512(T + 5));
            const __m512i r3 = _mm512_permutex2var_epi8(_mm512_load_si512(T + 6), qs, _mm512_load_si512(T + 7));
            return _mm512_mask_blend_epi8(m8, _mm512_mask_blend_epi8(m7, r0, r1), _mm512_mask_blend_epi8(m7, r2, r3));
        };
        const __m512i p0 = plane(T0), p1 = plane(T1), p2 = plane(T2), p3 = plane(T3);
        const __m512i t0 = _mm512_unpacklo_epi8(p0, p1), t1 = _mm512_unpackhi_epi8(p0, p1);
        const __m512i t2 = _mm512_unpacklo_epi8(p2, p3), t3 = _mm512_unpackhi_epi8(p2, p3);
        const __m512i u0 = _mm512_unpacklo_epi16(t0, t2), u1 = _mm512_unpackhi_epi16(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi16(t1, t3), u3 = _mm512_unpackhi_epi16(t1, t3);
        alignas(32) uint64_t sm[4];
        _mm256_store_si256((__m256i*) sm,
                           _mm256_permutexvar_epi16(widx, _mm256_loadu_si256((const __m256i*) (b + 74))));
        const uint8_t s0 = b[106], s1 = b[107], s2 = b[108], s3 = b[109];
        const __m512i sce = scales4(2 * (s0 & 15) + 1, 2 * (s1 & 15) + 1, 2 * (s2 & 15) + 1, 2 * (s3 & 15) + 1);
        const __m512i sco = scales4(2 * (s0 >> 4) + 1, 2 * (s1 >> 4) + 1, 2 * (s2 >> 4) + 1, 2 * (s3 >> 4) + 1);
        const int8_t* a = yp + 256 * i;
        auto part = [&](const __m512i& u, int m, const __m512i& sc) {
            const __m512i yv = _mm512_loadu_si512((const void*) (a + 64 * m));
            const __m512i ys = _mm512_mask_sub_epi8(yv, _cvtu64_mask64(sm[m]), zero, yv);
            return _mm512_madd_epi16(_mm512_maddubs_epi16(u, ys), sc);
        };
        const __m512i lo = _mm512_add_epi32(part(u0, 0, sce), part(u2, 2, sco));   // entries k = 0..3
        const __m512i hi = _mm512_add_epi32(part(u1, 1, sce), part(u3, 3, sco));   // entries k = 4..7
        const __m256i l2 = _mm256_add_epi32(_mm512_castsi512_si256(lo), _mm512_extracti64x4_epi64(lo, 1));
        const __m256i h2 = _mm256_add_epi32(_mm512_castsi512_si256(hi), _mm512_extracti64x4_epi64(hi, 1));
        const __m128i l4 = _mm_add_epi32(_mm256_castsi256_si128(l2), _mm256_extracti128_si256(l2, 1));
        const __m128i h4 = _mm_add_epi32(_mm256_castsi256_si128(h2), _mm256_extracti128_si256(h2, 1));
        const __m256i sumi = _mm256_set_m128i(h4, l4);
        const float d = h2f(u16(b)) * y[i].d;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accf);
    }
    return hsum_float_8_ggml(accf);
}

// ---- IQ3_XXS, one token, bitwise ggml-cpu's ggml_vec_dot_iq3_xxs_q8_K AVX2 path (2026-10-15)
// The same byte-plane decode with a 256-entry grid (2 vpermt2b + 1 blend a plane), the sign bytes from the 7-bit
// keven_signs indices with one vpmultishiftqb and one vpermt2b into a table of their sign bits, ggml's lanes,
// FMA order and final 0.25 * hsum.
alignas(64) uint8_t iq3xxs_plane[4][256];
alignas(64) uint8_t ksigns_bits[128];   // ksigns_iq2xs: bit b = the sign of weight b (ggml-cpu's keven_signs_q2xs bytes)
STRATA_AVX512_SAFE_INIT static bool iq3xxs_tables_ready() noexcept {
    static const bool ready = [] {
        for (int e = 0; e < 256; ++e)
            for (int b = 0; b < 4; ++b) iq3xxs_plane[b][e] = (uint8_t) (iq3xxs_grid[e] >> (8 * b));
        for (int i = 0; i < 128; ++i) ksigns_bits[i] = ksigns_iq2xs[i];
        return true;
    }();
    return ready;
}

inline float iq3xxs_row_1t(const uint8_t* row, int nb, const block_q8_K* y, const int8_t* yp) {
    const __m512i zero = _mm512_setzero_si512();
    const __m256i widx = _mm256_setr_epi16(0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15);
    // byte offsets 0, 7, 14, 21 of each 32-bit half: the four 7-bit sign indices of a sub-block
    const __m512i msh = _mm512_set1_epi64(0x352E2720150E0700ll);
    const __m512i T7 = _mm512_set1_epi8(127);
    const __m512i sbA = _mm512_load_si512((const void*) ksigns_bits), sbB = _mm512_load_si512((const void*) (ksigns_bits + 64));
    const __m512i* T0 = (const __m512i*) iq3xxs_plane[0];
    const __m512i* T1 = (const __m512i*) iq3xxs_plane[1];
    const __m512i* T2 = (const __m512i*) iq3xxs_plane[2];
    const __m512i* T3 = (const __m512i*) iq3xxs_plane[3];
    __m256 accf = _mm256_setzero_ps();
    for (int i = 0; i < nb; ++i) {
        const uint8_t* b = row + (size_t) i * 98;   // d, qs[64] grid indices, 8 x u32 signs + scale
        if (prefetch_ahead > 0) {
            _mm_prefetch((const char*) b + prefetch_ahead, _MM_HINT_T0);
            _mm_prefetch((const char*) b + prefetch_ahead + 64, _MM_HINT_T0);
        }
        const __m512i qs = _mm512_loadu_si512((const void*) (b + 2));
        const __mmask64 m7 = _mm512_movepi8_mask(qs);
        auto plane = [&](const __m512i* T) {
            const __m512i r0 = _mm512_permutex2var_epi8(_mm512_load_si512(T + 0), qs, _mm512_load_si512(T + 1));
            const __m512i r1 = _mm512_permutex2var_epi8(_mm512_load_si512(T + 2), qs, _mm512_load_si512(T + 3));
            return _mm512_mask_blend_epi8(m7, r0, r1);
        };
        const __m512i p0 = plane(T0), p1 = plane(T1), p2 = plane(T2), p3 = plane(T3);
        const __m512i t0 = _mm512_unpacklo_epi8(p0, p1), t1 = _mm512_unpackhi_epi8(p0, p1);
        const __m512i t2 = _mm512_unpacklo_epi8(p2, p3), t3 = _mm512_unpackhi_epi8(p2, p3);
        const __m512i u0 = _mm512_unpacklo_epi16(t0, t2), u1 = _mm512_unpackhi_epi16(t0, t2);
        const __m512i u2 = _mm512_unpacklo_epi16(t1, t3), u3 = _mm512_unpackhi_epi16(t1, t3);
        // the 32 sign bytes (weights 8j..8j+7 of sub-block sb at byte 4 sb + j) in natural order, then u_m's masks
        const __m256i aux = _mm256_loadu_si256((const __m256i*) (b + 66));
        const __m512i sidx = _mm512_and_si512(_mm512_multishift_epi64_epi8(msh, _mm512_castsi256_si512(aux)), T7);
        const __m512i sbits = _mm512_permutex2var_epi8(sbA, sidx, sbB);
        alignas(32) uint64_t sm[4];
        _mm256_store_si256((__m256i*) sm, _mm256_permutexvar_epi16(widx, _mm512_castsi512_si256(sbits)));
        uint32_t a32[8];
        std::memcpy(a32, b + 66, 32);
        const __m512i sce = scales4(2 * (int) (a32[0] >> 28) + 1, 2 * (int) (a32[2] >> 28) + 1, 2 * (int) (a32[4] >> 28) + 1,
                                    2 * (int) (a32[6] >> 28) + 1);
        const __m512i sco = scales4(2 * (int) (a32[1] >> 28) + 1, 2 * (int) (a32[3] >> 28) + 1, 2 * (int) (a32[5] >> 28) + 1,
                                    2 * (int) (a32[7] >> 28) + 1);
        const int8_t* a = yp + 256 * i;
        auto part = [&](const __m512i& u, int m, const __m512i& sc) {
            const __m512i yv = _mm512_loadu_si512((const void*) (a + 64 * m));
            const __m512i ys = _mm512_mask_sub_epi8(yv, _cvtu64_mask64(sm[m]), zero, yv);
            return _mm512_madd_epi16(_mm512_maddubs_epi16(u, ys), sc);
        };
        const __m512i lo = _mm512_add_epi32(part(u0, 0, sce), part(u2, 2, sco));
        const __m512i hi = _mm512_add_epi32(part(u1, 1, sce), part(u3, 3, sco));
        const __m256i l2 = _mm256_add_epi32(_mm512_castsi512_si256(lo), _mm512_extracti64x4_epi64(lo, 1));
        const __m256i h2 = _mm256_add_epi32(_mm512_castsi512_si256(hi), _mm512_extracti64x4_epi64(hi, 1));
        const __m128i l4 = _mm_add_epi32(_mm256_castsi256_si128(l2), _mm256_extracti128_si256(l2, 1));
        const __m128i h4 = _mm_add_epi32(_mm256_castsi256_si128(h2), _mm256_extracti128_si256(h2, 1));
        const __m256i sumi = _mm256_set_m128i(h4, l4);
        const float d = h2f(u16(b)) * y[i].d;
        accf = _mm256_fmadd_ps(_mm256_set1_ps(d), _mm256_cvtepi32_ps(sumi), accf);
    }
    return 0.25f * hsum_float_8_ggml(accf);
}

}  // namespace

STRATA_AVX512_SAFE_INIT bool iq3xxs_1t_supported() noexcept {
    return iq3s_1t_supported() && iq3xxs_tables_ready();
}

void iq3xxs_gu_rows_1t(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* act, float* ff, int r0,
                       int r1) {
    const block_q8_K* y = (const block_q8_K*) act;
    const int nb = n / QK_K;
    alignas(64) thread_local int8_t yp[16384];
    if (n > 16384) return;
    iq3s_permute_act(y, nb, yp);
    for (int r = r0; r < r1; ++r) {
        const float g = iq3xxs_row_1t(blob + (size_t) r * gu_row, nb, y, yp);
        const float u = iq3xxs_row_1t(blob + up_off + (size_t) r * gu_row, nb, y, yp);
        ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
}

STRATA_AVX512_SAFE_INIT bool iq3s_1t_supported() noexcept {
    static const bool ok = [] {
        unsigned a = 0, b = 0, c = 0, d = 0;
#if defined(_MSC_VER)
        int x[4];
        __cpuidex(x, 7, 0);
        a = (unsigned) x[0]; b = (unsigned) x[1]; c = (unsigned) x[2]; d = (unsigned) x[3];
#else
        __cpuid_count(7, 0, a, b, c, d);
#endif
        return ((b >> 30) & 1u) && ((b >> 16) & 1u) && ((c >> 1) & 1u);   // AVX512BW, F, VBMI
    }();
    return ok && iq3s_planes_ready();
}

void iq3s_gu_rows_1t(const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* act, float* ff, int r0,
                     int r1) {
    const block_q8_K* y = (const block_q8_K*) act;
    const int nb = n / QK_K;
    alignas(64) thread_local int8_t yp[16384];
    if (n > 16384) return;
    iq3s_permute_act(y, nb, yp);
    for (int r = r0; r < r1; ++r) {
        const float g = iq3s_row_1t(blob + (size_t) r * gu_row, nb, y, yp);
        const float u = iq3s_row_1t(blob + up_off + (size_t) r * gu_row, nb, y, yp);
        ff[r] = (g / (1.f + std::exp(-g))) * u;
    }
}

void iq3s_rows_1t(const uint8_t* w, size_t row_bytes, int n, const void* act, float* out, int r0, int r1) {
    const block_q8_K* y = (const block_q8_K*) act;
    const int nb = n / QK_K;
    alignas(64) thread_local int8_t yp[16384];
    if (n > 16384) return;
    iq3s_permute_act(y, nb, yp);
    for (int r = r0; r < r1; ++r) out[r] = iq3s_row_1t(w + (size_t) r * row_bytes, nb, y, yp);
}

void iq512_gu_rows(int type, const uint8_t* blob, size_t gu_row, size_t up_off, int n, const void* const* act, int nt,
                   float* const* ff, int r0, int r1) {
    switch (type) {
        case 16: gu_rows_nt<16>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 17: gu_rows_nt<17>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 18: gu_rows_nt<18>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 21: gu_rows_nt<21>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        case 22: gu_rows_nt<22>(nt, blob, gu_row, up_off, n, act, ff, r0, r1); break;
        default: break;
    }
}

void iq512_rows(int type, const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                int r0, int r1) {
    switch (type) {
        case 16: dot_rows_nt<16>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 17: dot_rows_nt<17>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 18: dot_rows_nt<18>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 21: dot_rows_nt<21>(nt, w, row_bytes, n, act, out, r0, r1); break;
        case 22: dot_rows_nt<22>(nt, w, row_bytes, n, act, out, r0, r1); break;
        default: break;
    }
}

}  // namespace strata::kernels::cpu
