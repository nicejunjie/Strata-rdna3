// [claude] Timing bench for the gfx11 FP16 prompt-expert kernels (fused::experts_native_f16, STRATA_PF_F16=2):
// one batch of E experts, N tokens routed top-K uniformly over them, synthetic blobs (a constant byte pattern, so
// the scales are small finite halves). Prints ms for gate/up and down (a hook event between the two launches).
//   hip_pf_f16_bench GU_TYPE D_TYPE [N=8192] [E=128] [K=10] [REPS=20]
// ggml types: IQ3_S 21, IQ4_XS 23, Q4_K 12, Q5_K 13 (gate/up); IQ4_NL 20, Q5_1 7, Q8_0 8 (down).
#include <cuda_runtime.h>
#include "strata/prefill/moe_fused.hpp"
#include "strata/prefill/moe_fused_iq.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace f = strata::prefill::fused;

static int block_bytes(int t) {
    switch (t) {
        case 21: return 110;   // IQ3_S / 256
        case 23: return 136;   // IQ4_XS / 256
        case 12: return 144;   // Q4_K / 256
        case 13: return 176;   // Q5_K / 256
        case 18: return 98;    // IQ3_XXS / 256
        case 22: return 82;    // IQ2_S / 256
        case 20: return 18;    // IQ4_NL / 32
        case 7: return 24;     // Q5_1 / 32
        case 8: return 34;     // Q8_0 / 32
        default: return 0;
    }
}
static int block_vals(int t) { return (t == 20 || t == 7 || t == 8) ? 32 : 256; }

static cudaEvent_t g_mid;
static void mid_hook(void* s) { cudaEventRecord(g_mid, (cudaStream_t) s); }

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", #x, cudaGetErrorString(e_)); return 1; } } while (0)

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s GU_TYPE D_TYPE [N] [E] [K] [REPS]\n", argv[0]); return 2; }
    const int gu = std::atoi(argv[1]), dt = std::atoi(argv[2]);
    const int64_t N = argc > 3 ? std::atoll(argv[3]) : 8192;
    const int E = argc > 4 ? std::atoi(argv[4]) : 128, K = argc > 5 ? std::atoi(argv[5]) : 10;
    const int reps = argc > 6 ? std::atoi(argv[6]) : 20;
    if (!getenv("STRATA_PF_F16")) setenv("STRATA_PF_F16", "2", 1);
    setenv("STRATA_PF_FUSED_KQ", "1", 0);
    if (!f::f16_on()) { std::fprintf(stderr, "FP16 expert path not available here\n"); return 77; }
    if (block_bytes(gu) == 0 || block_bytes(dt) == 0 || E > f::kMaxBatch) { std::fprintf(stderr, "bad types or E\n"); return 2; }
    f::NativeGeom g;
    g.gu_type = gu; g.d_type = dt;
    g.gu_row = (size_t) (2560 / block_vals(gu)) * block_bytes(gu);
    g.d_row = (size_t) (640 / block_vals(dt)) * block_bytes(dt);
    g.up_off = 640 * g.gu_row;
    g.down_off = 1280 * g.gu_row;
    const size_t blob = g.down_off + 2560 * g.d_row;
    uint8_t* blobs = nullptr;
    CK(cudaMalloc(&blobs, blob * E));
    CK(cudaMemset(blobs, 0x11, blob * E));
    f::Batch b;
    b.e0 = 0; b.e1 = E;
    for (int e = 0; e < E; ++e) b.blob[e] = blobs + (size_t) e * blob;
    // routing: K distinct experts a token, uniform
    std::vector<int32_t> ids((size_t) N * K);
    std::mt19937 rng(7);
    for (int64_t t = 0; t < N; ++t) {
        for (int j = 0; j < K; ++j) {
            int e;
            bool dup;
            do { e = (int) (rng() % (unsigned) E); dup = false; for (int i = 0; i < j; ++i) dup |= ids[t * K + i] == e; } while (dup);
            ids[t * K + j] = e;
        }
    }
    const int64_t n = N * K;
    int32_t *dids, *slot, *src;
    void* scratch;
    CK(cudaMalloc(&dids, n * 4));
    CK(cudaMalloc(&slot, n * 4));
    CK(cudaMalloc(&src, n * 4));
    CK(cudaMalloc(&scratch, f::group_bytes(n, E)));
    CK(cudaMemcpy(dids, ids.data(), n * 4, cudaMemcpyHostToDevice));
    cudaStream_t s;
    CK(cudaStreamCreate(&s));
    f::group(dids, n, K, E, scratch, slot, src, s);
    std::vector<float> xh((size_t) N * 2560);
    std::normal_distribution<float> nd(0.0f, 1.0f);
    for (auto& v : xh) v = nd(rng);
    float* dx;
    void *dxh, *dh;
    float* dm;
    CK(cudaMalloc(&dx, xh.size() * 4));
    CK(cudaMalloc(&dxh, xh.size() * 2));
    CK(cudaMalloc(&dh, (size_t) n * 640 * 2));
    CK(cudaMalloc(&dm, (size_t) n * 2560 * 4));
    CK(cudaMemcpy(dx, xh.data(), xh.size() * 4, cudaMemcpyHostToDevice));
    f::act_f16(dx, N, 2560, dxh, s);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0)); CK(cudaEventCreate(&e1)); CK(cudaEventCreate(&g_mid));
    f::set_mid_hook(mid_hook, (void*) s);
    for (int w = 0; w < 3; ++w) f::experts_native_f16(b, g, E, n, scratch, dxh, src, dh, dm, s);
    CK(cudaStreamSynchronize(s));
    double gu_ms = 0, d_ms = 0;
    for (int r = 0; r < reps; ++r) {
        CK(cudaEventRecord(e0, s));
        f::experts_native_f16(b, g, E, n, scratch, dxh, src, dh, dm, s);
        CK(cudaEventRecord(e1, s));
        CK(cudaEventSynchronize(e1));
        float a = 0, c = 0;
        cudaEventElapsedTime(&a, e0, g_mid);
        cudaEventElapsedTime(&c, g_mid, e1);
        gu_ms += a; d_ms += c;
    }
    gu_ms /= reps; d_ms /= reps;
    const double fl_gu = 2.0 * (double) n * 2560 * 1280, fl_d = 2.0 * (double) n * 640 * 2560;
    // a hash of the down rows (the whole chain's output), to compare builds bit for bit
    std::vector<uint32_t> hd((size_t) n * 2560);
    CK(cudaMemcpy(hd.data(), dm, hd.size() * 4, cudaMemcpyDeviceToHost));
    unsigned long long hh = 0;   // XOR of per-row hashes: the grouping's slot order (atomics) does not change it
    for (int64_t r = 0; r < n; ++r) {
        unsigned long long hr = 1469598103934665603ull;
        for (int c = 0; c < 2560; ++c) hr = (hr ^ hd[(size_t) r * 2560 + c]) * 1099511628211ull;
        hh ^= hr;
    }
    std::printf("gu %d d %d N %lld E %d K %d rows/expert %.0f | gate/up %.3f ms (%.1f TFLOPS) | down %.3f ms (%.1f TFLOPS) | hash %016llx\n",
                gu, dt, (long long) N, E, K, (double) n / E, gu_ms, fl_gu / gu_ms / 1e9, d_ms, fl_d / d_ms / 1e9, hh);
    return 0;
}
