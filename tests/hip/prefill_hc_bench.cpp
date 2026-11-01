// tests/hip/prefill_hc_bench.cpp - [claude] the prompt path's hyper-connection kernels one by one at the production
// shapes (N 2560, HC 4, LR 320, STRATA_PF_HCDOWN=1 + STRATA_HC_UPMIX=1 + the fused write-back), with the bytes and FLOPs
// each moves, so each can be placed against the 7900 XTX roofs (~960 GB/s, ~123 TFLOPS FP16/BF16 WMMA).  One half-layer:
//   write   gr_write_norm_rs   R += bo * 2 sigmoid(inj / 4); the next half's rs and BF16 image xn16 (padded stride)
//   down    strata_pf_hcdown   lo = xn16 Wd^T (320), inj = xn16 Wi^T (4), BF16 WMMA, FP32 out
//   silu    gr_silu            lo16 = bf16(silu(lo / 4))
//   upmix   gr_upmix           mixed = sum_c sigmoid(lo16 Wu^T)[c] * xn(R) / 4 (+ BF16 / FP16 images)
//   prefill_hc_bench [T=16384] [reps=20]
#include "strata/prefill/kernels.hpp"
#include "../../src/prefill/wmma_gemm.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <algorithm>
#include <random>
#include <vector>

namespace {
constexpr int N = 2560, HC = 4, D = N * HC, LR = 320, PAD = 64;
uint16_t bf(float f) { uint32_t u; std::memcpy(&u, &f, 4); return (uint16_t) ((u + 0x7fffu + ((u >> 16) & 1u)) >> 16); }
}  // namespace

int main(int argc, char** argv) {
    const int64_t T = argc > 1 ? std::atoll(argv[1]) : 16384;
    const int reps = argc > 2 ? std::atoi(argv[2]) : 20;
    const int64_t ldx = D + PAD;
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto upf = [&](size_t n, float sc, float off) { std::vector<float> v(n); for (auto& x : v) x = off + sc * nd(rng); float* d; cudaMalloc(&d, n * 4); cudaMemcpy(d, v.data(), n * 4, cudaMemcpyHostToDevice); return d; };
    auto upb = [&](size_t n, float sc) { std::vector<uint16_t> v(n); for (auto& x : v) x = bf(sc * nd(rng)); uint16_t* d; cudaMalloc(&d, n * 2); cudaMemcpy(d, v.data(), n * 2, cudaMemcpyHostToDevice); return d; };
    float *R = upf((size_t) T * D, 1.f, 0.f), *bo = upf((size_t) T * N, 1.f, 0.f), *inj = upf((size_t) T * HC, 1.f, 0.f);
    float *wn = upf(D, 0.1f, 0.5f), *rs, *lo, *injo, *mixed;
    uint16_t *xn16, *lo16, *mixed16, *mixedh;
    uint16_t *Wd = upb((size_t) LR * D, 0.02f), *Wi = upb((size_t) HC * D, 0.02f), *Wu = upb((size_t) D * LR, 0.05f);
    cudaMalloc(&rs, (size_t) T * HC * 4); cudaMalloc(&lo, (size_t) T * LR * 4); cudaMalloc(&injo, (size_t) T * HC * 4);
    cudaMalloc(&mixed, (size_t) T * N * 4); cudaMalloc(&xn16, (size_t) T * ldx * 2); cudaMalloc(&lo16, (size_t) T * LR * 2);
    cudaMalloc(&mixed16, (size_t) T * N * 2); cudaMalloc(&mixedh, (size_t) T * N * 2);
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaEvent_t e[5];
    for (auto& x : e) cudaEventCreate(&x);
    double t[4] = {0, 0, 0, 0}, best[4] = {1e9, 1e9, 1e9, 1e9};
    bool upok = true;
    for (int r = 0; r < reps + 2; ++r) {
        cudaEventRecord(e[0], s);
        strata::prefill::gr_write_norm_rs(R, bo, inj, HC, wn, 1e-6f, rs, xn16, T, s, nullptr, ldx, false);
        cudaEventRecord(e[1], s);
        if (!strata_pf_hcdown_bf16(xn16, ldx, Wd, Wi, LR, HC, lo, injo, T, D, s)) { std::printf("hcdown refused\n"); return 1; }
        cudaEventRecord(e[2], s);
        strata::prefill::gr_silu(lo, lo16, T, s);
        cudaEventRecord(e[3], s);
        upok = strata::prefill::gr_upmix(lo16, Wu, R, rs, wn, mixed, mixed16, mixedh, T, s);
        cudaEventRecord(e[4], s);
        cudaEventSynchronize(e[4]);
        if (r == 0) {   // the outputs' checksum, to compare kernel variants bit for bit (STRATA_HC_UPMIX2=0 / 1)
            std::vector<float> m((size_t) T * N);
            std::vector<uint16_t> m16((size_t) T * N), mh((size_t) T * N);
            cudaMemcpy(m.data(), mixed, m.size() * 4, cudaMemcpyDeviceToHost);
            cudaMemcpy(m16.data(), mixed16, m16.size() * 2, cudaMemcpyDeviceToHost);
            cudaMemcpy(mh.data(), mixedh, mh.size() * 2, cudaMemcpyDeviceToHost);
            auto fnv = [](const void* p, size_t n) { const uint8_t* b = (const uint8_t*) p; uint64_t h = 1469598103934665603ull; for (size_t i = 0; i < n; ++i) { h ^= b[i]; h *= 1099511628211ull; } return h; };
            std::printf("checksum mixed %016llx mixed16 %016llx mixed_h %016llx\n", (unsigned long long) fnv(m.data(), m.size() * 4),
                        (unsigned long long) fnv(m16.data(), m16.size() * 2), (unsigned long long) fnv(mh.data(), mh.size() * 2));
        }
        if (r < 2) continue;
        for (int k = 0; k < 4; ++k) {
            float ms = 0;
            cudaEventElapsedTime(&ms, e[k], e[k + 1]);
            t[k] += ms / reps;
            best[k] = std::min(best[k], (double) ms);
        }
    }
    const double tk = (double) T;
    const double bytes[4] = {tk * D * 4 * 2 + tk * N * 4 + tk * D * 2,            // R r/w, bo, xn16
                             tk * D * 2 + (double) (LR + HC) * D * 2 + tk * (LR + HC) * 4,
                             tk * LR * 4 + tk * LR * 2,
                             tk * LR * 2 + (double) D * LR * 2 + tk * D * 4 + tk * N * (4 + 2 + 2)};   // lo16, Wu, R, mixed x3
    const double flops[4] = {0, 2.0 * tk * D * (LR + HC), 0, 2.0 * tk * LR * D};
    const char* nm[4] = {"write+norm", "down GEMM", "silu", "upmix"};
    double tot = 0;
    for (int k = 0; k < 4; ++k) {
        tot += t[k];
        std::printf("%-11s %7.3f ms (best %7.3f)  %7.1f GB/s  %6.1f TFLOPS\n", nm[k], t[k], best[k], bytes[k] / t[k] / 1e6,
                    flops[k] / t[k] / 1e9);
    }
    std::printf("T %lld: a half-layer %.3f ms; x96 half-layers %.1f ms%s\n", (long long) T, tot, 96 * tot, upok ? "" : " (upmix refused)");
    return 0;
}
