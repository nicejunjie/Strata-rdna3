// [claude] decode-window QSA selection cost at a long context: block scores and top-k exactly as verify.cpp's
// idx_select launches them (no active block count, capacity --max-context), timed separately.
// Usage: qsa_dec_bench [ctx=262144] [rows=3] [reps=50] [capacity=262144]
#include "strata/kernels/qsa.hpp"
#include "strata/kernels/qsa_select.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace k = strata::kernels;

static void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", w, cudaGetErrorString(e)); std::exit(2); }
}

int main(int argc, char** argv) {
    const int64_t ctx = argc > 1 ? std::atoll(argv[1]) : 262144;
    const int64_t nq = argc > 2 ? std::atoll(argv[2]) : 3;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 50;
    const int64_t capacity = argc > 4 ? std::atoll(argv[4]) : 262144;
    const k::QsaShapes s = k::qsa_real_shapes();
    const int64_t max_blocks = capacity / 4 + 2, cap = k::qsa_selection_width(k::kTopkMaxCells, s);
    std::mt19937 rng(7);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> dir(128), pooled((size_t) (max_blocks * 128)), dead(128), q((size_t) (nq * 512));
    for (auto& x : dir) x = nd(rng);
    for (int64_t b = 0; b < max_blocks; ++b) {
        const float a = nd(rng);
        for (int d = 0; d < 128; ++d) pooled[(size_t) (b * 128 + d)] = 0.5f * a * dir[d] + nd(rng);
    }
    for (auto& x : dead) x = nd(rng);
    for (int64_t i = 0; i < nq; ++i)
        for (int d = 0; d < 512; ++d) q[(size_t) (i * 512 + d)] = 0.2f * dir[d % 128] + 0.1f * nd(rng);
    std::vector<int32_t> steps((size_t) (nq * k::kStepCount));
    for (int64_t i = 0; i < nq; ++i) {
        int32_t* st = steps.data() + i * k::kStepCount;
        const int64_t pos = ctx - nq + i;
        st[k::kStepPos] = (int32_t) pos;
        st[k::kStepNKv] = (int32_t) (pos + 1);
        st[k::kStepNBid] = (int32_t) ((pos + 1) / s.idx_block);
        st[k::kStepWidth] = (int32_t) k::qsa_selection_width(pos + 1, s);
    }
    float *d_pooled, *d_dead, *d_q, *sc;
    int32_t *d_steps, *ids;
    ck(cudaMalloc(&d_pooled, pooled.size() * 4), "m"); ck(cudaMemcpy(d_pooled, pooled.data(), pooled.size() * 4, cudaMemcpyHostToDevice), "u");
    ck(cudaMalloc(&d_dead, 512), "m"); ck(cudaMemcpy(d_dead, dead.data(), 512, cudaMemcpyHostToDevice), "u");
    ck(cudaMalloc(&d_q, q.size() * 4), "m"); ck(cudaMemcpy(d_q, q.data(), q.size() * 4, cudaMemcpyHostToDevice), "u");
    ck(cudaMalloc(&d_steps, steps.size() * 4), "m"); ck(cudaMemcpy(d_steps, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice), "u");
    ck(cudaMalloc(&sc, (size_t) (nq * max_blocks) * 4), "m");
    ck(cudaMalloc(&ids, (size_t) (nq * cap) * 4), "m");
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    auto scores = [&] { k::qsa_block_scores(d_pooled, d_dead, d_q, d_steps, nq, max_blocks, s, sc, st); };
    auto topk = [&] { k::qsa_block_topk(sc, d_steps, nq, max_blocks, cap, s, ids, st); };
    for (int w = 0; w < 3; ++w) { scores(); topk(); }
    ck(cudaStreamSynchronize(st), "sync");
    cudaEvent_t a0, a1, a2;
    cudaEventCreate(&a0); cudaEventCreate(&a1); cudaEventCreate(&a2);
    float ms_s = 0, ms_t = 0;
    for (int r = 0; r < reps; ++r) {
        cudaEventRecord(a0, st); scores(); cudaEventRecord(a1, st); topk(); cudaEventRecord(a2, st);
        cudaEventSynchronize(a2);
        float x = 0, y = 0;
        cudaEventElapsedTime(&x, a0, a1); cudaEventElapsedTime(&y, a1, a2);
        ms_s += x; ms_t += y;
    }
    ms_s /= reps; ms_t /= reps;
    const double gb = (double) (ctx / 4 + 1) * 512.0 / 1e9;
    std::vector<int32_t> h((size_t) (nq * cap));
    ck(cudaMemcpy(h.data(), ids, h.size() * 4, cudaMemcpyDeviceToHost), "d");
    unsigned long long x = 1469598103934665603ull;
    for (auto v : h) x = (x ^ (uint32_t) v) * 1099511628211ull;
    std::printf("ctx %lld rows %lld: scores %.1f us (%.0f GB/s of keys), topk %.1f us, ids hash %016llx\n", (long long) ctx,
                (long long) nq, 1e3 * ms_s, gb / (ms_s / 1e3), 1e3 * ms_t, x);
    return 0;
}
