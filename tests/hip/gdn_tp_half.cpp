// [claude] TP prefill (docs/STRATA_TP_PREFILL_DESIGN_20261101.txt): the GDN prompt kernels on the two head halves
// (gdn_conv_tp / gdn_recurrence_tp, ranks 0 and 1, full-layout buffers) against the one-card launches (gdn_conv /
// gdn_recurrence) on the production variants - bit for bit: h, the conv history, y16 and the state.  Also the
// recurrence run in two token ranges [0, a) + [a, T) (a on the super-block grid), as the TP read overlaps its output
// exchange.  Usage: hip_gdn_tp_half [T ...] (default 4096 2112 8192).
#include "strata/prefill/kernels.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <cmath>
#include <random>
#include <vector>

namespace {
constexpr int S = 128, HV = 48, C = 10240, ROW = HV * S, CONV = 3;
int64_t LD = ROW;   // y16's row stride (the engine pads it: 6208)
void ck(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", where, cudaGetErrorString(e)); std::exit(2); }
}
template <class X> X* upload(const std::vector<X>& v) {
    X* p = nullptr;
    ck(cudaMalloc(&p, v.size() * sizeof(X)), "malloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(X), cudaMemcpyHostToDevice), "upload");
    return p;
}
template <class X> std::vector<X> download(const X* p, size_t n) {
    std::vector<X> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(X), cudaMemcpyDeviceToHost), "download");
    return v;
}
template <class X> size_t diff(const std::vector<X>& a, const std::vector<X>& b) {
    size_t d = 0;
    for (size_t i = 0; i < a.size(); ++i) d += std::memcmp(&a[i], &b[i], sizeof(X)) != 0;
    return d;
}
}  // namespace

int main(int argc, char** argv) {
    // the production prompt variants (read once by the kernels' static switches: set before the first launch)
    setenv("STRATA_GDN_CONVL2", "1", 1);
    setenv("STRATA_GDN_HEAD", "1", 1);
    setenv("STRATA_GDN_CHUNKED", "1", 1);
    setenv("STRATA_GDN_CHUNK_GDV", "0", 1);
    setenv("STRATA_GDN_NOY", "1", 1);
    std::vector<int64_t> Ts;
    for (int i = 1; i < argc; ++i) { if (!std::strcmp(argv[i], "--pad")) LD = ROW + 64; else Ts.push_back(std::atoll(argv[i])); }
    if (Ts.empty()) Ts = {4096, 2112, 8192};
    int fails = 0;
    for (int64_t T : Ts) {
        std::mt19937 rng(7 + (unsigned) T);
        std::normal_distribution<float> nd(0, 1);
        std::uniform_real_distribution<float> ud(0, 1);
        std::vector<float> qkv((size_t) T * C), w((size_t) C * 4), hist((size_t) C * CONV), gate((size_t) T * HV),
            beta((size_t) T * HV), z((size_t) T * ROW), gamma(S), state((size_t) S * ROW);
        for (auto& x : qkv) x = nd(rng);
        const char* load = std::getenv("TPH_LOAD");
        for (auto& x : w) x = 0.5f * nd(rng);
        for (auto& x : hist) x = nd(rng);
        for (auto& x : gate) x = -0.05f - 0.3f * ud(rng);
        for (auto& x : beta) x = ud(rng);
        for (auto& x : z) x = nd(rng);
        for (auto& x : gamma) x = 1.0f + 0.1f * nd(rng);
        for (auto& x : state) x = 0.05f * nd(rng);
        auto rd = [&](const char* nm, std::vector<float>& v) {
            std::FILE* f = std::fopen((std::string(load) + "/" + nm).c_str(), "rb");
            if (!f || std::fread(v.data(), 4, v.size(), f) != v.size()) { std::printf("load %s failed\n", nm); std::exit(2); }
            std::fclose(f);
        };
        std::vector<float> hload;
        if (load) { hload.resize((size_t) T * C); rd("hbuf", hload); rd("gate", gate); rd("beta", beta); rd("z", z); rd("state", state); rd("gamma", gamma); }
        float *d_qkv = upload(qkv), *d_w = upload(w), *d_gate = upload(gate), *d_beta = upload(beta), *d_z = upload(z),
              *d_gamma = upload(gamma);
        // arm A: the one-card launches; arm B: rank 0 then rank 1; arm C: rank 0/1, the recurrence in two token ranges
        float *hA = nullptr, *hB = nullptr, *yA = nullptr, *yB = nullptr;
        uint16_t *y16A = nullptr, *y16B = nullptr, *y16C = nullptr;
        ck(cudaMalloc(&hA, (size_t) T * C * 4), "h");
        ck(cudaMalloc(&hB, (size_t) T * C * 4), "h");
        ck(cudaMalloc(&yA, (size_t) T * ROW * 4), "y");
        ck(cudaMalloc(&yB, (size_t) T * ROW * 4), "y");
        ck(cudaMalloc(&y16A, (size_t) T * LD * 2), "y16");
        ck(cudaMalloc(&y16B, (size_t) T * LD * 2), "y16");
        ck(cudaMalloc(&y16C, (size_t) T * LD * 2), "y16");
        cudaMemset(hB, 0, (size_t) T * C * 4);
        cudaMemset(y16A, 0, (size_t) T * LD * 2); cudaMemset(y16B, 0, (size_t) T * LD * 2);
        cudaMemset(y16C, 0, (size_t) T * LD * 2);
        float *histA = upload(hist), *histB = upload(hist), *stA = upload(state), *stB = upload(state), *stC = upload(state);
        cudaStream_t s;
        ck(cudaStreamCreate(&s), "stream");
        strata::prefill::gdn_conv(histA, d_qkv, d_w, hA, T, 1e-6f, s);
        bool ok = strata::prefill::gdn_conv_tp(histB, d_qkv, d_w, hB, T, 1e-6f, 0, s) &&
                  strata::prefill::gdn_conv_tp(histB, d_qkv, d_w, hB, T, 1e-6f, 1, s);
        if (!ok) { std::printf("T %lld: gdn_conv_tp refused\n", (long long) T); return 1; }
        if (load) ck(cudaMemcpy(hA, hload.data(), hload.size() * 4, cudaMemcpyHostToDevice), "hload");
        strata::prefill::gdn_recurrence(stA, hA, d_gate, d_beta, d_z, d_gamma, 1e-6f, yA, y16A, T, s, LD);
        ok = strata::prefill::gdn_recurrence_tp(stB, hA, d_gate, d_beta, d_z, d_gamma, 1e-6f, yB, y16B, T, s, LD, 0) &&
             strata::prefill::gdn_recurrence_tp(stB, hA, d_gate, d_beta, d_z, d_gamma, 1e-6f, yB, y16B, T, s, LD, 1);
        if (!ok) { std::printf("T %lld: gdn_recurrence_tp refused\n", (long long) T); return 1; }
        const int64_t a = (T / 2) / 32 * 32;   // the engine's split (tp_split)
        for (int r = 0; r < 2; ++r) {
            ok = strata::prefill::gdn_recurrence_tp(stC, hA, d_gate, d_beta, d_z, d_gamma, 1e-6f, yB, y16C, a, s, LD, r) &&
                 strata::prefill::gdn_recurrence_tp(stC, hA + a * C, d_gate + a * HV, d_beta + a * HV, d_z + a * ROW,
                                                    d_gamma, 1e-6f, yB + a * ROW, y16C + a * LD, T - a, s, LD, r);
            if (!ok) { std::printf("T %lld: gdn_recurrence_tp (two ranges) refused\n", (long long) T); return 1; }
        }
        ck(cudaStreamSynchronize(s), "sync");
        const size_t dh = diff(download(hA, (size_t) T * C), download(hB, (size_t) T * C));
        const size_t dhist = diff(download(histA, hist.size()), download(histB, hist.size()));
        const size_t dy = diff(download(y16A, (size_t) T * LD), download(y16B, (size_t) T * LD));
        const size_t ds = diff(download(stA, state.size()), download(stB, state.size()));
        const size_t dyc = diff(download(y16A, (size_t) T * LD), download(y16C, (size_t) T * LD));
        const size_t dsc = diff(download(stA, state.size()), download(stC, state.size()));
        if (load) {
            auto ya = download(y16A, (size_t) T * LD), yc = download(y16C, (size_t) T * LD);
            auto sa = download(stA, state.size());
            size_t na = 0, nc = 0, ns = 0;
            for (size_t i = 0; i < ya.size(); ++i) { na += (ya[i] & 0x7c00) == 0x7c00; nc += (yc[i] & 0x7c00) == 0x7c00; }
            for (float x : sa) ns += !std::isfinite(x);
            std::printf("loaded data: one-card y16 non-finite %zu, two-range %zu; one-card state non-finite %zu\n", na, nc, ns);
        }
        const bool pass = dh == 0 && dhist == 0 && dy == 0 && ds == 0 && dyc == 0 && dsc == 0;
        fails += !pass;
        std::printf("T %5lld: h %zu, history %zu, y16 %zu, state %zu differ; two token ranges (split %lld): y16 %zu, state %zu -> %s\n",
                    (long long) T, dh, dhist, dy, ds, (long long) a, dyc, dsc, pass ? "BITWISE" : "FAIL");
        for (void* p : {(void*) d_qkv, (void*) d_w, (void*) d_gate, (void*) d_beta, (void*) d_z, (void*) d_gamma, (void*) hA,
                        (void*) hB, (void*) yA, (void*) yB, (void*) y16A, (void*) y16B, (void*) y16C, (void*) histA,
                        (void*) histB, (void*) stA, (void*) stB, (void*) stC})
            cudaFree(p);
        cudaStreamDestroy(s);
    }
    return fails ? 1 : 0;
}
