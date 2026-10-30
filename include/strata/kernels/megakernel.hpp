// include/strata/kernels/megakernel.hpp - the per-layer persistent kernel (docs/STRATA_MEGAKERNEL_DESIGN_20261022.txt).
//
// Prototype 1: one GDN layer's dense chain of the verify window - the q8_1 quantization of the layer input, the qkv
// and z projections (Q8_0), the conv + L2 norm and the a/b projections, the recurrence with its output norm and q8_1
// image, the out projection (Q8_0) - as ONE persistent kernel whose blocks pull work items phase by phase and wait on
// per-phase counters, instead of six graph nodes.  Every phase keeps the arithmetic of the kernel it replaces
// (verify_kernels.cu, native_mmvq.cu), element for element: mk_gdn_parity checks it bitwise.
#pragma once

#include <cstdint>

namespace strata::kernels {

/// [claude] a projection read through a row / block-column map out of a full matrix (WeightRef::map_*); pitch 0: plain
struct MkMat { int pitch = 0, nr = 0, cw = 0, cgap = 0, cadd = 0; int rlo[8] = {}, radd[8] = {}; };
/// [claude] y = W x for a mapped Q8_0 matrix (the fallback where the persistent kernel does not run): n_in / n_out logical
void q80_mmvq_mapped(const void* w, const MkMat& m, const void* xq, float* y, int n_in, int n_out, int ncols, void* stream);
struct MkGdnArgs {
    // the layer input and its q8_1 image (T columns)
    const float* x = nullptr;        ///< (T, n_embd)
    void* xq = nullptr;              ///< T columns of q8_1 blocks of n_embd (scratch, written here)
    // the projections (Q8_0 GGUF blocks, row-major)
    const void* w_qkv = nullptr;     ///< (conv_channels, n_embd)
    const void* w_z = nullptr;       ///< (value_dim, n_embd)
    const void* w_out = nullptr;     ///< (n_embd, value_dim)
    float* qkv = nullptr;            ///< (T, conv_channels)
    float* z = nullptr;              ///< (T, value_dim)
    // the conv + a/b
    float* hist = nullptr;           ///< (conv_channels, d_conv - 1): read only (no commit in a window)
    const float* w_conv = nullptr;   ///< (conv_channels, d_conv)
    float* h = nullptr;              ///< (T, conv_channels)
    const uint16_t* w_alpha = nullptr;   ///< bf16 (h_v, n_embd)
    const uint16_t* w_beta = nullptr;
    const float* dt = nullptr;       ///< (h_v)
    const float* ssm_a = nullptr;    ///< (h_v)
    float* gate = nullptr;           ///< (T, h_v)
    float* beta = nullptr;           ///< (T, h_v)
    // the recurrence
    float* state = nullptr;          ///< (S, h_v, S): read only (the commit replays it)
    const float* gamma = nullptr;    ///< (S)
    float* y = nullptr;              ///< (T, value_dim)
    void* yq = nullptr;              ///< T columns of q8_1 blocks of value_dim (the out projection's input)
    float* bo = nullptr;             ///< (T, n_embd)
    int n_embd = 0, conv_channels = 0, value_dim = 0, h_k = 0, h_v = 0;
    MkMat m_qkv, m_z, m_out;         ///< [claude] the TP alias maps (pitch 0: plain)
    float eps = 1e-6f;
    unsigned* counters = nullptr;    ///< device, >= 16 uints, zeroed by mk_gdn_dense before the launch
    unsigned* stall = nullptr;       ///< optional mapped host word: a phase wait that timed out writes its phase + 1 (sticky)
    int phases = 5;                  ///< diagnostics: run only the first `phases` phases
    int chunk = 0;                   ///< diagnostics: items per scheduler grab in every phase (0 = the per-phase defaults)
    int grid = 0;                    ///< diagnostics: force the grid (0 = the co-resident count)
    int diag = 0;                    ///< diagnostics bits: 1 dynamic chunks of 1, 2 skip the quantize body, 4 barrier between GEMV items, 8 qkv rows only, 16 x_stride 0, 32 no quantize phase (STRATA_QFUSE), 128 no all-wave fence, 256 the dynamic scheduler
};

/// Device scratch for the counters (16 uints); the caller keeps it alive for every launch.
unsigned* mk_counters_alloc();
/// The persistent grid the kernel uses on this device (every block co-resident), for information.
int mk_gdn_grid(int n_tok);
/// The six-node chain as one launch (plus a memset of the counters) on `stream`.  n_tok 1..8.
void mk_gdn_dense(const MkGdnArgs& a, int n_tok, void* stream);

}  // namespace strata::kernels
