// src/core/tp.cpp - tensor-parallel decode: the half geometry, the exchange pages, the sliced weights (tp.hpp).
#include "strata/core/tp.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/core/on_device.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/megakernel.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"

#include <cuda_runtime.h>
#if defined(STRATA_USE_HIP)
#include <hip/hip_runtime.h>
#ifndef cudaMemcpy2D
#define cudaMemcpy2D hipMemcpy2D   // (the compat header has no 2-D copy)
#endif
#endif

#include <cstdio>
#include <cstring>
#include <exception>
#include <algorithm>
#include <set>
#include <thread>
#include <chrono>
#include <vector>
#include <cstdlib>

namespace strata::core {

int tp_shexp_mode() {
    static const int mode = [] {
        const char* v = std::getenv("STRATA_TP_SHEXP");
        if (v == nullptr) return 1;
        const std::string s(v);
        return s == "split" ? 0 : s == "card0" || s == "0" ? 2 : 1;
    }();
    return mode;
}

int tp_kheads0(const ModelGeometry& g) {
    static const int k = [] { const char* v = std::getenv("STRATA_TP_KHEADS"); return v ? std::atoi(v) : -1; }();
    if (k > 0 && k < g.ssm_k_heads) return k;
    return (int) (g.ssm_k_heads / 2);
}
namespace {
// card `rank`'s GDN k-heads [kb, ke)
void k_range(const ModelGeometry& g, int rank, int& kb, int& ke) {
    const int k0 = tp_kheads0(g);
    kb = rank == 0 ? 0 : k0;
    ke = rank == 0 ? k0 : (int) g.ssm_k_heads;
}
}  // namespace

bool tp_geometry(const ModelGeometry& g, int rank, int nranks, ModelGeometry& out, std::string& err) {
    if (nranks != 2 || rank < 0 || rank >= nranks) { err = "tp: two cards"; return false; }
    if (g.ssm_v_heads % g.ssm_k_heads || g.n_head % nranks || g.n_head_kv % nranks ||
        (g.n_head / g.n_head_kv) * (g.n_head_kv / nranks) != g.n_head / nranks) {
        err = "tp: the heads do not divide by the cards (GDN " + std::to_string(g.ssm_k_heads) + "/" +
              std::to_string(g.ssm_v_heads) + ", QSA " + std::to_string(g.n_head) + "/" + std::to_string(g.n_head_kv) + ")";
        return false;
    }
    if (g.ssm_conv_channels != 2 * g.ssm_k_heads * g.ssm_state_size + g.ssm_v_heads * g.ssm_state_size ||
        g.ssm_value_dim != g.ssm_v_heads * g.ssm_state_size) {
        err = "tp: the GDN channel layout is not q|k|v by head";
        return false;
    }
    out = g;
    int kb = 0, ke = 0;
    k_range(g, rank, kb, ke);
    out.ssm_k_heads = ke - kb;
    out.ssm_v_heads = (ke - kb) * (g.ssm_v_heads / g.ssm_k_heads);
    out.ssm_conv_channels = 2 * out.ssm_k_heads * g.ssm_state_size + out.ssm_v_heads * g.ssm_state_size;
    out.ssm_value_dim = out.ssm_v_heads * g.ssm_state_size;
    out.n_head = g.n_head / nranks;
    out.n_head_kv = g.n_head_kv / nranks;
    return true;
}

// ---- the exchange pages
namespace {
bool pinned_coherent(void** h, size_t bytes) {
#if defined(STRATA_USE_HIP)
    // coherent (fine-grained) host memory: a store from one card is seen by a kernel running on the other without
    // either stream syncing - the design's M0 finding (peer stores into VRAM are not)
    return hipHostMalloc(h, bytes, hipHostMallocMapped | hipHostMallocCoherent | hipHostMallocPortable) == hipSuccess;
#else
    return cudaHostAlloc(h, bytes, cudaHostAllocMapped | cudaHostAllocPortable) == cudaSuccess;
#endif
}
}  // namespace

TpPages::~TpPages() {
    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 2; ++p) if (inbox_h[c][p]) cudaFreeHost(inbox_h[c][p]);
        if (flag_h[c]) cudaFreeHost(flag_h[c]);
        for (int i = 0; i < 3; ++i)
            if (vram_owned[c][i] != nullptr) {
                const OnDevice on(dev[c]);
                cudaFree(vram_owned[c][i]);
            }
    }
}

bool tp_flag_store(uint32_t* flag_h, uint32_t* flag_d, int dev, uint32_t value, void* stream) {
    if (flag_h != nullptr) {
        *(volatile uint32_t*) flag_h = value;
        return true;
    }
    if (flag_d == nullptr || dev < 0) return false;
#if defined(STRATA_USE_HIP)
    const OnDevice on(dev);
    if (stream != nullptr) return hipMemsetD32Async((hipDeviceptr_t) flag_d, value, 1, (hipStream_t) stream) == hipSuccess;
    return hipMemsetD32((hipDeviceptr_t) flag_d, value, 1) == hipSuccess;
#else
    (void) value; (void) stream;
    return false;
#endif
}

bool TpPages::alloc(int dev0, int dev1, int64_t n, std::string& err) {
    dev[0] = dev0;
    dev[1] = dev1;
    floats = n;
    const size_t bytes = (size_t) n * sizeof(float);
#if defined(STRATA_USE_HIP)
    // [claude] STRATA_TP_INBOX=vram: the pages in the receiving card's VRAM (see tp.hpp).  Needs peer access both ways;
    // any failure falls back to the pinned pages below with a note.
    static const bool want_vram = [] { const char* v = std::getenv("STRATA_TP_INBOX"); return v != nullptr && std::strcmp(v, "vram") == 0; }();
    if (want_vram) {
        bool ok = true;
        int can01 = 0, can10 = 0;
        cudaDeviceCanAccessPeer(&can01, dev0, dev1);
        cudaDeviceCanAccessPeer(&can10, dev1, dev0);
        ok = can01 && can10;
        for (int c = 0; ok && c < 2; ++c) {
            const OnDevice on(dev[c]);
            const cudaError_t e = cudaDeviceEnablePeerAccess(dev[1 - c], 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) ok = false;
            (void) cudaGetLastError();
        }
        for (int c = 0; ok && c < 2; ++c) {
            const OnDevice on(dev[c]);
            for (int p = 0; ok && p < 2; ++p) {
                void* q = nullptr;
                if (hipExtMallocWithFlags(&q, bytes, hipDeviceMallocUncached) != hipSuccess || hipMemset(q, 0, bytes) != hipSuccess) { ok = false; break; }
                vram_owned[c][p] = q;
                for (int d = 0; d < 2; ++d) inbox_d[c][p][d] = (float*) q;
            }
            void* f = nullptr;
            if (ok && (hipExtMallocWithFlags(&f, 256, hipDeviceMallocUncached) != hipSuccess || hipMemset(f, 0, 256) != hipSuccess)) ok = false;
            if (ok) {
                vram_owned[c][2] = f;
                for (int d = 0; d < 2; ++d) flag_d[c][d] = (uint32_t*) f;
            }
            if (ok && hipDeviceSynchronize() != hipSuccess) ok = false;
        }
        if (ok) {
            vram = true;
            std::fprintf(stderr, "strata generate: tensor parallel: the exchange pages are in each card's VRAM (STRATA_TP_INBOX=vram)\n");
            return true;
        }
        (void) cudaGetLastError();
        for (int c = 0; c < 2; ++c)
            for (int i = 0; i < 3; ++i)
                if (vram_owned[c][i] != nullptr) {
                    const OnDevice on(dev[c]);
                    cudaFree(vram_owned[c][i]);
                    vram_owned[c][i] = nullptr;
                }
        std::memset(inbox_d, 0, sizeof(inbox_d));
        std::memset(flag_d, 0, sizeof(flag_d));
        std::fprintf(stderr, "strata generate: tensor parallel: STRATA_TP_INBOX=vram needs peer access between the two cards and "
                             "fine-grained VRAM; the pinned pages are used\n");
    }
#endif
    for (int c = 0; c < 2; ++c) {
        for (int p = 0; p < 2; ++p) {
            if (!pinned_coherent((void**) &inbox_h[c][p], bytes)) { err = "tp: the exchange inbox allocation failed"; return false; }
            std::memset(inbox_h[c][p], 0, bytes);
        }
        if (!pinned_coherent((void**) &flag_h[c], 256)) { err = "tp: the exchange flag allocation failed"; return false; }
        std::memset(flag_h[c], 0, 256);
    }
    for (int d = 0; d < 2; ++d) {
        const OnDevice on(dev[d]);
        for (int c = 0; c < 2; ++c) {
            for (int p = 0; p < 2; ++p)
                if (cudaHostGetDevicePointer((void**) &inbox_d[c][p][d], inbox_h[c][p], 0) != cudaSuccess) {
                    err = "tp: the exchange inbox has no device address on CUDA" + std::to_string(dev[d]);
                    return false;
                }
            if (cudaHostGetDevicePointer((void**) &flag_d[c][d], flag_h[c], 0) != cudaSuccess) {
                err = "tp: the exchange flag has no device address on CUDA" + std::to_string(dev[d]);
                return false;
            }
        }
    }
    return true;
}

// ---- the sliced weights
TpWeights::~TpWeights() {
    for (void* p : owned_) cudaFree(p);
}

void* TpWeights::take(uint64_t bytes, std::string& err) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
        (void) cudaGetLastError();
        err = "tp: " + std::to_string(bytes >> 20) + " MiB of VRAM for a sliced weight is not there";
        return nullptr;
    }
    owned_.push_back(p);
    bytes_ += bytes;
    return p;
}

namespace {
struct Range { uint64_t begin, count; };   // rows [begin, begin + count)
// THE GDN HEAD SPLIT.  A v-head H reads k-head H % h_k (gdn_step_norm_multi_kernel: qh = head % h_k), so card r, which
// holds k-heads [r * hk, (r + 1) * hk), must hold the v-heads whose H % h_k falls there: the h_v / h_k ranges
// j * h_k + r * hk + [0, hk), in that (local) order - local v-head h then reads local k-head h % hk, as the kernel does.
std::vector<Range> v_heads(const ModelGeometry& g, int rank, int nranks) {
    (void) nranks;
    int kb = 0, ke = 0;
    k_range(g, rank, kb, ke);
    const uint64_t HK = (uint64_t) g.ssm_k_heads;
    std::vector<Range> v;
    for (uint64_t j = 0; j < (uint64_t) g.ssm_v_heads / HK; ++j) v.push_back({j * HK + (uint64_t) kb, (uint64_t) (ke - kb)});
    return v;
}
std::vector<Range> scaled(const std::vector<Range>& r, uint64_t per, uint64_t base = 0) {
    std::vector<Range> o;
    for (const Range& x : r) o.push_back({base + x.begin * per, x.count * per});
    return o;
}
// the GDN qkv / conv channel order: q heads, k heads, v heads - card r's slices of each
std::vector<Range> qkv_rows(const ModelGeometry& g, int rank, int nranks, uint64_t per_head) {
    int kb = 0, ke = 0;
    k_range(g, rank, kb, ke);
    const uint64_t hk = (uint64_t) (ke - kb), kbh = (uint64_t) kb;
    const uint64_t k0 = (uint64_t) g.ssm_k_heads * per_head, v0 = 2 * (uint64_t) g.ssm_k_heads * per_head;
    std::vector<Range> r{{kbh * per_head, hk * per_head}, {k0 + kbh * per_head, hk * per_head}};
    for (const Range& x : scaled(v_heads(g, rank, nranks), per_head, v0)) r.push_back(x);
    return r;
}
}  // namespace

bool TpWeights::build(const WeightTable& full, const ModelGeometry& g, int rank, int nranks,
                      const std::vector<std::string>& shards, std::string& err) {
    if (!owned_.empty()) { err = "tp: weights already built"; return false; }
    table_ = full;   // the map of references: the small tensors keep their (device) pointers
    const uint64_t S = (uint64_t) g.ssm_state_size, HD = (uint64_t) g.head_dim;
    int kb0 = 0, ke0 = 0;
    k_range(g, rank, kb0, ke0);
    const uint64_t hk = (uint64_t) (ke0 - kb0), hv = hk * ((uint64_t) g.ssm_v_heads / (uint64_t) g.ssm_k_heads);
    const uint64_t nh = (uint64_t) g.n_head / nranks, nkv = (uint64_t) g.n_head_kv / nranks;
    // ---- the small tensors (pack arena, f32 / bf16 bits): row slices by head
    auto slice_rows = [&](const std::string& name, const std::vector<Range>& rows, int64_t want_ne1) -> bool {
        auto it = table_.table_.find(name);
        if (it == table_.table_.end()) { err = "tp: " + name + " is not in the table"; return false; }
        WeightRef& r = it->second;
        if (r.data == nullptr) { err = "tp: " + name + " has no bytes on this card (the arena must hold every layer)"; return false; }
        if (r.elements <= 0 || r.ne1 <= 0 || r.bytes % (uint64_t) r.elements != 0) { err = "tp: " + name + ": odd element size"; return false; }
        const uint64_t eb = r.bytes / (uint64_t) r.elements, row = (uint64_t) r.ne0 * eb;
        uint64_t total = 0;
        for (const Range& rg : rows) total += rg.count;
        if ((int64_t) total != want_ne1) { err = "tp: " + name + ": the slice does not add up"; return false; }
        void* dst = take(total * row, err);
        if (dst == nullptr) return false;
        uint64_t off = 0;
        for (const Range& rg : rows) {
            if (rg.begin + rg.count > (uint64_t) r.ne1) { err = "tp: " + name + ": slice past the end"; return false; }
            if (cudaMemcpy((uint8_t*) dst + off, (const uint8_t*) r.data + rg.begin * row, rg.count * row,
                           cudaMemcpyDeviceToDevice) != cudaSuccess) {
                err = "tp: " + name + ": the slice copy failed";
                return false;
            }
            off += rg.count * row;
        }
        r.data = dst;
        r.ne1 = want_ne1;
        r.elements = (int64_t) (total * (uint64_t) r.ne0);
        r.bytes = total * row;
        return true;
    };
    // a vector of per-head scalars: ne0 = heads (one row): slice along ne0
    auto slice_vec = [&](const std::string& name, const std::vector<Range>& rs) -> bool {
        auto it = table_.table_.find(name);
        if (it == table_.table_.end()) { err = "tp: " + name + " is not in the table"; return false; }
        WeightRef& r = it->second;
        if (r.data == nullptr) { err = "tp: " + name + " has no bytes on this card"; return false; }
        uint64_t count = 0;
        for (const Range& x : rs) count += x.count;
        if (r.elements <= 0 || r.bytes % (uint64_t) r.elements != 0) { err = "tp: " + name + ": odd vector"; return false; }
        const uint64_t eb = r.bytes / (uint64_t) r.elements;
        void* dst = take(count * eb, err);
        if (dst == nullptr) return false;
        uint64_t off = 0;
        for (const Range& x : rs) {
            if (x.begin + x.count > (uint64_t) r.elements) { err = "tp: " + name + ": slice past the end"; return false; }
            if (cudaMemcpy((uint8_t*) dst + off, (const uint8_t*) r.data + x.begin * eb, x.count * eb, cudaMemcpyDeviceToDevice) != cudaSuccess) {
                err = "tp: " + name + ": the slice copy failed";
                return false;
            }
            off += x.count * eb;
        }
        r.data = dst;
        r.ne0 = (int64_t) count;
        r.elements = (int64_t) count;
        r.bytes = count * eb;
        return true;
    };
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const std::string pfx = "blk." + std::to_string(l) + ".";
        if (!is_qsa_layer(g, l)) {
            // alpha / beta: bf16 [h_v][n_embd]; dt bias / ssm_a: [h_v]; conv1d: [channels][d_conv]
            const std::vector<Range> heads = v_heads(g, rank, nranks);
            if (!slice_rows(pfx + "ssm_alpha.weight", heads, (int64_t) hv) ||
                !slice_rows(pfx + "ssm_beta.weight", heads, (int64_t) hv) ||
                !slice_vec(pfx + "ssm_dt.bias", heads) || !slice_vec(pfx + "ssm_a", heads) ||
                !slice_rows(pfx + "ssm_conv1d.weight", qkv_rows(g, rank, nranks, S), (int64_t) (2 * hk * S + hv * S)))
                return false;
        }
        // (q_norm / k_norm / ssm_norm are per head-dim: shared; the indexer, router, hyper-connections: replicated)
    }
    // ---- the dense projections from the GGUF shards at half width
    // kind 0: GDN qkv rows; 1: rows [rank * n / nranks, ...); 2: an n_in slice [rank * n / nranks, ...); 3: the GDN
    // v-head rows (attn_gate); 4: the GDN v-head columns of n_in (ssm_out)
    struct Want { const char* suffix; int kind; };
    // (the shared expert too: gate / up by ff unit, down over the ff units - each card's partial is added in the
    // layer's second exchange)
    static const Want wants[] = {{".attn_qkv.weight", 0}, {".attn_gate.weight", 3}, {".ssm_out.weight", 4},
                                 {".attn_q.weight", 1}, {".attn_k.weight", 1}, {".attn_v.weight", 1}, {".attn_output.weight", 2},
                                 {".ffn_gate_shexp.weight", 1}, {".ffn_up_shexp.weight", 1}, {".ffn_down_shexp.weight", 2}};
    std::set<std::string> done;
    uint64_t max_in = 0;
    try {
        for (const auto& path : shards) {
            strata::GgufFile gguf(path);
            for (const auto& t : gguf.tensors()) {
                if (t.name.rfind("blk.", 0) != 0 || t.shape.size() != 2) continue;
                int kind = -1;
                for (const Want& w : wants) if (t.name.ends_with(w.suffix)) kind = w.kind;
                if (kind < 0) continue;
                const bool shexp = t.name.find("_shexp.") != std::string::npos;
                if (shexp && !tp_shexp_split()) {
                    // whole on one card: that card loads the layers its stage's table has no bytes for (kind 5)
                    auto it0 = table_.table_.find(t.name);
                    const bool mine = (tp_shexp_mode() == 1) == (rank == 1);
                    if (!mine || (it0 != table_.table_.end() && it0->second.native_data != nullptr)) continue;
                    kind = 5;
                }
                if (!strata::kernels::native_mmvq_supported(t.type)) { err = "tp: " + t.name + ": format not native"; return false; }
                if (!done.insert(t.name).second) { err = "tp: duplicate tensor " + t.name; return false; }
                auto it = table_.table_.find(t.name);
                if (it == table_.table_.end()) { err = "tp: " + t.name + " is not in the table"; return false; }
                WeightRef& r = it->second;
                const uint64_t n_in = t.shape[0], n_out = t.shape[1];
                if (n_in != (uint64_t) r.ne0 || n_out != (uint64_t) r.ne1) { err = "tp: " + t.name + ": shape differs from the pack"; return false; }
                int be = 0, bb = 0;
                if (!strata::block_geometry(t.type, be, bb) || n_in % (uint64_t) be) { err = "tp: " + t.name + ": block geometry"; return false; }
                const uint64_t row_bytes = n_in / (uint64_t) be * (uint64_t) bb;
                const uint8_t* src = gguf.tensor_data(t);
                uint64_t new_in = n_in, new_out = n_out;
                std::vector<Range> rows, cols;   // cols: n_in element ranges (kind 2 / 4)
                if (kind == 0) {
                    rows = qkv_rows(g, rank, nranks, S);
                    new_out = 2 * hk * S + hv * S;
                } else if (kind == 1) {
                    new_out = n_out / (uint64_t) nranks;
                    if (new_out * (uint64_t) nranks != n_out) { err = "tp: " + t.name + ": rows do not divide"; return false; }
                    rows = {{rank * new_out, new_out}};
                } else if (kind == 3) {
                    rows = scaled(v_heads(g, rank, nranks), S);
                    new_out = hv * S;
                } else if (kind == 5) {   // the whole matrix
                    rows = {{0, n_out}};
                } else if (kind == 2) {
                    new_in = n_in / (uint64_t) nranks;
                    if (new_in * (uint64_t) nranks != n_in) { err = "tp: " + t.name + ": the reduction does not divide"; return false; }
                    cols = {{rank * new_in, new_in}};
                } else {
                    cols = scaled(v_heads(g, rank, nranks), S);
                    new_in = hv * S;
                }
                for (const Range& c : cols)
                    if (c.begin % (uint64_t) be || c.count % (uint64_t) be) { err = "tp: " + t.name + ": a reduction slice cuts a block"; return false; }
                const uint64_t bytes = strata::kernels::native_mmvq_weight_bytes((int) t.type, (int) new_in, (int) new_out);
                // [claude] STRATA_TP_ALIAS (default on; 0 = copy every slice): a slice that is ONE contiguous range of
                // rows of a full matrix this card already holds natively (the DP prompt path keeps every layer's dense
                // weights on both cards) points into that matrix instead of a copy - the same bytes, no VRAM
                static const bool alias_env = [] { const char* v = std::getenv("STRATA_TP_ALIAS"); return v == nullptr || v[0] != '0'; }();
                if (alias_env && cols.empty() && rows.size() == 1 && r.native_data != nullptr && r.native_type == (int) t.type &&
                    strata::kernels::native_mmvq_weight_bytes((int) t.type, (int) n_in, (int) n_out) == n_out * row_bytes &&
                    bytes == rows[0].count * row_bytes && rows[0].begin + rows[0].count <= n_out &&
                    (rows[0].begin * row_bytes) % 256 == 0 && [&] {
                        // the full matrix must hold the file's bytes as they are (no repacked layout): its slice's first
                        // and last 64 KiB against the GGUF's
                        const uint64_t off = rows[0].begin * row_bytes, n = std::min<uint64_t>(bytes, 65536);
                        std::vector<uint8_t> a(n);
                        for (const uint64_t at : {(uint64_t) 0, bytes - n}) {
                            if (cudaMemcpy(a.data(), (const uint8_t*) r.native_data + off + at, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
                                (void) cudaGetLastError();
                                return false;
                            }
                            if (std::memcmp(a.data(), src + off + at, n) != 0) return false;
                        }
                        return true;
                    }()) {
                    r.native_data = (const uint8_t*) r.native_data + rows[0].begin * row_bytes;
                    r.ne0 = (int64_t) new_in;
                    r.ne1 = (int64_t) new_out;
                    r.elements = (int64_t) (new_in * new_out);
                    r.data = nullptr;
                    r.bytes = 0;
                    aliased_ += bytes;
                    if (new_in > max_in) max_in = new_in;
                    continue;
                }
                // [claude] STRATA_TP_ALIAS_MAP=1: a Q8_0 GDN projection (qkv / gate: up to 4 row ranges; ssm_out: up to 2
                // block-column ranges) reads the full matrix through a map (WeightRef::map_*, the verifier's Q8_0 GEMVs)
                static const bool alias_map = [] { const char* v = std::getenv("STRATA_TP_ALIAS_MAP"); return v != nullptr && v[0] == '1'; }();
                auto why_not = [&](const char* why) {   // once per kind: why a projection keeps its copy
                    static std::set<std::string> said;
                    if (alias_map && said.insert(std::string(why) + std::to_string(kind)).second)
                        std::fprintf(stderr, "strata generate: tensor parallel: %s keeps its copy (STRATA_TP_ALIAS_MAP: %s)\n", t.name.c_str(), why);
                    return false;
                };
                if (alias_env && alias_map && (kind == 0 || kind == 3 || kind == 4) && t.type == 8 &&
                    (r.native_data != nullptr || why_not("no full matrix on this card")) &&
                    (r.native_type == 8 || why_not("the full matrix is not Q8_0")) &&
                    ((rows.size() <= 8 && (rows.empty() != cols.empty()) && [&] {   // columns: equal ranges, equally spaced
                        for (size_t k = 1; k < cols.size(); ++k)
                            if (cols[k].count != cols[0].count || cols[k].begin - cols[k - 1].begin != cols[1].begin - cols[0].begin) return false;
                        const uint64_t cwb = cols.empty() ? 1 : cols[0].count / (uint64_t) be;   // the kernel shifts by log2
                        return cols.size() <= 1 || (cwb & (cwb - 1)) == 0;
                    }()) || why_not("ranges")) &&
                    (strata::kernels::native_mmvq_weight_bytes(8, (int) n_in, (int) n_out) == n_out * row_bytes || why_not("layout size")) && [&] {
                        const uint64_t full = n_out * row_bytes, n = std::min<uint64_t>(full, 65536);
                        std::vector<uint8_t> a(n);
                        for (const uint64_t at : {(uint64_t) 0, full - n}) {
                            if (cudaMemcpy(a.data(), (const uint8_t*) r.native_data + at, n, cudaMemcpyDeviceToHost) != cudaSuccess) {
                                (void) cudaGetLastError();
                                return false;
                            }
                            if (std::memcmp(a.data(), src + at, n) != 0) return why_not("the full matrix is not the file's bytes");
                        }
                        return true;
                    }()) {
                    r.map_pitch = (int) (n_in / (uint64_t) be);
                    r.map_nr = 1; r.map_rlo[0] = 0; r.map_radd[0] = 0;
                    r.map_cw = 0; r.map_cgap = 0; r.map_cadd = 0;
                    if (!rows.empty()) {
                        int lo = 0;
                        r.map_nr = (int) rows.size();
                        for (size_t k = 0; k < rows.size(); ++k) {
                            r.map_rlo[k] = lo; r.map_radd[k] = (int) rows[k].begin - lo; lo += (int) rows[k].count;
                        }
                    } else {
                        r.map_cadd = (int) (cols[0].begin / (uint64_t) be);
                        if (cols.size() > 1) {
                            r.map_cw = (int) (cols[0].count / (uint64_t) be);
                            r.map_cgap = (int) ((cols[1].begin - cols[0].begin - cols[0].count) / (uint64_t) be);
                        }
                    }
                    // [claude] STRATA_TP_ALIAS_MAP_CHECK (default on): the map against the slice itself - the same GEMV
                    // kernel through the map and on a dense copy of the slice built from the file, bit for bit
                    static const bool map_check = [] { const char* v = std::getenv("STRATA_TP_ALIAS_MAP_CHECK"); return v == nullptr || v[0] != '0'; }();
                    if (map_check) {
                        std::vector<uint8_t> h((size_t) bytes);
                        uint64_t off = 0;
                        if (!rows.empty()) {
                            for (const Range& x : rows) { std::memcpy(h.data() + off, src + x.begin * row_bytes, x.count * row_bytes); off += x.count * row_bytes; }
                        } else {
                            const uint64_t bb_ = (uint64_t) bb;
                            for (uint64_t o = 0; o < n_out; ++o)
                                for (const Range& c : cols) {
                                    const uint64_t cb = c.count / (uint64_t) be * bb_;
                                    std::memcpy(h.data() + off, src + o * row_bytes + c.begin / (uint64_t) be * bb_, cb);
                                    off += cb;
                                }
                        }
                        std::vector<float> xh((size_t) new_in);
                        uint32_t seed = 12345u + (uint32_t) done.size();
                        for (float& v : xh) { seed = seed * 1664525u + 1013904223u; v = (float) ((int32_t) (seed >> 8) % 2001 - 1000) / 1000.0f; }
                        void *dcopy = nullptr, *dx = nullptr, *dxq = nullptr, *dy = nullptr;
                        std::vector<float> y1((size_t) new_out), y2((size_t) new_out);
                        bool ok = off == bytes && cudaMalloc(&dcopy, bytes) == cudaSuccess && cudaMalloc(&dx, new_in * 4) == cudaSuccess &&
                                  cudaMalloc(&dxq, strata::kernels::native_q8_1_bytes((int) new_in, 1)) == cudaSuccess &&
                                  cudaMalloc(&dy, new_out * 4) == cudaSuccess &&
                                  cudaMemcpy(dcopy, h.data(), bytes, cudaMemcpyHostToDevice) == cudaSuccess &&
                                  cudaMemcpy(dx, xh.data(), new_in * 4, cudaMemcpyHostToDevice) == cudaSuccess;
                        cudaStream_t cks = nullptr;
                        ok = ok && cudaStreamCreateWithFlags(&cks, cudaStreamNonBlocking) == cudaSuccess;
                        if (ok) {
                            strata::kernels::native_quantize_q8_1((const float*) dx, dxq, (int) new_in, 1, cks);
                            strata::kernels::MkMat plain, m;
                            m.pitch = r.map_pitch; m.nr = r.map_nr; m.cw = r.map_cw; m.cgap = r.map_cgap; m.cadd = r.map_cadd;
                            for (int k = 0; k < 8; ++k) { m.rlo[k] = r.map_rlo[k]; m.radd[k] = r.map_radd[k]; }
                            strata::kernels::q80_mmvq_mapped(dcopy, plain, dxq, (float*) dy, (int) new_in, (int) new_out, 1, cks);
                            ok = cudaStreamSynchronize(cks) == cudaSuccess && cudaMemcpy(y1.data(), dy, new_out * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
                            strata::kernels::q80_mmvq_mapped(r.native_data, m, dxq, (float*) dy, (int) new_in, (int) new_out, 1, cks);
                            ok = ok && cudaStreamSynchronize(cks) == cudaSuccess && cudaMemcpy(y2.data(), dy, new_out * 4, cudaMemcpyDeviceToHost) == cudaSuccess;
                        }
                        if (cks) cudaStreamDestroy(cks);
                        for (void* p : {dcopy, dx, dxq, dy}) if (p) cudaFree(p);
                        (void) cudaGetLastError();
                        if (!ok || std::memcmp(y1.data(), y2.data(), y1.size() * 4) != 0) {
                            err = "tp: " + t.name + ": the mapped read differs from the slice (STRATA_TP_ALIAS_MAP=0 turns the map off)";
                            return false;
                        }
                    }
                    r.ne0 = (int64_t) new_in;
                    r.ne1 = (int64_t) new_out;
                    r.elements = (int64_t) (new_in * new_out);
                    r.data = nullptr;
                    r.bytes = 0;
                    aliased_ += bytes;
                    if (new_in > max_in) max_in = new_in;
                    continue;
                }
                void* dst = take(bytes, err);
                if (dst == nullptr) return false;
                cudaError_t st = cudaSuccess;
                if (!cols.empty()) {   // every row's blocks of the column ranges: strided copies from the mapped file
                    const uint64_t new_row = new_in / (uint64_t) be * (uint64_t) bb;
                    uint64_t doff = 0;
                    for (const Range& c : cols) {
                        const uint64_t w = c.count / (uint64_t) be * (uint64_t) bb, so = c.begin / (uint64_t) be * (uint64_t) bb;
                        st = cudaMemcpy2D((uint8_t*) dst + doff, new_row, src + so, row_bytes, w, n_out, cudaMemcpyHostToDevice);
                        if (st != cudaSuccess) break;
                        doff += w;
                    }
                } else {
                    uint64_t off = 0;
                    for (const Range& rg : rows) {
                        if (rg.begin + rg.count > n_out) { err = "tp: " + t.name + ": slice past the end"; return false; }
                        st = cudaMemcpy((uint8_t*) dst + off, src + rg.begin * row_bytes, rg.count * row_bytes, cudaMemcpyHostToDevice);
                        if (st != cudaSuccess) break;
                        off += rg.count * row_bytes;
                    }
                    if (off != bytes) { err = "tp: " + t.name + ": the slice bytes do not add up"; return false; }
                }
                if (st != cudaSuccess) { err = "tp: " + t.name + ": upload: " + cudaGetErrorString(st); return false; }
                r.native_data = dst;
                r.native_type = (int) t.type;
                r.ne0 = (int64_t) new_in;
                r.ne1 = (int64_t) new_out;
                r.elements = (int64_t) (new_in * new_out);
                r.data = nullptr;   // the canonical form (if any) is the full matrix: never read it by mistake
                r.bytes = 0;
                if (new_in > max_in) max_in = new_in;
            }
        }
    } catch (const std::exception& e) {
        err = std::string("tp: ") + e.what();
        return false;
    }
    if (done.size() < (size_t) (g.n_gdn_layers() * 3 + g.n_qsa_layers() * 4)) {
        err = "tp: the shards hold " + std::to_string(done.size()) + " dense projections, not every layer's";
        return false;
    }
    // the shared q8_1 scratch the table's references name (the window quantizes into its own buffers; the layer path's)
    void* scratch = take(strata::kernels::native_q8_1_bytes((int) max_in, 8), err);
    if (scratch == nullptr) return false;
    for (auto& [name, r] : table_.table_)
        if (r.native_data != nullptr) r.native_q8_1 = scratch;
    (void) nh; (void) nkv; (void) HD;
    return true;
}

// ---- the state hand-over (tp.hpp)
namespace {
// A strided copy between two devices through a pinned staging buffer (no GPU-to-GPU peer access needed): `height`
// rows of `width` bytes, pitches in bytes.
struct Stager {
    uint8_t* h = nullptr;
    size_t cap = 0;
    // the direct path: every piece an asynchronous device-to-device copy (on the destination card's stream 0; a peer copy
    // when the cards differ), waited for once at the end (finish).  Was: each piece through pinned host memory with two
    // synchronous copies - 94 ms a hand-over at 2K cells, 150 ms at 32K (2026-10-22)
    bool direct = false;
    bool used[64] = {};
    double bytes[2][2] = {};   // (STRATA_TP_TIMING) the direct copies' bytes, [source card][destination card] (cards 0 / 1)
    ~Stager() { if (h) cudaFreeHost(h); }
    bool ensure(size_t n) {
        if (n <= cap) return true;
        if (h) cudaFreeHost(h);
        h = nullptr;
        cap = 0;
        if (cudaHostAlloc((void**) &h, n, cudaHostAllocDefault) != cudaSuccess) { (void) cudaGetLastError(); return false; }
        cap = n;
        return true;
    }
    bool finish(std::string& err) {
        for (int d = 0; d < 64; ++d) {
            if (!used[d]) continue;
            const OnDevice on(d);
            if (cudaDeviceSynchronize() != cudaSuccess) { err = "tp: the hand-over copies failed"; return false; }
            used[d] = false;
        }
        return true;
    }
};
// The two cards can read each other's memory (the hand-over then copies device to device); STRATA_TP_MOVE_STAGED=1 keeps
// the host-staged copies.
bool peer_ready(int a, int b) {
    if (a == b) return true;
    static int ok[64][64] = {};   // 0 unknown, 1 yes, -1 no
    if (a < 0 || b < 0 || a >= 64 || b >= 64) return false;
    if (ok[a][b] == 0) {
        int can_ab = 0, can_ba = 0;
        cudaDeviceCanAccessPeer(&can_ab, a, b);
        cudaDeviceCanAccessPeer(&can_ba, b, a);
        bool good = can_ab && can_ba;
        for (int k = 0; good && k < 2; ++k) {
            const OnDevice on(k == 0 ? a : b);
            const cudaError_t e = cudaDeviceEnablePeerAccess(k == 0 ? b : a, 0);
            if (e != cudaSuccess && e != cudaErrorPeerAccessAlreadyEnabled) good = false;
            (void) cudaGetLastError();
        }
        ok[a][b] = ok[b][a] = good ? 1 : -1;
    }
    return ok[a][b] > 0;
}
bool copy2d(Stager& st, int ddev, void* dst, size_t dpitch, int sdev, const void* src, size_t spitch, size_t width,
            size_t height, std::string& err) {
    if (height == 0 || width == 0) return true;
    if (st.direct) {
        const OnDevice on(ddev);
        if (cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDefault, 0) != cudaSuccess) {
            err = "tp: a hand-over copy failed to start";
            return false;
        }
        if (ddev >= 0 && ddev < 2 && sdev >= 0 && sdev < 2) st.bytes[sdev][ddev] += (double) width * (double) height;
        if (ddev >= 0 && ddev < 64) st.used[ddev] = true;
        if (sdev >= 0 && sdev < 64) st.used[sdev] = true;
        return true;
    }
    if (!st.ensure(width * height)) { err = "tp: the hand-over staging buffer"; return false; }
    {
        const OnDevice on(sdev);
        if (cudaMemcpy2D(st.h, width, src, spitch, width, height, cudaMemcpyDeviceToHost) != cudaSuccess) {
            err = "tp: the hand-over read failed";
            return false;
        }
    }
    {
        const OnDevice on(ddev);
        if (cudaMemcpy2D(dst, dpitch, st.h, width, width, height, cudaMemcpyHostToDevice) != cudaSuccess) {
            err = "tp: the hand-over write failed";
            return false;
        }
    }
    return true;
}
}  // namespace

// [claude] the streamed K/V's host copies (pinned RAM on both sides): copied by CPU threads (STRATA_TP_MOVE_THREADS,
// default 12; 0 = through cudaMemcpy2DAsync as before).  A 261,625-cell hand-over moved ~3 GB of them through the
// copy engine (host -> host over the link) in ~250 ms, beside the late refill's copies on the same link.
struct HostJob { uint8_t* dst; size_t dpitch; const uint8_t* src; size_t spitch; size_t w, h; };
static int tp_move_threads() {
    static const int v = [] { const char* e = std::getenv("STRATA_TP_MOVE_THREADS"); return e ? std::max(0, std::atoi(e)) : 12; }();
    return v;
}
// [claude] STRATA_TP_PRECOPY_THREADS (default 4): the precopy's threads (it runs beside the prompt's last chunk: fewer
// threads leave the memory bus to it, and the chunk takes far longer than the copy)
static int tp_precopy_threads() {
    static const int v = [] { const char* e = std::getenv("STRATA_TP_PRECOPY_THREADS"); return e ? std::max(1, std::atoi(e)) : 4; }();
    return v;
}
// [claude] STRATA_TP_PRECOPY_CHECK=1 (a test): the hand-over compares the precopied pages with the stages' instead of
// trusting them, and says how many rows differ (0: the precopy was exact)
static bool tp_precopy_check() {
    static const bool v = [] { const char* e = std::getenv("STRATA_TP_PRECOPY_CHECK"); return e != nullptr && std::atoi(e) != 0; }();
    return v;
}
static size_t check_host_jobs(const std::vector<HostJob>& jobs) {
    size_t bad = 0;
    for (const HostJob& j : jobs)
        for (size_t r = 0; r < j.h; ++r)
            if (std::memcmp(j.dst + r * j.dpitch, j.src + r * j.spitch, j.w) != 0) ++bad;
    return bad;
}
static void run_host_jobs(const std::vector<HostJob>& jobs, int nt) {
    size_t rows = 0;
    for (const HostJob& j : jobs) rows += j.h;
    if (rows == 0) return;
    // row ranges of the concatenated jobs, one per thread
    auto work = [&](size_t r0, size_t r1) {
        size_t base = 0;
        for (const HostJob& j : jobs) {
            const size_t a = std::max(r0, base), b = std::min(r1, base + j.h);
            for (size_t r = a; r < b; ++r) std::memcpy(j.dst + (r - base) * j.dpitch, j.src + (r - base) * j.spitch, j.w);
            base += j.h;
            if (base >= r1) break;
        }
    };
    nt = (int) std::min<size_t>((size_t) std::max(1, nt), rows);
    std::vector<std::thread> th;
    const size_t per = (rows + (size_t) nt - 1) / (size_t) nt;
    for (int t = 1; t < nt; ++t) th.emplace_back(work, (size_t) t * per, std::min(rows, (size_t) (t + 1) * per));
    work(0, std::min(rows, per));
    for (auto& x : th) x.join();
}

bool tp_state_move(const ModelGeometry& g, const ModelGeometry* tg, SessionState* const* stage_ss, const int* stage_dev,
                   int n_stages, const int64_t* stage_lb, SessionState* const* tp_ss, const int* tp_dev, int64_t cells,
                   bool to_tp, std::string& err, int64_t host_skip_pages, int64_t host_only_pages) {
    const bool check_only = host_only_pages <= -2;   // [claude] the precopy check (STRATA_TP_PRECOPY_CHECK)
    if (check_only) host_only_pages = -2 - host_only_pages;
    const bool host_only = host_only_pages >= 0;
    using strata::kernels::QsaShapes;
    Stager stg;
    std::vector<HostJob> hjobs;
    const bool host_threads = tp_move_threads() > 0;
    const auto t_move0 = std::chrono::steady_clock::now();
    const int R = 2;
    {
        static const bool staged = [] { const char* v = std::getenv("STRATA_TP_MOVE_STAGED"); return v != nullptr && v[0] == '1'; }();
        bool direct = !staged;
        for (int st = 0; direct && st < n_stages; ++st)
            for (int r = 0; direct && r < R; ++r) direct = peer_ready(stage_dev[st], tp_dev[r]);
        stg.direct = direct;
    }
    const uint64_t S = (uint64_t) g.ssm_state_size;
    const uint64_t HV = (uint64_t) g.ssm_v_heads;
    const uint64_t gfull = S * HV * S + (uint64_t) g.ssm_conv_channels * (uint64_t) (g.ssm_d_conv - 1);
    uint64_t hvr[2], ghalfr[2];
    for (int r = 0; r < R; ++r) {
        hvr[r] = (uint64_t) tg[r].ssm_v_heads;
        ghalfr[r] = S * hvr[r] * S + (uint64_t) tg[r].ssm_conv_channels * (uint64_t) (tg[r].ssm_d_conv - 1);
    }
    const QsaShapes sf = [&] { QsaShapes s = strata::kernels::qsa_real_shapes(); s.n_head = g.n_head; s.n_head_kv = g.n_head_kv; s.head_dim = g.head_dim; s.idx_n_head = g.idx_q_heads; s.idx_dim = g.idx_key_dim; return s; }();
    auto stage_of = [&](int64_t l) { int st = 0; while (st + 1 < n_stages && l >= stage_lb[st + 1]) ++st; return st; };
    int64_t gi = 0, qi = 0;
    for (int64_t l = 0; l < g.n_layers; ++l) {
        const int st = stage_of(l);
        SessionState& S0 = *stage_ss[st];
        const int d0 = stage_dev[st];
        if (!is_qsa_layer(g, l) && host_only) {
            ++gi;
        } else if (!is_qsa_layer(g, l)) {
            float* full = S0.gdn_state + (size_t) (gi - S0.gdn_ord0) * gfull;
            for (int r = 0; r < R; ++r) {
                const uint64_t hv = hvr[r];
                float* half = tp_ss[r]->gdn_state + (size_t) gi * ghalfr[r];
                // the recurrence [S][HV][S]: this card's v-head ranges (v_heads), in local order
                uint64_t lh = 0;
                for (const Range& vr : v_heads(g, r, R)) {
                    float* f_st = full + vr.begin * S;
                    float* h_st = half + lh * S;
                    if (to_tp) { if (!copy2d(stg, tp_dev[r], h_st, hv * S * 4, d0, f_st, HV * S * 4, vr.count * S * 4, S, err)) return false; }
                    else if (!copy2d(stg, d0, f_st, HV * S * 4, tp_dev[r], h_st, hv * S * 4, vr.count * S * 4, S, err)) return false;
                    lh += vr.count;
                }
                // the conv history [C][d_conv - 1]: the q, k and v channel ranges of this card's heads
                float* fconv = full + S * HV * S;
                float* hconv = half + S * hv * S;
                const uint64_t dc = (uint64_t) (g.ssm_d_conv - 1);
                uint64_t ho = 0;
                for (const Range& cr : qkv_rows(g, r, R, S)) {
                    const size_t bytes = cr.count * dc * 4;
                    if (to_tp) { if (!copy2d(stg, tp_dev[r], hconv + ho * dc, bytes, d0, fconv + cr.begin * dc, bytes, bytes, 1, err)) return false; }
                    else if (!copy2d(stg, d0, fconv + cr.begin * dc, bytes, tp_dev[r], hconv + ho * dc, bytes, bytes, 1, err)) return false;
                    ho += cr.count;
                }
            }
            ++gi;
        } else {
            const QsaState& F = S0.qsa_states[qi];
            const uint64_t NKV = (uint64_t) g.n_head_kv, nkv = (uint64_t) tg[0].n_head_kv, PS = (uint64_t) sf.page_size, HD = (uint64_t) g.head_dim;
            const uint64_t pages_used = std::min<uint64_t>((uint64_t) F.n_pages, (uint64_t) ((cells + PS - 1) / PS) + 1);
            const uint64_t scale_row = HD / (uint64_t) strata::kernels::KV_Q8_GROUP * 2, q4_row = strata::kernels::kv_q4_bytes_per_head((int) HD);
            for (int r = 0; r < R; ++r) {
                const QsaState& H = tp_ss[r]->qsa_states[qi];
                if (H.n_pages != F.n_pages || H.n_slots != F.n_slots || H.kv_mode != F.kv_mode || H.kv_int8 != F.kv_int8 ||
                    H.kv_q4 != F.kv_q4 || H.kv_hybrid != F.kv_hybrid) {
                    err = "tp: the TP session's K/V differs from the stage's (layer " + std::to_string(l) + ")";
                    return false;
                }
                // [page][kv_head][PS][...]: card r's kv-heads of every page
                auto pool = [&](const void* fp, const void* hp, uint64_t per_head, uint64_t pages) -> bool {
                    if (fp == nullptr || hp == nullptr || pages == 0) return true;
                    const uint64_t fpitch = NKV * PS * per_head, hpitch = nkv * PS * per_head, w = nkv * PS * per_head;
                    const uint8_t* f = (const uint8_t*) fp + r * w;
                    if (to_tp) return copy2d(stg, tp_dev[r], (void*) hp, hpitch, d0, f, fpitch, w, pages, err);
                    return copy2d(stg, d0, (void*) f, fpitch, tp_dev[r], hp, hpitch, w, pages, err);
                };
                // the VRAM pools: every slot (streamed) or the used pages (resident)
                const uint64_t vp = F.kv_mode == 1 ? (uint64_t) F.n_slots : pages_used;
                if (host_only && F.kv_mode != 1) continue;
                // [claude] STRATA_TP_MOVE_NOPOOL=1: a streamed layer's VRAM pool is a cache of the host copy (every writer
                // writes the host copy): the receiving side starts with nothing resident (its map reset, the first
                // windows' resolve reads the blocks they select back) instead of copying every slot - 4 x 131 MiB
                // of device copies a hand-over on IQ4 TP (~17 ms each way at 1K)
                static const bool nopool = [] { const char* v = std::getenv("STRATA_TP_MOVE_NOPOOL"); return v != nullptr && v[0] == '1'; }();
                const bool skip_pool = nopool && F.kv_mode == 1 && H.kv_mode == 1 && !check_only;
                if (skip_pool && !host_only && (to_tp || r == 0)) {
                    const int rd = to_tp ? tp_dev[r] : d0;
                    const OnDevice on(rd);
                    strata::kernels::kv_stream_reset(to_tp ? H.map : F.map, nullptr);
                    if (cudaPeekAtLastError() != cudaSuccess) { err = "tp: the K/V map reset failed"; return false; }
                    if (rd >= 0 && rd < 64) stg.used[rd] = true;
                }
                if (!host_only && !skip_pool &&
                    (!pool(F.k_pool, H.k_pool, HD * 2, vp) || !pool(F.v_pool, H.v_pool, HD * 2, vp) ||
                     !pool(F.k_q, H.k_q, HD, vp) || !pool(F.v_q, H.v_q, HD, vp) || !pool(F.k_scale, H.k_scale, scale_row, vp) ||
                     !pool(F.v_scale, H.v_scale, scale_row, vp) || !pool(F.k_q4, H.k_q4, q4_row, vp) || !pool(F.v_q4, H.v_q4, q4_row, vp)))
                    return false;
                if (F.kv_mode == 1) {   // streamed: the host copy (the used pages), the page table, the slot map
                    // [claude] both host copies are pinned RAM: CPU threads (run_host_jobs) unless STRATA_TP_MOVE_THREADS=0
                    // [claude] pages [p0, p1) only: host_skip_pages / host_only_pages
                    const uint64_t p0 = (uint64_t) std::max<int64_t>(0, host_skip_pages);
                    const uint64_t p1 = host_only ? std::min<uint64_t>(pages_used, (uint64_t) host_only_pages) : pages_used;
                    auto hpool = [&](const void* fp0, const void* hp0, uint64_t per_head, uint64_t) -> bool {
                        if (fp0 == nullptr || hp0 == nullptr || p1 <= p0) return true;
                        const uint64_t fpitch = NKV * PS * per_head, hpitch = nkv * PS * per_head, w = nkv * PS * per_head;
                        const void* fp = (const uint8_t*) fp0 + p0 * fpitch;
                        const void* hp = (const uint8_t*) hp0 + p0 * hpitch;
                        const uint64_t pages = p1 - p0;
                        if (!host_threads && !check_only) return pool(fp, hp, per_head, pages);
                        uint8_t* f = (uint8_t*) fp + r * w;
                        if (to_tp) hjobs.push_back({(uint8_t*) hp, (size_t) hpitch, f, (size_t) fpitch, (size_t) w, (size_t) pages});
                        else hjobs.push_back({f, (size_t) fpitch, (const uint8_t*) hp, (size_t) hpitch, (size_t) w, (size_t) pages});
                        return true;
                    };
                    if (!hpool(F.host.k_pool, H.host.k_pool, HD * 2, pages_used) || !hpool(F.host.v_pool, H.host.v_pool, HD * 2, pages_used) ||
                        !hpool(F.host.k_q, H.host.k_q, HD, pages_used) || !hpool(F.host.v_q, H.host.v_q, HD, pages_used) ||
                        !hpool(F.host.k_scale, H.host.k_scale, scale_row, pages_used) || !hpool(F.host.v_scale, H.host.v_scale, scale_row, pages_used) ||
                        !hpool(F.host.k_q4, H.host.k_q4, q4_row, pages_used) || !hpool(F.host.v_q4, H.host.v_q4, q4_row, pages_used))
                        return false;
                    if (host_only || skip_pool) goto indexer;
                    {
                    auto words = [&](const int32_t* f, const int32_t* h, uint64_t n) -> bool {
                        if (f == nullptr || h == nullptr || n == 0) return true;
                        if (to_tp) return copy2d(stg, tp_dev[r], (void*) h, n * 4, d0, f, n * 4, n * 4, 1, err);
                        return copy2d(stg, d0, (void*) f, n * 4, tp_dev[r], h, n * 4, n * 4, 1, err);
                    };
                    if (!words(F.page_table, H.page_table, (uint64_t) F.n_pages) || !words(F.map.slot_block, H.map.slot_block, (uint64_t) F.n_slots) ||
                        !words(F.map.slot_stamp, H.map.slot_stamp, (uint64_t) F.n_slots) || !words(F.map.slot_ref, H.map.slot_ref, (uint64_t) F.n_slots) ||
                        !words(F.map.miss_block, H.map.miss_block, (uint64_t) F.n_slots) || !words(F.map.miss_slot, H.map.miss_slot, (uint64_t) F.n_slots) ||
                        !words(F.map.ctl, H.map.ctl, (uint64_t) strata::kernels::kKvCtlInts))
                        return false;
                    }
                }
            indexer:
                if (host_only) continue;
                // the indexer: replicated (both cards select identically)
                const uint64_t ID = (uint64_t) g.idx_key_dim, TS = (uint64_t) (sf.idx_block - 1) * ID;
                const uint64_t prow = std::min<uint64_t>((uint64_t) F.idx_pooled_rows, (uint64_t) (cells / sf.idx_block) + 2);
                auto flat = [&](const void* f, const void* h, uint64_t bytes) -> bool {
                    if (f == nullptr || h == nullptr || bytes == 0) return true;
                    if (to_tp) return copy2d(stg, tp_dev[r], (void*) h, bytes, d0, f, bytes, bytes, 1, err);
                    return copy2d(stg, d0, (void*) f, bytes, tp_dev[r], h, bytes, bytes, 1, err);
                };
                if (to_tp || r == 0) {   // back: rank 0's copy (both are the same)
                    if (!flat(F.idx_tail, H.idx_tail, TS * 4) || !flat(F.idx_dead, H.idx_dead, ID * 4) ||
                        !flat(F.idx_pooled, H.idx_pooled, prow * ID * 4) || !flat(F.idx_block_pos, H.idx_block_pos, 4))
                        return false;
                }
            }
            ++qi;
        }
    }
    // the PLE history and token window (stage 0's session holds layer 1), replicated
    if (!host_only) {
        SessionState& S0 = *stage_ss[0];
        const uint64_t hb = (uint64_t) strata::kernels::NG_HIST * strata::kernels::NG_HC_DIM * 4;
        for (int r = 0; r < R; ++r) {
            if (to_tp) {
                if (!copy2d(stg, tp_dev[r], tp_ss[r]->ple_hist, hb, stage_dev[0], S0.ple_hist, hb, hb, 1, err)) return false;
                tp_ss[r]->ple_prev[0] = S0.ple_prev[0];
                tp_ss[r]->ple_prev[1] = S0.ple_prev[1];
                tp_ss[r]->ple_token = S0.ple_token;
            } else if (r == 0) {
                if (!copy2d(stg, stage_dev[0], S0.ple_hist, hb, tp_dev[0], tp_ss[0]->ple_hist, hb, hb, 1, err)) return false;
                S0.ple_prev[0] = tp_ss[0]->ple_prev[0];
                S0.ple_prev[1] = tp_ss[0]->ple_prev[1];
                S0.ple_token = tp_ss[0]->ple_token;
            }
        }
    }
    const auto t_issue = std::chrono::steady_clock::now();
    if (!host_only && host_skip_pages > 0 && tp_precopy_check()) {
        // the skipped pages [0, host_skip_pages) (precopied, or on the way back the stages' own): compared
        std::string ce;
        (void) tp_state_move(g, tg, stage_ss, stage_dev, n_stages, stage_lb, tp_ss, tp_dev, cells, to_tp, ce, 0, -2 - host_skip_pages);
    }
    if (check_only) {   // (the check pass: compare, copy nothing)
        const size_t bad = check_host_jobs(hjobs);
        size_t rows = 0;
        for (const HostJob& j : hjobs) rows += j.h;
        std::fprintf(stderr, "strata tp: skipped pages check (%s): %zu of %zu host K/V rows differ\n", to_tp ? "to TP" : "back", bad, rows);
        return true;
    }
    run_host_jobs(hjobs, host_only ? tp_precopy_threads() : tp_move_threads());   // (the device copies run meanwhile; finish waits for them)
    const auto t_host = std::chrono::steady_clock::now();
    const bool ok = stg.finish(err);
    static const bool say = std::getenv("STRATA_TP_TIMING") != nullptr;
    if (say) {   // [claude] STRATA_TP_TIMING: the hand-over's parts
        size_t hb = 0;
        for (const HostJob& j : hjobs) hb += j.w * j.h;
        const auto ms = [](auto a, auto b) { return std::chrono::duration<double, std::milli>(b - a).count(); };
        std::fprintf(stderr, "strata tp: %s parts: issue %.1f ms, host K/V %.1f ms (%.0f MiB), device copies' wait %.1f ms; "
                             "device bytes (src->dst MiB): 0->0 %.0f 0->1 %.0f 1->0 %.0f 1->1 %.0f\n",
                     host_only ? "precopy" : "hand-over", ms(t_move0, t_issue), ms(t_issue, t_host), hb / 1048576.0, ms(t_host, std::chrono::steady_clock::now()),
                     stg.bytes[0][0] / 1048576.0, stg.bytes[0][1] / 1048576.0, stg.bytes[1][0] / 1048576.0, stg.bytes[1][1] / 1048576.0);
    }
    return ok;
}

}  // namespace strata::core
