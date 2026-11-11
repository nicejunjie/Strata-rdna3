// src/core/arch_defaults.cpp - see include/strata/core/arch_defaults.hpp.
#include "strata/core/arch_defaults.hpp"

#include "strata/kernels/gfx_arch.hpp"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::core {

std::vector<std::pair<std::string, std::string>> arch_default_env(const char* gcn_arch) {
    std::vector<std::pair<std::string, std::string>> t;
    if (strata::kernels::gfx_arch_is_gfx12(gcn_arch)) {
        // gfx1200 / gfx1201 (RDNA4): only the PROMPT half of the gfx1151 table.  Reported exact (identical answers) on an R9700
        // in #1478 and re-checked bitwise here (notes-142-amd.md); the decode half measured slower on that card, so it stays out.
        // STRATA_HCD_EXACT is left out: it needs the gfx1151 hipBLASLt tuning table, finds none on gfx12, says so on every start
        // and does nothing (measured: with and without it identical, 0.0 to +0.1%).
        // STRATA_GDN_CONVL2 is AMD-only on purpose: it is not bitwise on CUDA 13.3 sm_120 (#1522), and this table is never
        // consulted by a CUDA build.
        const char* off12 = std::getenv("STRATA_GFX12_DEFAULTS");
        if (off12 != nullptr && off12[0] == '0') return t;
        t = {
            {"STRATA_GDN_HEAD", "1"},
            {"STRATA_GDN_PP", "2"},
            {"STRATA_GDN_CONVL2", "1"},
            {"STRATA_GDN_NOY", "1"},
            {"STRATA_CVEC_FUSE", "1"},
        };
        return t;
    }
    if (!strata::kernels::gfx_arch_is_gfx1151(gcn_arch)) return t;
    const char* off = std::getenv("STRATA_GFX1151_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    // Exact (bitwise) on Aurora, each with its measured gain in aurora_s23.md:
    t = {
        // prompt
        {"STRATA_GDN_HEAD", "1"},        // the GDN recurrence, four lanes a column + the grid-stride norm (S23)
        {"STRATA_GDN_PP", "2"},          //   two columns per lane: 64K prompt +2.9% (stream A, round 2)
        {"STRATA_GDN_CONVL2", "1"},      // conv + q/k L2 norm in one kernel: 64K -0.3 s (stream A)
        {"STRATA_GDN_NOY", "1"},         // no dead FP32 store of the norm (stream A)
        {"STRATA_CVEC_FUSE", "1"},       // a steered layer's write, control vector and next norm in one pass (S23)
        {"STRATA_HCD_EXACT", "1"},       // the HC down GEMM in hipBLASLt solution 1176's k order: 64K +1.8% (stream S)
        {"STRATA_PF_PAD", "1"},          // padded GEMM row strides (acts with STRATA_PF_GEMM; inert without it) (S23)
        {"STRATA_TOPK_REG_MIN_BLOCKS", "4096"},   // the register top-k from 16K cells, not 28K: same ids; per 2,048 queries the ref takes 1.0 / 2.4 / 3.9 ms at 16K / 20K / 24K cells, the scan 0.87 / 0.93 / 1.00
        // decode
        {"STRATA_Q8_PACKED", "1"},       // the packed Q8_0 decode layout: +2.3% (S26)
        {"STRATA_Q6_PACKED", "1"},       // the packed Q6_K heads (S26)
        {"STRATA_MMVF_ROWS", "1"},       // bf16 multi-row GEMV, 4 rows per block (S25)
        {"STRATA_ATTN_LANECELL", "1"},   // decode attention scores, one cell per thread (S25)
        {"STRATA_EXPERT_V2", "1"},       // the grouped decode experts, IQ3_S gate/up + IQ4_NL down (S26)
        {"STRATA_TSUM", "1"},            // several warp sums as one transposed butterfly (S26)
        {"STRATA_LFUSE", "1"},           // fewer launches around the shared expert and the KV append (S26)
        {"STRATA_GDN_SPLIT", "1"},       // the GDN step over 4 blocks per head (S25/S26)
        {"STRATA_QFUSE", "1"},           // activation q8_1 images written by their producers (S26)
        {"STRATA_PLE_BATCH", "1"},       // the verify window's PLE key / value projections at once (S25)
        {"STRATA_SH_STREAM", "1"},       // the shared expert on its own stream: decode +1.8% / +6.7% (UD-Q4_K_XL) (140-m)
    };
    return t;
}

// [claude] 2026-11-11 (review 2, section 4): the discrete gfx1100 (RX 7900 XTX) settings that every production entry of a
// line set identically, as defaults (a setting in the environment always wins; STRATA_GFX1100_DEFAULTS=0 turns the table off).
// Each value is the one the aibox profiles ran with when it moved here; the per-pack thresholds stay in the profiles.
// Coupled groups are kept in one tier: PF_NO_GATED + HC_UPMIX; TP_PREFILL + DP_PREFILL + KV_HOST_DMA + the GDN set;
// TP_SHARE_DP + DP_PREFILL; PF_PAD + PF_GEMM; PA_FAST + HIP_WMMA; PREFILL_SMALL_MAX + _SMALL_RING; the REFILL_* set.
std::vector<std::pair<std::string, std::string>> gfx1100_default_env(const char* gcn_arch, int n_devices, bool tensor_parallel) {
    std::vector<std::pair<std::string, std::string>> t;
    if (gcn_arch == nullptr || n_devices < 1 || !strata::kernels::gfx_arch_is(gcn_arch, "gfx1100")) return t;   // (0: count unknown)
    const char* off = std::getenv("STRATA_GFX1100_DEFAULTS");
    if (off != nullptr && off[0] == '0') return t;
    t = {   // tier A: one card or two
        {"STRATA_HIP_WMMA", "1"}, {"STRATA_SELECT_WMMA", "1"}, {"STRATA_RDNA_MMVQ_WAVE", "3"},
        {"STRATA_COMBINE_CPU", "1"}, {"STRATA_EMB_REUSE_ACCOUNT", "1"}, {"STRATA_FUSE_CONVAB", "1"},
        {"STRATA_GDN_CHUNKED", "1"}, {"STRATA_GDN_CHUNK_GDV", "0"}, {"STRATA_GDN_CONVL2", "1"}, {"STRATA_GDN_HEAD", "1"},
        {"STRATA_GDN_NOY", "1"}, {"STRATA_GDN_PP", "1"},
        {"STRATA_GR_FUSED", "1"}, {"STRATA_GR_INJ_UP", "1"}, {"STRATA_GR_RDNA", "1"}, {"STRATA_GR_RDNA_A8", "1"}, {"STRATA_GR_RDNA_Q8", "1"},
        {"STRATA_HC_UPMIX", "1"}, {"STRATA_HEAD_MIX_MULTI", "1"}, {"STRATA_KV_HOST_DMA", "1"}, {"STRATA_PA_FAST", "1"},
        {"STRATA_PF_F16", "2"}, {"STRATA_PF_FUSED", "1"}, {"STRATA_PF_GEMM", "1"}, {"STRATA_PF_HCDOWN", "1"},
        {"STRATA_PF_NO_GATED", "1"}, {"STRATA_PF_PAD", "1"}, {"STRATA_QFUSE", "1"}, {"STRATA_QSA_ROWS", "1"},
        {"STRATA_RDNA_Q6K", "1"}, {"STRATA_ROUTE_PUBLISH", "1"}, {"STRATA_ROUTE_TAIL_SKIP", "0"}, {"STRATA_STAGER_RING", "64"},
        {"STRATA_TOPK_SPLIT_MIN", "98304"}, {"STRATA_VERIFY_QDEDUP", "1"}, {"STRATA_WAIT_FUSE", "1"},
    };
    if (tensor_parallel && n_devices >= 2) {   // tier B: the TP pair
        const std::vector<std::pair<std::string, std::string>> b = {
            {"STRATA_CKPT_POOL", "1"}, {"STRATA_CKPT_THREADS", "1"}, {"STRATA_DP_PREFILL", "1"}, {"STRATA_MEGAKERNEL", "1"},
            {"STRATA_MTP_DRAFT_ALL", "1"}, {"STRATA_MTP_GR_R8", "1"}, {"STRATA_MTP_MULTICOPY", "1"}, {"STRATA_MTP_QFUSE", "1"},
            {"STRATA_SPLIT_MTP_BATCH", "1"}, {"STRATA_PREFILL_HELP", "1"}, {"STRATA_PREFILL_LEND_PCT", "93"},
            {"STRATA_PREFILL_RING", "384"}, {"STRATA_PREFILL_SMALL_MAX", "4096"}, {"STRATA_PREFILL_SMALL_RING", "96"},
            {"STRATA_PREFILL_STREAM_MIN", "480"}, {"STRATA_FUSED_ROUTED", "1"}, {"STRATA_FUSED_CPU_FRAC", "0.6"},
            {"STRATA_REFILL_OVERLAP", "1"}, {"STRATA_REFILL_SDMA", "1"}, {"STRATA_REFILL_EARLY_TP", "1"},
            {"STRATA_REFILL_JOIN", "landed"}, {"STRATA_REFILL_JOIN_NOW", "1"}, {"STRATA_REFILL_TRANCHE_SLOTS", "128"},
            {"STRATA_REFILL_HEAD_TP_MIN", "8449"}, {"STRATA_TP_ALIAS_MAP", "rows"}, {"STRATA_TP_BACK_SKIP", "1"},
            {"STRATA_TP_PRECOPY", "1"}, {"STRATA_TP_MOVE_NOPOOL", "1"}, {"STRATA_TP_NOMEMSET", "1"}, {"STRATA_TP_PREFILL", "1"},
            {"STRATA_TP_PREFILL_OWNROWS", "1"}, {"STRATA_TP_SHARE_DP", "1"},
        };
        t.insert(t.end(), b.begin(), b.end());
    } else if (!tensor_parallel && n_devices == 1) {   // tier C: one card
        const std::vector<std::pair<std::string, std::string>> c = {
            {"STRATA_ADAPT_OVERLAP", "1"}, {"STRATA_PREFILL_CPU_SHARE", "0.6"}, {"STRATA_PREFILL_EQUAL", "1"},
        };
        t.insert(t.end(), c.begin(), c.end());
    }
    return t;
}

bool igpu_prompt_defaults_arch(const char* gcn_arch) {
    return strata::kernels::gfx_arch_is(gcn_arch, "gfx1103") || strata::kernels::gfx_arch_is(gcn_arch, "gfx1150") ||
           strata::kernels::gfx_arch_is(gcn_arch, "gfx1151");
}

bool igpu_prompt_set_enabled(const char* gcn_arch, bool windows) {
    const char* off = std::getenv("STRATA_IGPU_DEFAULTS");
    if (off != nullptr && off[0] == '0') return false;
    const char* e = std::getenv("STRATA_IGPU_PROMPT_SET");
    if (e != nullptr && e[0] == '1') return true;
    if (e != nullptr && e[0] == '0') return false;
    return kIgpuPromptSetDefaultOn && !windows && strata::kernels::gfx_arch_is(gcn_arch, "gfx1103");
}

bool igpu_uma_arena_default(const char* gcn_arch, bool integrated, bool windows, bool on) {
    if (!integrated) return false;
    const char* reg = std::getenv("STRATA_UMA_REGISTER");
    if (reg != nullptr && reg[0] == '1') return false;
    // 0.2.1.2: a Strix Halo (gfx1151) on Windows.  Registering the 47 GiB arena with the driver took ALL the free VRAM (91 GiB before,
    // 0 after; BIOS carve-out 64 GB + 32 GB shared): the expert cache got 0 slots and the engine exited ("the embedding buffer
    // failed"), 3 of 3 starts (notes-022-strix.md).  Unregistered, the arena is held by the working-set lock and the cache gets
    // 24,576 slots.  Independent of `on` (that is the 780M's opt-in).  STRATA_ARENA_PIN_GIB=N / 0 still registers it.
    if (windows) return strata::kernels::gfx_arch_is(gcn_arch, "gfx1151");
    return on && strata::kernels::gfx_arch_is(gcn_arch, "gfx1103");
}

std::vector<std::pair<std::string, std::string>> igpu_default_env(const char* gcn_arch, bool integrated) {
    std::vector<std::pair<std::string, std::string>> t;
    if (!integrated || !igpu_prompt_defaults_arch(gcn_arch)) return t;
#if defined(_WIN32)
    const bool win = true;
#else
    const bool win = false;
#endif
    if (!igpu_prompt_set_enabled(gcn_arch, win)) return t;
    // Measured on a Radeon 780M (gfx1103, P2006, Q2_0, notes-153-igpu.md / notes-148-amd.md): prompts 109.5 -> 214.7 tok/s
    // with the first four, 233 with HC_UPMIX, decode unchanged.  Dense projections are 63% of a prompt's GPU time there.
    t = {
        {"STRATA_HIP_WMMA", "1"},      // the prompt attention on matrix cores
        {"STRATA_PF_GEMM", "1"},       // the prompt's FP16 projections on the 128 x 256 WMMA GEMM
        {"STRATA_PA_FAST", "1"},       // the prompt attention with single FP16 q and p
        {"STRATA_SELECT_WMMA", "1"},   // the block scorer on matrix cores
        {"STRATA_HC_UPMIX", "1"},      // the hyper-connection up projection with its epilogue
    };
    return t;
}

static void set_default(const std::pair<std::string, std::string>& kv) {
#if defined(_WIN32)
    _putenv_s(kv.first.c_str(), kv.second.c_str());
#else
    setenv(kv.first.c_str(), kv.second.c_str(), 0);
#endif
}

std::vector<std::string> apply_arch_defaults(const char* gcn_arch, bool integrated, int n_devices, bool tensor_parallel) {
    std::vector<std::string> set;
    std::vector<std::string> g;   // (the gfx1100 table: its own line, joined to the list at the end)
    {
        for (const auto& kv : gfx1100_default_env(integrated ? nullptr : gcn_arch, n_devices, tensor_parallel)) {
            if (std::getenv(kv.first.c_str()) != nullptr) continue;
            set_default(kv);
            g.push_back(kv.first);
        }
        if (!g.empty()) {
            std::fprintf(stderr, "strata: gfx1100 (%s): %zu tuned switches on by default (STRATA_GFX1100_DEFAULTS=0 turns them off; a switch "
                                 "you set is kept): ", tensor_parallel && n_devices >= 2 ? "TP pair" : n_devices == 1 ? "one card" : "cards", g.size());
            for (size_t i = 0; i < g.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", g[i].c_str() + 7);
            std::fprintf(stderr, "\n");
        }
    }
    for (const auto& kv : arch_default_env(gcn_arch)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;   // the user's setting wins, whatever it is
        set_default(kv);
        set.push_back(kv.first);
    }
    if (!set.empty()) {
        const bool g12 = strata::kernels::gfx_arch_is_gfx12(gcn_arch);
        std::fprintf(stderr, "strata: %s: %zu exact speed switches on by default (%s=0 turns "
                             "them off; a switch you set is kept): ",
                     g12 ? "gfx12 (RDNA4)" : "gfx1151 (Strix Halo)", set.size(),
                     g12 ? "STRATA_GFX12_DEFAULTS" : "STRATA_GFX1151_DEFAULTS");
        for (size_t i = 0; i < set.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", set[i].c_str() + 7);
        std::fprintf(stderr, "\n");
    }
    std::vector<std::string> igpu;
    for (const auto& kv : igpu_default_env(gcn_arch, integrated)) {
        if (std::getenv(kv.first.c_str()) != nullptr) continue;
        set_default(kv);
        igpu.push_back(kv.first);
    }
    if (!igpu.empty()) {
        std::fprintf(stderr, "strata: integrated GPU (%s): the matrix-core prompt set is on (1.6x to 2x faster prompts on a "
                             "Radeon 780M, the answers differ in the last bits): ", gcn_arch);
        for (size_t i = 0; i < igpu.size(); ++i) std::fprintf(stderr, "%s%s", i ? " " : "", igpu[i].c_str() + 7);
        std::fprintf(stderr, ".  Turn one off with NAME=0 (for example STRATA_HC_UPMIX=0), all of them with STRATA_IGPU_PROMPT_SET=0");
#if defined(_WIN32)
        std::fprintf(stderr, ".  Not yet tested on Windows");
#endif
        std::fprintf(stderr, "\n");
        set.insert(set.end(), igpu.begin(), igpu.end());
    }
    set.insert(set.end(), g.begin(), g.end());
    return set;
}

}  // namespace strata::core
