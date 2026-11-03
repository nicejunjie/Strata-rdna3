# Reproducing the RDNA3 numbers

Everything needed to rebuild the engine and re-measure the README's table and charts on your own RX 7900 XTX
(gfx1100) from this repository alone. Every version is pinned, and `setup.py` downloads each one.

## What changed on 2026-11-02

The six configs in [`configs/`](configs/) and [`reference.json`](reference.json) now follow production as of
2026-11-02 (they were the 2026-10-13 README run before):

| setups | production release | engine | measured by |
|---|---|---|---|
| `iq3-2gpu`, `iq4-2gpu`, `q4-2gpu` | claude-20261102-p16k | `main` at `da76c449` | sweeps 2026-11-02 (`*-2gpu-p16k-english`) |
| `iq3-1gpu`, `iq4-1gpu`, `q4-1gpu` | claude-20261101-rg23 | `3dbb4051` (an ancestor of `da76c449`) | sweeps 2026-11-01 (`*-1gpu-rg23`) |

Build `main` (`da76c449` or later): every switch the 1-GPU release added is in it and off unless its config turns it
on, so one engine serves all six. What the configs now carry, against the 10-13 ones:

- 2 GPUs: `--tensor-parallel 2` (every layer on both cards for decode; the layer split stays for long prompts),
  `--spec 4`, `--adapt-every 1 --adapt-swaps 48`, `STRATA_MEGAKERNEL=1`, the TP prompt path in one chunk
  (`STRATA_TP_PREFILL=1`, `_MIN` / `_MAX` / `_CHUNK`: 3K-18K tokens for IQ3_S, 5K-18K for UD-IQ4_XS, 10K-28K for
  UD-Q4_K_XL), the TP-only refill head (`STRATA_REFILL_HEAD_TP_KEEP`, `STRATA_REFILL_HEAD_TP_MIN=8449`), the late
  refill on SDMA (`STRATA_REFILL_SDMA=1` and the other `STRATA_REFILL_*`), the TP deal / copy switches
  (`STRATA_TP_*`), `STRATA_DP_PIPE_CHUNK` (8448 / 11264 / 18432), `STRATA_TOPK_SPLIT_MIN=98304`,
  `STRATA_QSA_ROWS=1`.
- 1 GPU: `--adapt-every 1 --adapt-swaps 32` (IQ) / `48` (Q4), `STRATA_ADAPT_OVERLAP=1`, `STRATA_TOPK_SPLIT_MIN`,
  `STRATA_QSA_ROWS`, `STRATA_PREFILL_CPU_SHARE=0.6` with a per-model `_MAX` (4096 / 5120 / 7936), and
  `STRATA_PF_FUSED_KQ=1` on UD-IQ4_XS.
- all six: `--mtp-draft-vocab rdna3/draft_vocab_zh60k.bin` and `--conversation-cache-mib 8192
  --conversation-cache-slots 8` (below).

The full command lines are in [`docs/RDNA3_SETTINGS.md`](../docs/RDNA3_SETTINGS.md).

Notes on what differs from the production files, and why:

- **`draft_vocab_zh60k.bin` is shipped here** ([`rdna3/draft_vocab_zh60k.bin`](draft_vocab_zh60k.bin), 238 KB, sha256
  `e93f8cd2...6122`). It is the MTP draft head's token subset production uses (59,607 token ids, uint32, the
  format of `data/draft_vocab*.bin`): the 40.5K English/code tokens + the Han tokens covering 99.5% of the model's
  own Chinese answers + all kana, Hangul and CJK punctuation. We generated it from the model's tokenizer and its
  own answers; it is a list of token ids, nothing of a third party. Against `setup.py`'s default `cjk` subset (106K
  tokens) it makes the draft step ~12% cheaper (English decode +4%, Chinese equal). It is lossless: drafts are
  verified. Remove the two arguments to use `<mtp dir>/draft_vocab.bin` instead.
- **Conversation cache**: production runs `--conversation-cache-mib 8192 --conversation-cache-slots 8` (RAM for
  reusing a conversation's earlier turns); the configs keep it. The sweeps behind `reference.json` ran with
  `--conversation-cache-mib 0` so that no request reuses an earlier one; `bench.py`'s prompts are all different and
  every request must report none cached, so the cache only costs RAM here. Short of RAM, set it to `0`.
- **Dropped** (production plumbing, not engine settings): the proxy / llama-swap port and the private log path
  (`run.py` takes `--port` and `--log`), `server_source` (production pins the HTTP server file of an older release;
  here it is this checkout's `serve/server.py`), the per-GPU-layout tables (`aibox_layouts`, `aibox_single_gpu`:
  already resolved into each config) and the frozen per-layout executable paths and their sha256 (here: the
  engine you build). `engine_silence_s` stays at the sweeps' 1800 s (production's proxy uses 600).
- **presence_penalty 0** (the fork's default since 2026-11-02, and explicit in each config's `sampling` block).
  Upstream's `serve/server.py` (#2685) fills the sampling fields a request leaves out from the model card, and for
  Qwen3.8-Flash-Next with thinking off that card adds `presence_penalty` 1.5 (Qwen's anti-repetition default). The
  fork sets it to 0 in `MODEL_CARD_SAMPLING`; a request that sends its own `presence_penalty` still gets it. Measured
  on `iq3-2gpu` (public server, warm; 3x 16K per kind, 256 tokens; qeval2 = 40 reasoning problems with thinking on;
  tools = 24 seeded tool calls; soak = 40 long replies - 10 English, 10 Chinese, 10 code with thinking off, 10 with
  thinking on - up to 1024 / 1536 tokens, mean ~1,000 tokens; texts kept):

  | presence_penalty | 16K English decode | ms / window | tokens / window | 16K code decode | ms / window | tokens / window | qeval2 | tools | soak loops (textcheck) | replies with a 12-word n-gram repeated 4+ times | repeated 12-grams (mean share) |
  |---|---|---|---|---|---|---|---|---|---|---|---|
  | 0   | 144.8 | 14.99 | 2.17 | **177.2** | 16.46 | **2.88** | 40/40 | 24/24 | 0 / 40 | 0 / 40 | 0.28% (code 1.0%) |
  | 1.5 | 147.3 | 15.34 | 2.21 | 157.1 | 15.96 | 2.54 | 40/40 | 24/24 | 0 / 40 | 0 / 40 | 0.05% (code 0.1%) |

  1.5 showed no quality advantage (no loops either way; what it removes is code's natural repetition of names and
  calls), and it costs code decode 11% through MTP acceptance (drafts that repeat earlier tokens are rejected more).
  English is equal within noise. Before this was found, the template A/B against production (whose older server has
  no model-card defaults) read 1.99 vs 2.08 tokens a window on Q4 2 GPUs at 16K with the same ms per window.
- The vision encoder (`--vision`, the BF16 mmproj from `setup.py`) is as in production; `run.py --no-vision` drops it.

## What was measured, on what

| | |
|---|---|
| GPUs | 1 or 2 × AMD Radeon RX 7900 XTX, 24 GB each, PCIe 4.0 x8 each (HIP devices 1 and 1,0 below) |
| CPU, RAM | AMD Ryzen 9 9950X3D2, 128 GB DDR5 |
| Disk | NVMe SSD (the models, packs and the n-gram table are read from it) |
| OS | Ubuntu 26.04.1 LTS, kernel 7.0.0-38-generic with its in-kernel `amdgpu` driver (no DKMS, no system ROCm) |
| Compiler | GCC 15.2.0 (any C++20 GCC should do) |
| Python | 3.13 (any 3.10+ with `venv`) |
| ROCm | TheRock wheels `rocm-sdk-*==7.10.0a20251120`, `gfx110X-dgpu` index, installed into `.venv` by `setup.py` |
| rocWMMA | headers at commit `1ab208f4` (tag `rocm-7.1.1`), fetched by `setup.py` |
| llama.cpp (ggml, gguf-py) | commit `3cf03257`, fetched by `setup.py` |
| cmake, ninja | from `PATH` when present (here cmake 4.2.3, ninja 1.13.2), else 4.4.3 / 1.13.2 from pip (`requirements.txt`) |
| Engine | this branch (`rdna3`), compiled by `./setup.sh --build` |

RAM: a 1-GPU setup keeps the experts that do not fit the card in RAM, so you need roughly the size of the pack's
`experts.bin` plus ~15 GB: IQ3_S 50 GB, UD-IQ4_XS 60 GB, UD-Q4_K_XL 77 GB. With less, the engine reads experts from
the SSD while it answers and is much slower than the table. No `ulimit -l` change and no root are needed.

## 1. System packages

```bash
sudo apt install build-essential git python3-venv     # Fedora: gcc-c++ git python3
```

Your user must be able to open `/dev/kfd` and `/dev/dri/renderD*` (the `render` and `video` groups on Ubuntu).

## 2. Get the code

```bash
git clone https://github.com/nicejunjie/Strata-rdna3.git
cd Strata-rdna3
```

## 3. Install the toolchain and build the engine (and fetch the models)

One run per model. The first one makes `.venv`, installs ROCm from AMD's wheels, fetches llama.cpp and rocWMMA, and
compiles the engine and the CPU image encoder for gfx1100 with the fork's options (`-DSTRATA_MMQ_KQUANTS=ON` and
`STRATA_ROCWMMA_INCLUDE_DIR`; 10-20 minutes). Later runs reuse all of that. Each run downloads its model's GGUF
shards from Hugging Face (checked against their published sha256), writes its pack, and fetches the MTP draft head.

> [!WARNING]
> `setup.py` (upstream behaviour) **moves** the model files of every Strata install it ran from before, listed in
> `~/.config/strata/settings.json`, into the data folder you give it. If this PC already has a Strata install whose
> models you want to keep where they are, give this one its own settings: `export XDG_CONFIG_HOME=$PWD/.config`
> before the commands below (and before every later `./setup.sh`).

```bash
D=$PWD/../Strata-data     # models/, packs/, mtp/: an absolute path
./setup.sh --setup --build --backend hip --family qwen    --model IQ3_S      --vision cpu --data-dir $D --no-start --yes
./setup.sh --setup --build --backend hip --family unsloth --model UD-IQ4_XS  --vision cpu --data-dir $D --no-start --yes
./setup.sh --setup --build --backend hip --family unsloth --model UD-Q4_K_XL --vision cpu --data-dir $D --no-start --yes
```

Already have the GGUFs? Put them, under their original names, in `$D/models/IQ3_S/`, `$D/models/unsloth-UD-IQ4_XS/`
and `$D/models/unsloth-UD-Q4_K_XL/` before running setup (it then only checks that each file is whole), and compare
them with [`SHA256SUMS`](SHA256SUMS). `--gguf-dir` also works for running the model with `./setup.sh`, but
`rdna3/run.py` expects the files in `$D/models/`. `UD-Q4_K_XL` prints a warning about AMD cards:
that warning is upstream's, written before this fork's K-quant prompt kernels; `--yes` accepts it.

## 4. One expert file per model

The measured setups read the experts from one prepacked file per model (on Linux with `O_DIRECT`, ~1.9 GiB/s from a
PCIe 4 drive) instead of out of the GGUF shards. It is a byte-for-byte copy of the GGUF's expert tensors in the order
the engine reads them, written once:

```bash
export STRATA_GGUF_PY=$PWD/third_party/llama.cpp/gguf-py
.venv/bin/python tools/iq_pack.py --gguf $D/models/IQ3_S/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf \
    --out $D/packs/iq3_s --experts-bin
.venv/bin/python tools/iq_pack.py --gguf $D/models/unsloth-UD-IQ4_XS/Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf \
    --out $D/packs/unsloth-ud-iq4_xs --compat-bf16 --experts-bin
.venv/bin/python tools/iq_pack.py --gguf $D/models/unsloth-UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    --out $D/packs/unsloth-ud-q4_k_xl --compat-bf16 --experts-bin
```

(50, 60 and 77 GB. `--compat-bf16` is what `setup.py` passes for the Unsloth files: their Q8_0 hyper-connection
projections become BF16, the form the engine reads; without it `iq_pack.py` stops before writing anything.)

## 5. Run a setup

`rdna3/configs/` holds the six setups exactly as production runs them (2026-11-02, above): the engine arguments
and every environment switch.
`rdna3/run.py` fills in your paths and starts the OpenAI-compatible server (`/v1/chat/completions`):

```bash
.venv/bin/python rdna3/run.py iq4-2gpu --data $D --gpus 1,0      # iq3-1gpu iq3-2gpu iq4-1gpu iq4-2gpu q4-1gpu q4-2gpu
```

`--gpus` takes HIP device numbers. With two, the first holds layers 0-23 and the second 24-47 (the layer split
used by long prompts and the TP refill; decode and TP prompts run every layer on both). The README used
`1` and `1,0`; on your PC `0` / `0,1` (the default) is as good, unless one card also drives the desktop. Leave that
card out of a 1-GPU setup, or put it second. `--print-config` shows the filled-in config; the same settings are
written out in [`docs/RDNA3_SETTINGS.md`](../docs/RDNA3_SETTINGS.md).

## 6. Measure

```bash
.venv/bin/python rdna3/bench.py iq4-2gpu --data $D --gpus 1,0                        # 1K ... 256K, ~20 min
.venv/bin/python rdna3/bench.py q4-1gpu  --data $D --gpus 1 --lengths 8192,32768     # a quick check
```

`bench.py` starts the setup's server itself (stop any other one first: both GPUs must be idle) and measures the
README's way. Prompts are EXACTLY 1K, 2K, ... 128K and 261,632 tokens, built with the model's own tokenizer and chat
template. Three secret codes sit at the start, middle and end, followed by a request for a long answer. Each length
runs 3 times with 256 output tokens, temperature 0, seed 42, no prompt cache and thinking off. Every request must
report exactly n prompt tokens (none cached) and 256 completion tokens, and must return the three codes. Speeds are
the engine's own timings; a cell is the median of the three. It prints a table against
[`reference.json`](reference.json): production of 2026-11-02 (the sweeps above; `decode_code` in it is the same
lengths with a code-writing request, for reference only). Results go to `rdna3/results/<setup>/`.

What to expect: correctness checks pass every time. Speeds within a few percent of the reference on the same
hardware, with decode varying more than prefill (it depends on how many MTP drafts are accepted, and on the
cards' clocks and temperature). A 1K-8K first cell can read lower while the cards warm up.

## Checked from scratch

(This check was of the 2026-10-13 configs and reference; the 2026-11-02 configs were started from the production
models and packs and smoke-tested: a short chat and an exact 16K prompt on each of the six.)

On 2026-10-14, a fresh anonymous clone of this repository (commit `7b30f1bb`) went through steps 1-6 exactly as
written above, with system Python 3.14. The fresh install was kept apart from an existing Strata install with
`XDG_CONFIG_HOME`. The results:

- the model files matched `SHA256SUMS`;
- the three `experts.bin`, the packs' `dense.bin`, `index.txt` and `conversions.json`, and the MTP head came out
  byte-identical to the ones the README was measured with;
- all 162 benchmark requests (6 setups x 9 lengths x 3) passed every check;
- prefill was within 0.97-1.03x of `reference.json` in every cell;
- decode was within 0.95-1.09x of `reference.json` in every cell but one. The exception, `iq3-1gpu` at 4K, read
  1.26x: the reference cell there is low (86 tok/s against 95-108 for its neighbours).

