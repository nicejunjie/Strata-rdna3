# Community benchmark on AMD Radeon PRO W7800 48 GB (gfx1100), Ubuntu 24.04.4, Ryzen 7 9800X3D

Measured on 2026-11-10. Strata engine **0.1.49.1** built from source for gfx1100,
`Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S`, one card, 262,144-token context, `--batch 2` with MTP.
Runs: the standard community suite from `tools/strata_bench.py`, a 128K fresh-prompt
sweep, and a concurrency scan 1→4 clients. **This answers the "AMD RX 7000 series
(RDNA 3) on Linux" line of [TEST_REQUESTS.md](../TEST_REQUESTS.md): none yet.**

Main limitation: one machine, one engine session for the suite, **power limit and clocks
not captured**, and the GPU name had to be read from the engine log — see
[Two things the tooling did not capture](#two-things-the-tooling-did-not-capture).

## Hardware and software

- **GPU:** AMD Radeon PRO W7800 48 GB (gfx1100), `DEVICE_ID 0x7449`, subsystem
  `0x1458:0x2428`, 51,522,830,336 B (47.98 GiB) VRAM as the kernel reports it.
  The engine logs `GPU 0: AMD Radeon PRO W7800 48GB (gfx1100)`; `amd-smi static` reports
  `MARKET_NAME: AMD Radeon Pro W7800 48GB`. **Power limit and clocks not recorded.**
  Note: `results.json` in this folder says `"gpu": ["AMD Radeon"]` and contains no
  `W7800` or `gfx1100` string — **the model name in this README was added by hand** from
  the two sources above, not read by the tool. See
  [Two things the tooling did not capture](#two-things-the-tooling-did-not-capture).
- **PCIe:** card0 `current_link_speed 16.0 GT/s`, `current_link_width 16`,
  `max_link_speed 16.0 GT/s`, `max_link_width 16` — a full PCIe 4.0 x16 link, not degraded.
- **CPU / RAM:** AMD Ryzen 7 9800X3D, 8 cores / 16 threads, 92.0 GiB installed RAM.
- **OS:** Ubuntu 24.04.4 LTS, kernel `7.0.0-42-generic`, amdgpu driver.
- **ROCm:** 10.1.0 (`/opt/rocm-10.1`), hipBLASLt tuning table
  `tools/hip/gfx1100-hipblaslt-100401.txt` (1,784 rows, calibrated on this card).
- **Engine:** source build, `strata 0.1.49.1`, binary sha256 prefix `c802383d6d49bfac`,
  from the `Strata-rdna3` fork at commit `f086ec6d` (base `c9bb9d4e`
  `upstream-merge-20261024`). Not an official release.
- **Background:** the box runs this engine as its production server; no other benchmark
  or build ran during the measurements. Nothing was isolated or CPU-pinned.

## Model and configuration

- `Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S` GGUF, two shards:
  - `...-00001-of-00002.gguf` 54,817,524,224 B, SHA-256
    `4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3`
  - `...-00002-of-00002.gguf` 28,800,138,432 B, SHA-256
    `316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113`
  - Repository and revision not recorded (downloaded by setup, no manifest kept).
- Expert profile `data/expert-profile.bin`, SHA-256
  `8f59b4aa8873209dff11c11e37bcda9529a1335b724a1afeea37bf6388975baf`.
- MTP draft head `mtp/rt` (built from the model's safetensors; revision not recorded).
- **Context** 262,144 (`--max-context`), **KV** `int8` with 32,768 of 262,144 cells per
  QSA layer resident in VRAM and the K/V in 3.09 GiB of pinned RAM.
- **Expert cache** `auto` → 14,588 slots, 38.39 GiB free with 2,048 MiB reserved
  (+218 MiB for the draft head). **1,946 MiB of VRAM free** with everything loaded.
- **Speculation** `--spec 5 --spec-min-p 0.70` plus the MTP draft layer, `--mtp-q4 all`.
- `--prefill auto`, `--pool-workers 5`, `--vram-reserve-mib 900`, `--ple-io direct`,
  `--prompt-cache 24 --prompt-cache-spread --prompt-cache-tail --prompt-cache-every 16384`,
  `--conversation-cache-mib 10240 --conversation-cache-slots 4`, `--vision`,
  `--mmap-experts --resident-cpu-experts`, **`--batch 2`**.

## Method

Three separate measurements, all against the running server's HTTP API:

1. **Community suite**: `python tools/strata_bench.py --url <server> --clients 2 --long`,
   3 runs per test, medians. Thinking off, temperature 0. Speeds are the engine's own
   `timings`; first-token and latency are measured by the client.
2. **128K fresh prompts**: three requests of 201,714–201,715 prompt tokens, each prefixed
   with a unique nonce so the conversation cache cannot help (`cached_tokens: 0` in all
   three), 24 new tokens, `temperature 0`.
3. **Concurrency scan**: `tools/serve_load.py --clients 1,2,3,4 --rounds 3 --max-tokens 128`
   against the same `--batch 2` server — 1, 2, 3 and 4 simultaneous requests.

## Results

### Community suite (medians, 3 runs per test)

| Configuration | Prompt tokens | Generated | Prompt tok/s | Decode tok/s | TTFT (s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| short chat | 46 | 100 | not measured | 91.8 (56.0–135.2) | 0.27 |
| 4K prompt, fresh | 4,019 | 24 cap | **1,462.7** (1,460.6–1,467.3) | not measured | 2.77 |
| 16K prompt, fresh | 16,058 | 24 cap | **2,284.6** (2,270.0–2,289.5) | not measured | 7.05 |
| 32K prompt, fresh | 32,049 | 24 cap | **2,441.9** (2,441.0–2,453.5) | not measured | 13.16 |
| agent follow-up turns | 14,573 at the last turn | up to 64 | see below | 99.0 (67.7–117.0) | 2.41 |
| 2 clients at once | — | 256 total | — | 83.2 total (78.9–84.3) | p50 0.611 / p95 0.854 |

Prompt throughput **rises** with length on this card (1,463 → 2,285 → 2,442 tok/s): the
fixed overhead of the first chunk dominates at 4K, and the prefill path reaches steady
state by 16K. The 4K figure is therefore the pessimistic end, not the card's ceiling.

### 128K fresh prompts (3 runs, all `cached_tokens: 0`)

| Run | Prompt tokens | Cached | Prompt tok/s | Wall (s) | TTFT (s) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 201,715 | 0 | 2,266.8 | 89.32 | 88.99 |
| 2 | 201,714 | 0 | 2,256.0 | 89.90 | 89.41 |
| 3 | 201,714 | 0 | 2,263.7 | 89.53 | 89.11 |

**Median 2,263.7 tok/s**, tight to ±0.5 %. At 201K tokens the prefill rate holds at the
16K/32K level — the card does not degrade on long prompts. Note the requested 131,072
characters expanded to 201.7K tokens by the tokenizer; that is the measured prompt size.

### Agent session (conversation cache)

Cache hit over the follow-up turns **69.3 %** (69.3–69.7); earlier-turn tokens read again
per session 144; first turn (nothing cached) TTFT 1.29 s.

| Turn | Prompt tokens | Reused | Read | Prompt tok/s | TTFT (s) | Decode tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 609 | 0 | 609 | 488.0 | 1.266 | 96.7 |
| 1 | 4,108 | 635 | 3,473 | 1,275.9 | 2.743 | 99.0 |
| 2 | 5,641 | 4,101 | 1,540 | 796.2 | 1.955 | 91.8 |
| 3 | 9,074 | 5,634 | 3,440 | 1,239.8 | 2.797 | 81.2 |
| 4 | 11,821 | 9,127 | 2,694 | 1,127.6 | 2.412 | 117.0 |
| 5 | 14,573 | 11,856 | 2,717 | 1,135.2 | 2.416 | 75.7 |

### Concurrency scan on `--batch 2` (3 rounds each, 128 new tokens per request)

| Clients | Total tok/s | Per-request tok/s | TTFT p50/p95 (s) | Latency p50/p95 (s) | Errors |
| ---: | ---: | ---: | --- | --- | ---: |
| 1 | 67.4 | 81.9 | 0.34 / 0.68 | 1.90 / 2.38 | 0 |
| 2 | **85.0** | 52.4 | 0.52 / 0.70 | 2.97 / 3.01 | 0 |
| 3 | 79.2 | 49.3 | 0.70 / 3.29 | 3.45 / 4.87 | 0 |
| 4 | 85.8 | 49.4 | 2.01 / 3.71 | 4.60 / 5.97 | 0 |

**This is the shape `--batch 2` predicts, and it is the honest reading of the setting:**
1→2 clients gains +26 % total throughput (67.4 → 85.0), then 3 and 4 add nothing
(79.2 / 85.8, inside noise) while per-request speed falls 52.4 → 49.4 and p95 TTFT goes
0.70 → 3.71 s. Requests beyond the second **wait their turn**; they do not run in
parallel. So `--batch 2` buys ~+26 % aggregate for interactive use and nothing beyond two
concurrent conversations — users wanting more need a larger `--batch`, which these runs
did **not** test. No errors at any client count.


### Batch size against the VRAM budget (six engine restarts, one batch value each)

The 48 GB card holds one resident model, so the batch size decides how much of the VRAM is left for
experts. Startup figures below are the engine's own lines; the throughput rows are `serve_load.py`
(`--max-tokens 128`, 3 rounds). Full numbers: `batch-vram-sweep.json`.

| `--batch` | expert cache | expert slots | vs batch 2 | VRAM free |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 38.47 GiB | 15,069 | — | 786 MiB |
| 3 | 37.47 GiB | 14,665 | −404 | 784 MiB |
| 4 | 36.47 GiB | 14,260 | −809 | 786 MiB |
| 5 | 35.47 GiB | 13,856 | −1,213 | 782 MiB |
| 6 | 34.47 GiB | 13,452 | −1,617 | 780 MiB |
| 8 | 32.48 GiB | 12,643 | −2,426 | 778 MiB |

**Every extra slot costs about 1 GiB of VRAM and displaces about 404 expert slots.** At 262,144-token
context with `--kv int8` and MTP, that is the slot's own KV plus its verification window.

Concurrency at each batch value (total tok/s; ✗ = collapse):

| `--batch` | c=1 | c=2 | c=3 | c=4 | c=6 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 68.5 | **84.2** | — | — | — |
| 3 | 66.1 | 82.9 | **93.3** | — | — |
| 4 | 67.1 | 81.3 | 91.7 | **0.1** ✗ | — |
| 5 | 64.4 | 81.5 | 91.8 | 0.1 ✗ | 0.1 ✗ |
| 6 | **0.0** ✗ | — | — | 0.1 ✗ | 0.1 ✗ |

Two cliffs. **At 4 concurrent requests** the rate falls from ~90 to 0.1 tok/s — 5 to 9 tokens per
round, a 22–24 s first token. **At batch 6 even a single request gets 0 tokens in 21.6 s.**
`errors` is 0 in every row: the server does not fail, it stops keeping up.

Batch 3 is the sweet spot on this card: **93.3 tok/s at 3 concurrent requests, +10.8 % over batch 2's
two-request 84.2**. Batch 4 and up is a net loss. Note the gain arrives only when the third
conversation is actually used: with 2 concurrent requests batch 3 is 1.5 % slower than batch 2.

### Why it collapses: VRAM budget, not the SSD

The tempting explanation is disk reading, and it is wrong. The engine counts its own expert-file
reads, and `files N blobs` stays **0 at every batch value** while `resident RAM` grows with the batch
(19.52 → 21.53 → 23.53 GiB). The 50 GB "OS read" line is `(0 major faults)` with
`page cache held 25269.6 MB` — it is a page-cache copy **out of RAM**, not a disk read.

The mechanism is the RAM→VRAM tier:

```text
--batch up → each slot takes 1 GiB of VRAM → expert cache loses ~404 slots per slot
           → hot experts no longer fit in VRAM → they must come from the RAM tier each step
           → 33–35 GB moved per decode round over PCIe → prefill and decode fall together
```

`exchanged with the VRAM tier` falls from 649 (batch 2) to 0 (batch 6) not because nothing moves but
because the experts sit entirely in the RAM tier and every step fetches them. `decode gate: cache
room 0.82–0.84` and `page cache held 0% of the decode's reads` show the decode waiting on the copy.

**So the limit is PCIe bandwidth between the RAM tier and VRAM, not disk.** A larger batch needs a
smaller context or an explicit `--expert-cache` before it needs a faster SSD.

### What this machine runs

Batch 3, `"parallel": 3`, `"max_queue": 3` (admission limit 6): 9 simultaneous requests returned
**6 × 200 and 3 × 429**, and 3 concurrent requests measured **95.3 tok/s**, +13.2 % over batch 2.
`admission_limit = batch + max_queue` (the server's own formula), so the queue is bounded rather than
unlimited, which is the safe choice for a served box.

## Two things the tooling did not capture

Both are small gaps worth reporting because they bite anyone sharing a report:

1. **The GPU name came out as `AMD Radeon`**, so the generated folder is named
   `...-community-unknown-qwen3...`. `strata_bench.py` asks `collect_machine()` for the
   hardware name, which reads `/v1/status`, `/metrics`, `/props` and `/v1/models`; on this
   build the card name is not among them, so the report and the folder say "unknown".
   The engine does know the name (it logs `GPU 0: AMD Radeon PRO W7800 48GB (gfx1100)`)
   and `amd-smi static` reports `MARKET_NAME`. I filled the name in by hand above.
2. **`/v1/models` does not name the architecture for a client.** It returns
   `id`, `status`, `meta.n_ctx`, `architecture.{input_modalities,output_modalities}` —
   enough for a client to pick a model, not enough to report what it ran on.

Neither is a defect in the suite; both just mean a contributor must add hardware detail by
hand. A GPU name in the machine line the tool already writes would remove the guesswork.

## What this run does and does not answer

- **Answers:** the RX 7000 / RDNA 3 on Linux question (gfx1100, single card, ROCm 10.1,
  256K context, MTP): prompt 1,463 / 2,285 / 2,442 tok/s at 4K / 16K / 32K, **2,264 tok/s
  at 201K**, decode ~92 tok/s (chat) and 99 tok/s (agent turns), agent cache hit 69 %,
  concurrency shape of `--batch 2`. No crashes, no GPU resets, zero errors across every
  request in this report.
- **Does not answer:** power limit, clocks, thermals, VRAM measured by `rocm-smi`
  (unavailable on this host — `amd-smi` used instead); `--batch` above 2 (the box holds a
  single resident 48 GB model, so a second instance could not be started beside the
  production one and the service could not be stopped without interactive authorisation);
  a fresh-process comparison for the 4K/16K/32K sweep (one engine session, as the suite
  runs); the standard benchmark protocol (`strata bench run`, thinking on, model-card
  sampling) — this is the quick suite, which the guide says is not comparable with
  standard results.

## Files

- `README.md` this report
- `results.json` every request of the community suite (engine timings, usage, machine)
- `long128k.json` the three 128K fresh-prompt requests
- `concurrency.json` the 1/2/3/4-client scan
- `batch-vram-sweep.json` the six batch values: VRAM budget, expert slots, throughput, tier state

Per-request data is in the JSON files; answers were not saved, so no model output is shared.
