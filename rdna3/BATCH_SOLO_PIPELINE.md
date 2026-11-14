# Solo pipelines in a two-slot RDNA3 service

A two-slot layer-split service normally disables window pipelining, including
when only one request is active. `STRATA_PW_WITH_BATCH=1` lets a lone request
use the solo pipeline, while admissions and active batches keep their existing
serial slot path. The switch is off by default. This is a backport of the opt-in
and slot-drafter device binding from
[Niko1221/Strata#3095](https://github.com/Niko1221/Strata/pull/3095), with an
additional HIP prompt-read safety guard.

The first version was withdrawn after reproduced hangs. The repair below passed
the recorded HIP checks. CUDA/SYCL builds and execution are still unverified;
the PR stays draft for those checks. These results concern this machine and a
batch-enabled **layer split**, not the repository author's TP preset.

## What hung and what changed

The pipeline calls `MtpDrafter::prefill(..., sync=false)` while its host thread
also services verifier CPU doorbells. HIP's default per-group implementation
ignored `sync=false` and waited on the drafter stream. An in-flight verifier
could be waiting for that same host thread, leaving both unable to progress.
GDB caught the main thread in `hipStreamSynchronize` inside `MtpDrafter::prefill`.
It also reproduced with batch disabled, so this was not unique to slot admission.

A trial that honored `sync=false` still hung in `hipMemcpyAsync` while copying
the final stage's residual rows back to the stage-0 drafter. Replacing this with
a direct device-kernel read was invalid on this machine's cross-device memory
and was discarded. Neither core change is included in this branch.

The final change keeps **prompt reads serial on HIP when this batch/solo opt-in
is enabled**, using the already existing prompt loop. No other prompt verifier
is then waiting for the host when MTP synchronizes or transfers its residuals.
Solo **GEN still pipelines**, which is where the measured throughput benefit
comes from. The core MTP implementation and HIP's per-group policy are unchanged.
The checks explicitly keep `STRATA_MTP_PREFILL_SYNC=1`.

The backport places each slot drafter's context, RoPE and embedding on the solo
drafter's device. It also excludes batch admissions and active batches from
solo verifiers after diagnostic overrides. GPU kernels, quantization, sampling
math and MTP acceptance rules are not changed. This is a guard for the opt-in
combination, not a general repair of the existing non-batch prompt pipeline.

## Enable and disable

Build the branch using the normal HIP build. In an existing, working server
configuration, set `parallel: 2`, `max_active_requests: 2` and retain:

```text
--pipeline-windows 2 --batch-groups 1 --spec 4
--mtp /path/to/mtp --mtp-draft-vocab /path/to/rdna3/draft_vocab_zh60k.bin
```

Add these tested environment settings to its `env` object:

```json
{
  "STRATA_PW_WITH_BATCH": "1",
  "STRATA_MTP_STAGE0": "1",
  "STRATA_BATCH_MTP": "1"
}
```

The server supplies `--batch 2` from `parallel`; direct engine users supply it
themselves. Use two physical GPUs with a layer split. Tensor-parallel decode
disables the window pipeline. This does not add MTP to pipelined batch groups;
groups greater than one and coupled/repetition-penalty fallbacks retain their
existing restrictions.

Remove `STRATA_PW_WITH_BATCH` or set it to `0` and restart to restore the original
batch gate. Both the opt-in and its HIP prompt guard are then inactive.

## Same-machine measurements after repair

| Mode | Chinese 1 client | Python 1 client | Chinese 2 clients total | Python 2 clients total |
|---|---:|---:|---:|---:|
| Patched, flag off | 77.06 | 98.63 | 85.39 | 95.05 |
| Patched, flag on | 110.48 | 159.58 | 85.37 | 95.21 |
| Flag on, fresh process (one round) | 107.11 | 158.97 | 84.13 | 95.68 |

Values are completion tokens divided by full request/burst wall time, including
prefill. Off/on use three rounds; fresh-process on uses one confirmation round.
They are not the sum of separately reported instantaneous decode rates. The
single-client gains do not imply improved two-client throughput.

Environment: Ubuntu 26.04, Intel ES 14 cores/20 threads with AVX2, 96 GiB RAM,
two Sapphire RX 7900 XTX 24GB (Platinum and BLOWER), 303/231W caps, ROCm 10.1,
HIP/gfx1100 Release with MMQ and K-quants. Upstream links: PCIe 4.0 x16 and an
external CPU M.2 PCIe 3.0 x4 link. Layer split 26/22, int8 KV with 32768 resident
tokens, two 131072-token slots, prefill 16384, 1536 MiB reserve, batch groups 1,
hardware queues 6 and REFILL_SDMA 0. All arms use the fork's unchanged Python
server, the same GSQ-RCO IQ3_S weights and a fresh copy of the same expert profile.

Chinese prose and Python code allow up to 1024 output tokens, temperature 0, top-k 1,
seed 42, reasoning off, zero presence/frequency penalties and no prefix cache.
Normal early EOS is retained and its actual token count is used; individual
output lengths are in the record. The exact short prompts and all runs are in the smoke script and
[repair validation record](tests/results/pw-with-batch-fixed-gfx1100-20261114.json).
Fixed ordering and one three-round trial per primary arm limit small-difference
claims. The profile adapts during performance runs; placement is frozen only for
the separate output comparison.

## Checks and reproduction

The native-server validation completed 280 requests across off, on,
fresh-process on and frozen-placement comparison arms. It checks repeated exact
echo/JSON, nonzero penalties, an admission after the first request emits,
overlapping generation and return to solo. Each of the first three arms also
reads actual tokenizer-measured 8192/32768/122880-token retrieval prompts, followed
immediately by repeated short echo/JSON requests. The long requests are solo;
two simultaneous 120K requests were not measured.

Frozen-placement checks compare serial, pipeline, repeated serial and forced-miss
pipeline output for echo, JSON, arithmetic, code and prose. See the record for
the individual hashes and outcomes. This is a small correctness check, not a
full quality evaluation or proof of every possible interleaving. A separate
frozen-placement comparison checks ten stock/off output pairs.

To repeat the standalone smoke and throughput checks, start an isolated server
with no other clients, then run:

```sh
python3 rdna3/tests/pw_batch_smoke.py \
  --url http://127.0.0.1:8101 \
  --model Qwen3.8-Flash-Next-IQ3_S \
  --benchmark --repeats 3 --allow-early-stop --output fixed-on.json
```

Restart with the flag off for the control, restoring the same initial expert
profile each time. The command covers short requests and slot transitions;
the attached record's full long-prompt sequence additionally sends the three
retrieval prompts and repeats the smoke checks afterward.

## Limits and retained failures

The [initial validation record](tests/results/pw-with-batch-gfx1100-20261114.json)
retains the withdrawn candidate's watchdog failures and earlier measurements.
Those are not passes for the repaired revision. The new record identifies both
source hashes, binary hash, server hash, settings, output hashes and health
counters. Daily service and GPU settings were restored after the checks.

An earlier throughput trial of the same repaired executable ended one Chinese
answer normally at 669 tokens. The strict-length harness stopped the trial, so
its later long checks did not run. All twelve bursts from that trial are retained
separately in the repair record. The current run allows normal EOS and uses
actual completion lengths without changing the model's stopping rules.

Only HIP/gfx1100 on ROCm 10.1 was built and exercised. CUDA/SYCL, other GPUs,
TP mode, groups greater than one, full model-quality testing, forced-pageable
diagnostics and allocation-failure injection were not run. Extra verifiers use
VRAM and can reduce expert-cache capacity. This machine has an asymmetric
external link with correctable AER events; the record includes their count.
Measure your own workload before enabling the opt-in.

Both controls and candidate reuse the same local ggml/rocWMMA source archives.
Their Git metadata is unavailable; the record includes file probes and hashes,
with commits explicitly unknown. This is not a clean-room installer reproduction.
