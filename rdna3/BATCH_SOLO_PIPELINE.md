# Solo window pipelines in a two-slot RDNA3 service

**Draft: a watchdog exit was observed with this opt-in enabled.** After the
throughput and long-context requests passed, a short JSON request stalled and the
engine exited with code -6. This branch is for investigation; it is not ready for
production use or merge. The original gate remains the default.

The fork's own unmodified server also stalled on the candidate's first short echo
after a fresh load; it reported engine exit -15. Its stock control passed. The
problem is therefore not limited to the separately maintained serving tree.

At `f557b5440fe345757cae69ede2ed34826da5e612`, enabling two request slots turns
`--pipeline-windows` off even when only one request is running. This patch lets
an explicitly opted-in service keep the solo GEN pipeline while retaining batch
admission. Requests admitted to a batch do not use the solo pipeline verifiers.

The opt-in and slot-drafter placement are backported from
[Niko1221/Strata#3095](https://github.com/Niko1221/Strata/pull/3095).
This adaptation also excludes admissions and active batches after the diagnostic
pipeline overrides have been applied. The flag is off by default.

## Enable and roll back

Build this branch with your normal HIP build and keep an existing working model
configuration. In the Python server configuration, use `parallel: 2` and
`max_active_requests: 2`. Keep these engine arguments:

```text
--pipeline-windows 2 --batch-groups 1 --spec 4
--mtp /path/to/mtp --mtp-draft-vocab /path/to/rdna3/draft_vocab_zh60k.bin
```

Add `STRATA_PW_WITH_BATCH=1` to the configuration's `env` object. On gfx1100, this
test also explicitly sets `STRATA_MTP_STAGE0=1` and `STRATA_BATCH_MTP=1`.
The server adds `--batch 2` from `parallel`; direct engine users supply it themselves.
Use a layer split across two physical GPUs. Tensor-parallel decode disables the
window pipeline, so this is **not** an addition to the `--tensor-parallel 2` preset.

Remove `STRATA_PW_WITH_BATCH` or set it to `0`, then restart the isolated server to
restore the original batch gate. No weights, quantization, kernels, sampling math,
MTP acceptance rules, or default presets are changed by this patch.

## Validation on two RX 7900 XTX cards

| Mode | Chinese 1 client | Python 1 client | Chinese 2 clients total | Python 2 clients total |
|---|---:|---:|---:|---:|
| Unmodified base | 76.49 | 99.12 | 84.30 | 94.02 |
| Patched, flag off | 75.86 | 97.57 | 86.25 | 94.86 |
| Patched, flag on | 107.32 | 156.12 | 85.12 | 95.26 |

All values are tokens/s; medians of three requests or bursts.

Hardware: 2x Sapphire 7900 XTX 24GB (Platinum + BLOWER), 303/231W caps,
96 GiB installed RAM, Intel ES 14-core/20-thread AVX2 CPU, Ubuntu 26.04,
ROCm 10.1, HIP gfx1100 Release. Upstream links are PCIe 4.0 x16 and an
external CPU M.2 PCIe 3.0 x4 link. Layer split 26/22, 1536 MiB VRAM reserve,
int8 KV with 32768 resident tokens, prefill 16384, two 131072-token slots,
slot MTP enabled, batch groups 1, hardware queues 6, REFILL_SDMA disabled.

Checks: all short length/cache and long retrieval checks passed; candidate-on later stalled on the short JSON check. This is not ready for review.

Native fork server smoke: stock passed; candidate-on stalled on its first echo and failed the CLI. See the native-fork record.


The three-arm measurements used one shared, separately maintained Python serving
tree for all engines; its server hash is in the record. They isolate the engine
change but are not an untouched, end-to-end fork installation. A separate smoke
run uses the fork's own unmodified `serve/server.py` and the published CLI script;
those outcomes are recorded separately.

These are same-machine results for a batch-enabled **layer-split** service,
not a comparison against the repository author's best TP configuration. Aggregate
throughput is total completion tokens divided by elapsed wall time, including
prefill; it is not the sum of independently reported instantaneous decode rates.
The individual runs, timing fields, failures, checks and sanitized configuration are in
[the validation record](tests/results/pw-with-batch-gfx1100-20261114.json).

The short requests ask for 1024 output tokens with greedy sampling, seed 42,
reasoning off, zero presence/frequency penalties and prefix/session caches disabled.
Chinese operations prose and Python code each have three rounds with one and two
simultaneous clients. A warm-up precedes each arm. The order is stock, patched with
the flag off, patched with the flag on. Fixed ordering and a single server start
per arm limit claims about small differences.

Long retrieval requests use actual tokenizer-measured prompts of 8,192, 32,768
and 122,880 tokens, each once per arm, with a 256-token output limit. Configured
capacity is 131,072 tokens per slot; the long requests are solo, so this does not
establish two simultaneous 120K-request performance.

## Reproduce the checks

Start an isolated server on a spare local port, with no unrelated clients. Stop it
before loading the next arm on the same GPUs. Keep the same initial expert-profile
file for all arms. Compare the unmodified base, this branch with the flag off, and
this branch with the flag on. All other settings must stay identical.

```sh
python3 rdna3/tests/pw_batch_smoke.py \
  --url http://127.0.0.1:8101 \
  --model Qwen3.8-Flash-Next-IQ3_S \
  --benchmark --output candidate-on.json
```

The functional checks request an exact echo and a JSON object, exercise nonzero
presence/frequency penalties, and admit a second request only after the first has
started emitting tokens. They check overlap, distinct request markers and a fresh
solo request after both finish. The optional benchmark uses the same short prompts
as the validation record. This is a smoke test, not a model-quality score or proof
that every possible interleaving is safe. Its normal order does not replay the
full long-context failure sequence below.

## Observed failure

The first run used three rounds of short single/dual-client requests, then 8K,
32K and 122,880-token retrieval prompts, then a short exact echo and this JSON
request, with the same greedy/cache settings as above:

```text
Output only this JSON object, without Markdown: {"ok":true,"count":3}
```

The echo completed. The JSON request stopped making progress. After 60 seconds
the watchdog reported `verify window (pipelined): the CPU experts of layer 22`,
then exited with code -6. All 27 CPU-expert jobs were done and all eight workers
were sleeping; the host was at step 23, the GPU had rung 24, and served/plan/copy
flags were all 23. About 28 GiB RAM was available. The sanitized failure excerpt
is in the record.

The two off controls completed the full sequence. The cause of the on-path stalls
is **unconfirmed**; these observations do not separate the new combination, existing
pipeline/routing behavior and the machine's link errors. A
normal smoke pass does not clear this failure. It needs investigation before the
flag should be recommended.

## Limits and remaining checks

- This does not add MTP to pipelined batch groups. `--batch-groups 1` exercises
  the slot-drafter device placement; the fork's existing restrictions on groups
  greater than one remain.
- Repetition-penalty/coupled-sampling paths retain their existing serial fallback.
- Exact default-path output parity is not established. The small off-control
  functional outputs match, but only 1/18 fixed-length outputs matched across
  the stock/off arms. The unmodified stock also produced three distinct texts
  for each of its three repeated solo prompts. Adaptive expert placement was
  retained; a frozen-placement parity test is still needed. No full quality
  evaluation or quality improvement is claimed.
- Extra pipeline verifiers consume VRAM and can reduce expert-cache capacity;
  measure a representative workload before enabling the flag.
- Only HIP/gfx1100 is built and exercised here. CUDA and SYCL compilers are absent
  on the test machine. Their builds and hardware execution remain unverified;
  targeted validation is needed before this is ready for review.
- The machine has an asymmetric external PCIe link with a history of correctable
  AER events. The validation record includes counter changes. Results are local
  observations, not predictions for symmetric x8/x8 machines or other GPUs.
