#!/usr/bin/env python3
"""Measure one RDNA3 setup the way the README's table and charts were measured, and compare with them.

    .venv/bin/python rdna3/bench.py iq4-2gpu --data ../Strata-data --gpus 1,0            # 1K ... 256K
    .venv/bin/python rdna3/bench.py q4-1gpu --data ../Strata-data --lengths 8192,32768

Protocol (rdna3/README.md): the setup's server (rdna3/run.py) on a private port; one warm-up; for every input length
three prompts of EXACTLY that many tokens (the model's own tokenizer and chat template) holding three secret codes at
the start, middle and end, followed by a request for a long answer; 256 output tokens, temperature 0, seed 42, no
prompt cache, thinking off.  Each request must report exactly n prompt tokens (none cached) and 256 completion tokens,
and must return the three codes (retrieval).  Speeds are the engine's own timings (prompt_per_second,
predicted_per_second); a cell is the median of the three.  Results: <out>/<setup>/, summary.json, and a table against
rdna3/reference.json (production, 2026-11-02).
"""
import argparse
import hashlib
import json
import os
import signal
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT / "rdna3"))
import run as rdna3_run                                   # noqa: E402
LENGTHS = [1024 * 2 ** i for i in range(8)] + [261632]   # 1K ... 128K, and 256K less 512 tokens of headroom
CODES = ["ALPHA-7391", "BRAVO-2846", "CHARLIE-6158"]
WARMUP = ("Explain how a refrigerator works. Cover the compressor, condenser, expansion valve and evaporator, "
          "thermodynamics, common faults and efficiency. Use at least 1500 words.")


def encoder(cfg):
    """The server's own prompt encoding (tokenizer + chat template), without starting an engine."""
    from serve.server import Service, literal_tags
    from serve.frontend import ChatTemplate
    import strata_tokenizer as ST
    path = Path(cfg["tokenizer"])
    vocab = json.loads((path / "vocab.json").read_text())
    tokens = [None] * len(vocab)
    for token, index in vocab.items():
        tokens[index] = token
    service = Service.__new__(Service)
    service.tok = ST.Tokenizer(tokens, (path / "merges.txt").read_text().split("\n"),
                               json.loads((path / "token_type.json").read_text()))
    service.template = ChatTemplate(path / "chat_template.jinja")
    service.effort_end = False
    service.literals = literal_tags(getattr(service.tok, "control_tokens", ()))
    service.prompts = None                                # no kept-prompt reuse: every count is a full encode
    return service


def payload(model, text, n):
    return {"model": model, "messages": [{"role": "user", "content": text}], "max_tokens": n, "temperature": 0,
            "seed": 42, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}, "stream": True,
            "stream_options": {"include_usage": True}}


def exact_prompt(service, model, target, repeat):
    """A prompt of exactly `target` tokens: records (three with a secret code), padding, then the request."""
    from serve.frontend import openai_to_messages

    def count(text):
        return len(service.encode_prompt(*openai_to_messages(payload(model, text, 256))))

    def build(records, padding=0):
        lines = [f"Record {i}: warehouse cedar has blue crates and green labels; inventory is checked daily."
                 for i in range(records)]
        for i, code in zip([0, records // 2, records - 1], CODES):
            lines[i] += " Secret code: " + code + "."
        return (f"Benchmark trial {repeat}. Read all records.\n" + "\n".join(lines) + "\nPadding:" + " x" * padding
                + "\nReturn the three secret codes in order, then write a detailed inventory auditing guide of at "
                  "least 1500 words. Do not stop after listing the codes.")
    lo, hi = 3, max(4, target // 10)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(build(mid)) <= target:
            lo = mid
        else:
            hi = mid - 1
    text = build(lo, target - count(build(lo)))
    if count(text) != target:
        raise RuntimeError(f"could not build a {target}-token prompt")
    return text


def request(port, body):
    """One streamed chat request: the text, usage, the engine's timings, time to first token."""
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    started, first, content, usage, timings = time.monotonic(), None, [], {}, {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for raw in r:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            e = json.loads(data)
            if "error" in e:
                raise RuntimeError(e)
            timings, usage = e.get("timings", timings), e.get("usage") or usage
            for choice in e.get("choices", []):
                delta = choice.get("delta", {})
                if (delta.get("content") or delta.get("reasoning_content")) and first is None:
                    first = time.monotonic() - started
                content.append(delta.get("content") or "")
    return {"content": "".join(content), "usage": usage, "timings": timings, "ttft_s": first,
            "wall_s": time.monotonic() - started}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("setup", choices=rdna3_run.SETUPS)
    ap.add_argument("--data", required=True, help="setup.py's data folder")
    ap.add_argument("--gpus", help="as rdna3/run.py (the README used 1 and 1,0)")
    ap.add_argument("--lengths", help="comma-separated input lengths (default 1024 ... 131072 and 261632)")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--port", type=int, default=18087)
    ap.add_argument("--out", default=str(ROOT / "rdna3" / "results"))
    ap.add_argument("--load-timeout", type=int, default=1800)
    a = ap.parse_args()
    lengths = [int(x) for x in a.lengths.split(",")] if a.lengths else LENGTHS
    cfg = rdna3_run.config(a.setup, a.data, a.gpus, vision=True)
    out = Path(a.out).resolve() / a.setup
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(cfg, indent=1) + "\n")
    exe = Path(cfg["exe"])
    meta = {"setup": a.setup, "lengths": lengths, "repeats": a.repeats, "gpus": cfg["gpu"],
            "engine_sha256": hashlib.sha256(exe.read_bytes()).hexdigest(),
            "git": subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], capture_output=True,
                                  text=True).stdout.strip()}
    (out / "meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"{a.setup}: building the exact prompts ...", flush=True)
    service = encoder(cfg)
    prompts = {(n, r): exact_prompt(service, cfg["model_name"], n, r) for n in lengths for r in range(a.repeats)}
    argv = [sys.executable, str(ROOT / "rdna3" / "run.py"), a.setup, "--data", a.data, "--port", str(a.port),
            "--log", str(out / "engine.log")] + (["--gpus", a.gpus] if a.gpus else [])
    started = time.monotonic()
    with (out / "server.log").open("w") as log:
        proc = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    rows, failed = [], []
    try:
        while True:
            if proc.poll() is not None:
                raise RuntimeError(f"the server exited while loading (see {out / 'server.log'})")
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{a.port}/v1/models", timeout=5).read()
                break
            except OSError:
                if time.monotonic() - started > a.load_timeout:
                    raise TimeoutError("the server did not come up")
                time.sleep(1)
        meta["ready_s"] = round(time.monotonic() - started, 1)
        print(f"{a.setup}: ready after {meta['ready_s']} s", flush=True)
        request(a.port, payload(cfg["model_name"], WARMUP, 128))
        for n in lengths:
            for r in range(a.repeats):
                v = request(a.port, payload(cfg["model_name"], prompts[(n, r)], 256))
                t, u = v["timings"], v["usage"]
                v.update(name=f"n{n}-r{r}", target=n,
                         exact_input_pass=u.get("prompt_tokens") == n and t.get("prompt_n") == n and not t.get("cache_n"),
                         output_length_pass=u.get("completion_tokens") == 256,
                         retrieval_pass=all(code in v["content"] for code in CODES))
                (out / f"n{n}-r{r}.json").write_text(json.dumps(v, indent=1, ensure_ascii=False) + "\n")
                rows.append(v)
                ok = v["exact_input_pass"] and v["output_length_pass"] and v["retrieval_pass"]
                if not ok:
                    failed.append(v["name"])
                print(f"  {v['name']:>13}  prefill {t.get('prompt_per_second', 0):7.0f}  decode "
                      f"{t.get('predicted_per_second', 0):6.1f}  {'pass' if ok else 'FAIL'}", flush=True)
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    ref = json.loads((ROOT / "rdna3" / "reference.json").read_text()).get(a.setup, {})
    cells = {}
    print(f"\n{a.setup}  tokens/s, median of {a.repeats} (reference: rdna3/reference.json)")
    print(f"{'input':>8}  {'prefill':>8} {'ref':>7} {'ratio':>6}   {'decode':>7} {'ref':>6} {'ratio':>6}")
    for n in lengths:
        ts = [v["timings"] for v in rows if v["target"] == n]
        if not ts:
            continue
        pp = statistics.median(t["prompt_per_second"] for t in ts)
        dec = statistics.median(t["predicted_per_second"] for t in ts)
        cells[str(n)] = {"prefill": pp, "decode": dec, "decode_min": min(t["predicted_per_second"] for t in ts),
                         "decode_max": max(t["predicted_per_second"] for t in ts)}
        r = ref.get(str(n))
        print(f"{n:>8}  {pp:8.0f} {r['prefill'] if r else 0:7.0f} {pp / r['prefill'] if r else 0:6.2f}   "
              f"{dec:7.1f} {r['decode'] if r else 0:6.1f} {dec / r['decode'] if r else 0:6.2f}")
    meta.update(cells=cells, failed=failed, passed=not failed and len(rows) == len(lengths) * a.repeats)
    (out / "summary.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(("all " + str(len(rows)) + " requests passed") if meta["passed"] else f"FAILED: {failed}")
    return 0 if meta["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
