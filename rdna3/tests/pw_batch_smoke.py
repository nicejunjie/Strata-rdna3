#!/usr/bin/env python3
"""Exercise solo GEN -> overlapping admission -> solo GEN on a batch service.

Start an isolated two-slot server first. This sends real inference requests;
it does not start, stop, or reconfigure a server. Standard library only.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import threading
import time
import urllib.request


def request(url, model, prompt, max_tokens=128, first_event=None, **sampling):
    body = dict(model=model, messages=[dict(role="user", content=prompt)],
                max_tokens=max_tokens, temperature=0, top_k=1, top_p=1,
                seed=42, presence_penalty=0, frequency_penalty=0,
                cache_prompt=False, reasoning_effort="none",
                chat_template_kwargs={"enable_thinking": False}, stream=True,
                stream_options={"include_usage": True})
    body.update(sampling)
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    first, finish, usage, timings, chunks = None, None, {}, {}, []
    with urllib.request.urlopen(req, timeout=240) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                continue
            data = json.loads(raw)
            if data.get("error"):
                raise RuntimeError(data["error"])
            usage = data.get("usage") or usage
            timings = data.get("timings") or timings
            for choice in data.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("content") or delta.get("reasoning_content") or ""
                if text:
                    if first is None:
                        first = time.perf_counter()
                    chunks.append(text)
                    if first_event is not None:
                        first_event.set()
                finish = choice.get("finish_reason") or finish
    end = time.perf_counter()
    if first is None or not usage.get("completion_tokens", 0):
        raise RuntimeError("No generated text or completion-token usage")
    return dict(start=start, first=first, end=end, ttft_s=first-start,
                wall_s=end-start, text="".join(chunks), usage=usage,
                timings=timings, finish=finish)


def run(url, model):
    def req(prompt, **kwargs):
        return request(url, model, prompt, **kwargs)

    checks = []
    echo = req("Only output the exact text CHECK_GAMMA_49. Do not add punctuation or explanation.")
    echo.update(case="echo", valid=echo["text"].strip() == "CHECK_GAMMA_49")
    checks.append(echo)
    obj = req('Output only this JSON object, without Markdown: {"ok":true,"count":3}')
    try:
        valid = json.loads(obj["text"]) == {"ok": True, "count": 3}
    except ValueError:
        valid = False
    obj.update(case="json", valid=valid)
    checks.append(obj)
    penalty = req("List the integers 1 to 30 in ascending order, separated by commas. Do not add explanation.",
                  max_tokens=256, presence_penalty=0.5, frequency_penalty=0.1)
    import re
    penalty.update(case="nonzero-penalties",
                   valid=[int(v) for v in re.findall(r"\d+", penalty["text"])] == list(range(1, 31)))
    checks.append(penalty)

    first = threading.Event()
    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(req, "Write a long numbered list of practical Linux command-line tips. "
                        "Begin with CHECK_ALPHA_17 and continue until the token limit.",
                        max_tokens=1024, first_event=first)
        if not first.wait(60):
            raise RuntimeError("Solo GEN did not emit before the admission timeout")
        b = pool.submit(req, "Only output the exact text CHECK_BETA_28. Do not add punctuation or explanation.")
        vb, va = b.result(timeout=180), a.result(timeout=180)
    vc = req("Only output the exact text CHECK_GAMMA_49. Do not add punctuation or explanation.")
    overlap = max(0, min(va["end"], vb["end"]) - max(va["first"], vb["first"]))
    request_overlap = max(0, min(va["end"], vb["end"]) - max(va["first"], vb["start"]))
    transition = dict(overlap_s=overlap, request_overlap_s=request_overlap, a=va, b=vb, after=vc,
                      valid=(overlap > 0 and "CHECK_ALPHA_17" in va["text"]
                             and "CHECK_BETA_28" not in va["text"]
                             and vb["text"].strip() == "CHECK_BETA_28"
                             and vc["text"].strip() == "CHECK_GAMMA_49"))
    return dict(checks=checks, transition=transition,
                passed=all(v["valid"] for v in checks) and transition["valid"])


def benchmark(url, model, repeats=3):
    """Same cold-prefix, fixed-output requests used for the attached short runs."""
    prompts = {
        "zh": "请编写一份完整、详细的本地大模型运维手册，必须逐项写满120个编号条目，每项包含一条具体操作和验证方法，依次涵盖模型文件校验、显存、KV缓存、并发、批处理、PCIe、内存、日志、容器、API、备份、监控与故障恢复。不要省略条目，不要提前总结。",
        "code": "Write a complete production Python HTTP job service of at least 500 lines using only the standard library. Include dataclasses, persistence with SQLite, retry/backoff, thread workers, bounded queues, cancellation, deadlines, CLI, JSON HTTP routes, metrics and unittest tests. Output the full code without abbreviations or explanation.",
    }
    warmup = request(url, model, prompts["zh"], max_tokens=256)
    bursts = []
    for rep in range(repeats):
        for kind in (["code", "zh"] if rep % 2 == 0 else ["zh", "code"]):
            for n in [1, 2]:
                barrier = threading.Barrier(n)

                def lane(i):
                    barrier.wait(timeout=15)
                    return request(url, model, f"任务编号{i}。\n" + prompts[kind], max_tokens=1024)

                with ThreadPoolExecutor(max_workers=n) as pool:
                    rows = list(pool.map(lane, range(n)))
                wall = max(v["end"] for v in rows) - min(v["start"] for v in rows)
                valid = all(v["usage"]["completion_tokens"] == 1024 and
                            v["timings"].get("cache_n", v["usage"].get("prompt_tokens_details", {}).get("cached_tokens", 0)) == 0
                            for v in rows)
                bursts.append(dict(kind=kind, n=n, repeat=rep, rows=rows,
                                   aggregate_tps=sum(v["usage"]["completion_tokens"] for v in rows)/wall,
                                   wall_s=wall, valid=valid))
    return dict(warmup=warmup, bursts=bursts, passed=all(v["valid"] for v in bursts))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8101")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--benchmark", action="store_true", help="Also send 3 rounds of 1/2-client throughput requests")
    parser.add_argument("--repeats", type=int, default=3, help="Benchmark rounds (default: 3)")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    result = run(args.url, args.model)
    if args.benchmark:
        result["benchmark"] = benchmark(args.url, args.model, repeats=args.repeats)
        result["passed"] = result["passed"] and result["benchmark"]["passed"]
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"passed": result["passed"],
                      "checks": {v["case"]: v["valid"] for v in result["checks"]},
                      "overlapping_admission": result["transition"]["valid"]}))
    raise SystemExit(0 if result["passed"] else 1)
