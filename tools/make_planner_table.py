"""tools/make_planner_table.py - the one-card planner's table (`data/planner/qwen-<quant>.txt` + `.counts`), read by `--planner`.

`--planner TABLE` derives the one-card staged limit X at start-up (option C, the frozen rule in
include/strata/program/planner_c.hpp): it times 8 canned English 2048-token start reads, corrects this table's staged
remainder by the difference, and finds per text (English, code, Chinese) the last length where a staged read is
predicted faster than streaming every expert.  What the rule needs that is not a start-up fact is in this table:

  per text and builder length T  the per-layer per-expert routed-token counts of every staged read of that length
                                 (as 2-bit classes: not routed / 1..maxt tokens / more - all the rule reads of a count),
                                 and the builder's part times (ms)
  win                            the decode windows' ms (the median of the builder's "windows (short read <= 64)" lines)
  stream_compute A B, o_s        the stream-all compute shape (ms = A + B * tokens) and offset: a developer fit of
                                 separate short-read probes (phase timings); passed in, not measured here
  probe_tokens                   the canned start prompts, token ids exactly as the server sends them (chat template
                                 included), one line per variant
  const                          the rule's constants (grid, margin, guard, the crossing search lo/hi/step, ...)
  profile                        the FNV-1a 64 hash of the expert profile (--expert-profile) the builder run loaded: the
                                 rule's residency and the staged times come from its ranking, so the engine refuses the
                                 table under any other profile (a new profile means a new builder run and table)
  ref_start                      (optional, --ref-start MS) the reference start read in the ENGINE's own statistic: the mean
                                 of the start reads (the median of reads 4-8 of `--planner`'s 8 canned reads) over validated
                                 cold engine derivations on this host and table.  The rule at this value on a load's facts is
                                 level 3 (key absent, no X this load) and the bracket's X_ref.  Without the line the engine
                                 uses the builder's English median at start - 7, a different statistic (the builder's staged
                                 reads, about 4-16 ms off the engine's).  The shipped values (1657.1 / 1885.7 / 2426.3 ms for
                                 IQ3_S / IQ4_XS / Q4_K_XL) are the means of 20 cold `--planner` loads per quant on build
                                 build-hip-apply (results/planner-20261112/next/apply20/tally.txt, keys present, 2026-11-13).

Only the lengths the rule reads are kept: the start read's part (start - 7) and the rungs that bracket the crossing
search [lo, hi) - the rule interpolates between neighbouring rungs, so the others never enter X.

The builder run is one engine start (one card, the model's production config with `--calibration-plans`, the CPU
share's key unset) that reads each text at the ladder lengths, every read STAGED (PLAN_STAGED_MAX 16641), prefix 0,
with `--dump-prompt-counts COUNTS.bin`; its engine log has the plan lines (`strata plan: part a-b ... | N ms`).  The
reads must come in text order, `--reads-per-text` staged reads each (the shipped tables: 11 lengths x 4 = 44).

    python tools/make_planner_table.py --log ENGINE.LOG --counts COUNTS.bin --quant iq4 --engine-pack unsloth-ud-iq4_xs \\
        --stream-compute A B --o-s X --probe-tokens PROBES.txt --profile data/expert-profile.bin \\
        --out data/planner/qwen-iq4.txt

writes OUT and OUT with `.txt` replaced by `.counts`.  PROBES.txt: one line per variant, comma-separated ids.
"""
import argparse
import re
import statistics
import struct
from pathlib import Path

CONST = dict(layers=48, experts=512, maxt=8, share=0.6, grid=256, margin=128, guard=1.6, conv=0.003, max_reads=8,
             start=2048, lo=3328, hi=12288, step=16, start_limit=16641)


def read_counts(path, layers, experts):
    """`--dump-prompt-counts` records (int32 layer, int32 T, int32 n_expert, n_expert int32 counts) -> chunks [(T, counts)];
    a new chunk at layer 0 or a new T, kept only when every layer has a routed expert (as the engine loads them)"""
    blob = Path(path).read_bytes()
    off, chunks = 0, []
    while off + 12 <= len(blob):
        layer, T, ne = struct.unpack_from("<3i", blob, off)
        off += 12
        if ne != experts or not 0 <= layer < layers:
            raise SystemExit(f"{path}: a record of layer {layer} with {ne} experts does not fit {layers}x{experts}")
        row = struct.unpack_from(f"<{ne}i", blob, off)
        off += 4 * ne
        if layer == 0 or not chunks or chunks[-1][0] != T:
            chunks.append((T, [[0] * experts for _ in range(layers)]))
        chunks[-1][1][layer] = list(row)
    return [c for c in chunks if all(any(v != 0 for v in r) for r in c[1])]


def plan_rows(log):
    """the staged parts of the builder's reads, in order: (part tokens, ms)"""
    out = []
    for m in re.finditer(r"strata plan: part (\d+)-(\d+) \w+ .*? kernels: chunks \d+ stream_all (\d+) staged (\d+) .*?\| ([\d.]+) ms",
                         Path(log).read_text(errors="replace")):
        a, b = int(m.group(1)), int(m.group(2))
        if b - a > 64 and int(m.group(4)):
            out.append((b - a, float(m.group(5))))
    return out


def window_ms(log):
    w = [float(m.group(3)) for m in re.finditer(r"strata plan: part (\d+)-(\d+) windows \(short read <= 64\) ([\d.]+) ms",
                                                Path(log).read_text(errors="replace")) if int(m.group(1)) > 900]
    if not w:
        raise SystemExit(f"{log}: no decode-window plan lines")
    return statistics.median(w)


def kept_lengths(Ts, c):
    """the rungs the rule reads: start - 7 and the ones bracketing every crossing-search length in [lo, hi)"""
    Ts = sorted(Ts)
    cs = list(range(c["lo"], c["hi"], c["step"]))
    lo_b = max([T for T in Ts if T <= cs[0]], default=Ts[0])
    hi_b = min([T for T in Ts if T >= cs[-1]], default=Ts[-1])
    keep = [T for T in Ts if lo_b <= T <= hi_b or T == c["start"] - 7]
    if c["start"] - 7 not in keep:
        raise SystemExit(f"no builder reads at the start read's part ({c['start'] - 7} tokens)")
    return keep


def fnv1a64(path):
    """the engine's planner_c::file_hash: FNV-1a 64 over the file's bytes"""
    h = 1469598103934665603
    for b in Path(path).read_bytes():
        h = ((h ^ b) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def classes(counts, maxt):
    """2-bit classes, 4 a byte, low bits first: 0 not routed, 1 routed by 1..maxt tokens, 2 by more"""
    flat = [0 if v <= 0 else 1 if v <= maxt else 2 for row in counts for v in row]
    flat += [0] * (-len(flat) % 4)
    return bytes(flat[i] | flat[i + 1] << 2 | flat[i + 2] << 4 | flat[i + 3] << 6 for i in range(0, len(flat), 4))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", required=True, help="the builder run's engine log")
    ap.add_argument("--counts", required=True, help="its --dump-prompt-counts file")
    ap.add_argument("--quant", required=True)
    ap.add_argument("--engine-pack", required=True, help="the --pack directory name the table belongs to (checked at load)")
    ap.add_argument("--stream-compute", nargs=2, type=float, required=True, metavar=("A", "B"))
    ap.add_argument("--o-s", type=float, required=True)
    ap.add_argument("--probe-tokens", required=True, help="one line per start-prompt variant: comma-separated token ids")
    ap.add_argument("--profile", required=True, help="the --expert-profile file the builder run loaded (its hash goes in the table)")
    ap.add_argument("--texts", type=int, default=3, help="texts in the builder run, in order (english, code, chinese)")
    ap.add_argument("--reads-per-text", type=int, default=44)
    ap.add_argument("--ref-start", type=float, default=None, metavar="MS",
                    help="the reference start read in the engine's own statistic (ms): the mean start of validated cold --planner loads")
    ap.add_argument("--out", required=True, help="the table (.txt); the counts go next to it as .counts")
    a = ap.parse_args()
    if a.ref_start is not None and not (0 < a.ref_start < 1e6):
        raise SystemExit(f"--ref-start {a.ref_start}: not a start read in ms")
    c = dict(CONST)
    chunks = read_counts(a.counts, c["layers"], c["experts"])
    rows = plan_rows(a.log)
    n = a.reads_per_text
    if len(chunks) != a.texts * n or len(rows) != a.texts * n:
        raise SystemExit(f"{len(chunks)} counted chunks and {len(rows)} staged plan lines; {a.texts} texts x {n} expected")
    for i, ((T, _), (part, _)) in enumerate(zip(chunks, rows)):   # review 15 S5: pair by position, and check every pair
        if abs(part - T) >= 16:
            raise SystemExit(f"read {i}: the counted chunk has {T} tokens, its staged plan line a {part}-token part")
    lines_text, out_chunks = [], []
    for t in range(a.texts):
        idx = range(n * t, n * (t + 1))
        Ts = sorted({chunks[i][0] for i in idx})
        for T in kept_lengths(Ts, c):
            ids = []
            for i in idx:
                if chunks[i][0] == T:
                    ids.append(len(out_chunks))
                    out_chunks.append(chunks[i])
            ms = [m for part, m in rows[n * t: n * (t + 1)] if abs(part - T) < 16]
            if len(ms) != len(ids):
                raise SystemExit(f"text {t} length {T}: {len(ids)} counted chunks but {len(ms)} part times")
            lines_text.append(f"text {t} T {T} chunks {' '.join(map(str, ids))} ms {' '.join(repr(float(x)) for x in ms)}")
    probes = [ln.strip() for ln in Path(a.probe_tokens).read_text().splitlines() if ln.strip()]
    for p in probes:
        if len(p.split(",")) != c["start"]:
            raise SystemExit(f"a probe variant has {len(p.split(','))} tokens, not {c['start']}")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cpath = out.with_suffix(".counts")
    L = [f"quant {a.quant}", f"engine_pack {a.engine_pack}", f"profile {fnv1a64(a.profile):016x}", f"counts_file {cpath.name}", f"n_chunks {len(out_chunks)}",
         f"win {repr(float(window_ms(a.log)))}", f"stream_compute {repr(a.stream_compute[0])} {repr(a.stream_compute[1])}",
         f"o_s {repr(a.o_s)}"] + ([f"ref_start {repr(float(a.ref_start))}"] if a.ref_start is not None else []) + ["const " + " ".join(f"{k} {v}" for k, v in c.items())]
    L += [f"probe_tokens {v} {p}" for v, p in enumerate(probes)]
    out.write_text("\n".join(L + lines_text) + "\n")
    body = b"".join(struct.pack("<i", T) + classes(cnt, c["maxt"]) for T, cnt in out_chunks)
    cpath.write_bytes(b"SPC2" + struct.pack("<5I", 1, c["layers"], c["experts"], c["maxt"], len(out_chunks)) + body)
    print(f"{out}: {len(out_chunks)} chunks, {sum(1 for _ in lines_text)} text rungs; {cpath}: {cpath.stat().st_size} bytes")


if __name__ == "__main__":
    main()
