#!/usr/bin/env python3
"""[claude] the two-card threshold rows of a --planner table (phase 3 Deploy A; B2 since 2026-11-21).
usage: make_planner_rows2.py INPUTS.json QUANT [--json] [--engine-derives] > data/planner/qwen-<quant>-2gpu.txt

INPUTS.json, B2 form (results/planner-20261112/b2/review-fable-structural.md, "Decision 1" to "Decision 3"):
  {"measured": {"STRATA_KEY": {"value": N, "source": "..."}},   class A text-measurable (TP_PREFILL_MIN / MAX), class B once measured
   "copied":   {"STRATA_KEY": {"value": N, "why": "..."}},      class A declined-only (DP_MIN), class B until measured, IQ3 DP_PIPE_CHUNK;
                                                                a class A text row (TP_PREFILL_MIN / MAX) only with "pending": "<why the
                                                                measured value is not used yet>" (B2 11-25: TP_PREFILL_MAX, session 1
                                                                MOVE-EDGE, bracket extension pending; classA/review-fable-s1.md 2)
   "policy":   {"STRATA_KEY": N}}                               class C refill rows
  or the Deploy A form: a flat launcher environment {"STRATA_KEY": "value", ...} (every non-derived key copied).
Every row is classified in its comment: derived / measured / copied / policy.  Two rows are never inputs; they are derived
(B2 structural pass, structural-pass.md: identical plans in every text and declined state, any cap):
  STRATA_TP_PREFILL_CHUNK = ceil256(STRATA_TP_PREFILL_MAX)   asserted >= ceil256(TP_PREFILL_MAX); no row when it equals the
                                                             engine default 16384 (generate.cpp:210; IQ3)
  STRATA_DP_PIPE_CHUNK    = ceil1024(max(8448, STRATA_DP_MIN) + 2)
                                                             only where a DP_MIN row exists; asserted >= ceil256(DP_MIN + 2);
                                                             without a DP_MIN row DP_PIPE_CHUNK is a copied input (IQ3 8448)
                                                             (the engine's planner_rows::derive uses the same formula,
                                                             results/exitfix-20261121/review-fable-impl.md required change 2)
A flat Deploy A environment may still carry the two derived keys: their values must satisfy the same bound (plan-identical
to the derived value), else the generator fails.  --json prints the classified rows as JSON instead of the table.
--engine-derives: the table is for an engine whose --planner fill derives the two rows itself (planner_rows::derive, Strata
exitfix-20261121 7a9bf46c4 / 9345506e1, release claude-20261122-exitfix and later): the derived rows are left out (one comment
line each) because the engine computes the same value.  NEVER use it for claude-20261121-headp1 or older (no derive rule: IQ4 / Q4
would fall back to the parse-site defaults 16384 / 8192 and change plans).  Without the flag the derived rows are written; that
table is value-identical on both engines (with the rows set, the engine's derive is dormant)."""
import json, sys

ALLOWED = ['STRATA_TP_PREFILL_MIN', 'STRATA_TP_PREFILL_MAX', 'STRATA_TP_PREFILL_CHUNK', 'STRATA_DP_MIN', 'STRATA_DP_PIPE_CHUNK',
           'STRATA_PIPE_HALF_MIN', 'STRATA_PIPE_HALF_MAX', 'STRATA_PREFILL_EQUAL_TAIL_MAX', 'STRATA_PREFILL_AUTO_MAX',
           'STRATA_TP_DEAL_BUDGET', 'STRATA_FUSED_ROUTED_MAX', 'STRATA_REFILL_HEAD', 'STRATA_REFILL_HEAD_MIN', 'STRATA_REFILL_HEAD2',
           'STRATA_REFILL_HEAD2_MIN', 'STRATA_REFILL_HEAD_BIG', 'STRATA_REFILL_HEAD_BIG_CHUNK', 'STRATA_REFILL_HEAD_TP_KEEP',
           'STRATA_REFILL_HEAD_TP_MIN']
TEXT_A = {'STRATA_TP_PREFILL_MIN', 'STRATA_TP_PREFILL_MAX'}            # class A, measurable by a text tool run
DECLINED_A = {'STRATA_DP_MIN'}                                          # class A, read only when run_tp declines: copied
POLICY = {k for k in ALLOWED if k.startswith('STRATA_REFILL_')}         # class C
DERIVED = {'STRATA_TP_PREFILL_CHUNK', 'STRATA_DP_PIPE_CHUNK'}
CLASS_B = set(ALLOWED) - TEXT_A - DECLINED_A - POLICY - DERIVED
TP_CHUNK_DEFAULT = 16384                                                # generate.cpp:210, segment_plan.hpp:33
DP_MIN_FLOOR = 2 * 4224                                                 # an explicit DP_MIN is floored at 2 x kDpMinLast


def ceil_to(v, g): return -(-v // g) * g


def fail(msg): sys.exit(f'make_planner_rows2: {msg}')


def check(ok, msg):   # an assert that also holds under python -O
    if not ok: fail('assert: ' + msg)


def classify(inp):
    """{key: (value, kind, class, note)} for the input rows (no derived key yet)"""
    rows = {}
    if any(k in inp for k in ('measured', 'copied', 'policy')):
        for kind in ('measured', 'copied', 'policy'):
            for k, v in (inp.get(kind) or {}).items():
                if k not in ALLOWED: fail(f'{k} is not a threshold key a row may set')
                if k in rows: fail(f'{k} appears twice')
                val = v if not isinstance(v, dict) else v['value']
                note = '' if not isinstance(v, dict) else (v.get('source') or v.get('why') or '')
                cls = 'A' if k in TEXT_A | DECLINED_A else 'C' if k in POLICY else 'B'
                if k in DERIVED and not (k == 'STRATA_DP_PIPE_CHUNK' and kind == 'copied'): fail(f'{k} is derived, never an input')
                if kind == 'measured' and k not in TEXT_A | CLASS_B: fail(f'{k} cannot be a measured row (declined-only, derived or policy)')
                pending = v.get('pending', '') if isinstance(v, dict) else ''
                if kind == 'copied' and k in POLICY: fail(f'{k} cannot be a copied row in B2 (measured or policy)')
                if kind == 'copied' and k in TEXT_A and not pending:
                    fail(f'{k} cannot be a copied row in B2 without a "pending" note (measured or policy)')
                if pending and not (kind == 'copied' and k in TEXT_A): fail(f'{k}: "pending" is only for a copied class A text row')
                if pending: note = (note + '; ' if note else '') + pending
                if kind == 'policy' and k not in POLICY: fail(f'{k} is not a refill policy row')
                rows[k] = (int(val), kind, cls, note)
        if 'STRATA_DP_PIPE_CHUNK' in rows and 'STRATA_DP_MIN' in rows:
            fail('DP_PIPE_CHUNK is derived where a DP_MIN row exists; it is a copied input only without one')
        return rows, {}
    legacy = {}   # Deploy A: a flat environment
    for k in ALLOWED:
        if inp.get(k) in (None, ''): continue
        v = int(inp[k])
        if k in DERIVED and not (k == 'STRATA_DP_PIPE_CHUNK' and inp.get('STRATA_DP_MIN') in (None, '')):
            legacy[k] = v; continue
        cls = 'A' if k in TEXT_A | DECLINED_A else 'C' if k in POLICY else 'B'
        rows[k] = (v, 'policy' if cls == 'C' else 'copied', cls, 'Deploy A environment')
    return rows, legacy


def derive(rows, legacy, engine_derives=False):
    out = dict(rows)
    if 'STRATA_TP_PREFILL_MAX' not in rows: fail('no STRATA_TP_PREFILL_MAX row: TP_PREFILL_CHUNK cannot be derived')
    tp_max = rows['STRATA_TP_PREFILL_MAX'][0]
    ch = ceil_to(tp_max, 256)
    check(ch >= ceil_to(tp_max, 256), 'TP_PREFILL_CHUNK below ceil256(TP_PREFILL_MAX): plans change')
    if 'STRATA_TP_PREFILL_CHUNK' in legacy:
        check(legacy['STRATA_TP_PREFILL_CHUNK'] >= ceil_to(tp_max, 256), \
            f"input TP_PREFILL_CHUNK {legacy['STRATA_TP_PREFILL_CHUNK']} < ceil256(TP_PREFILL_MAX): not plan-identical to the derived row")
    out['STRATA_TP_PREFILL_CHUNK'] = (ch, 'derived', 'B', 'ceil256(STRATA_TP_PREFILL_MAX)' +
                                      (' = engine default: no row' if ch == TP_CHUNK_DEFAULT else
                                       ' = the engine fill derives it: no row' if engine_derives else ''))
    if 'STRATA_DP_MIN' in rows:
        dp_min = max(DP_MIN_FLOOR, rows['STRATA_DP_MIN'][0])
        pc = ceil_to(dp_min + 2, 1024)   # = planner_rows::derive (the bound holds by construction; the check stays)
        check(pc >= ceil_to(dp_min + 2, 256), 'DP_PIPE_CHUNK below ceil256(DP_MIN + 2): plans change')
        if 'STRATA_DP_PIPE_CHUNK' in legacy:
            check(legacy['STRATA_DP_PIPE_CHUNK'] >= ceil_to(dp_min + 2, 256), \
                f"input DP_PIPE_CHUNK {legacy['STRATA_DP_PIPE_CHUNK']} < ceil256(DP_MIN + 2): not plan-identical to the derived row")
        out['STRATA_DP_PIPE_CHUNK'] = (pc, 'derived', 'B', 'ceil1024(max(8448, STRATA_DP_MIN) + 2)' +
                                       (' = the engine fill derives it: no row' if engine_derives else ''))
    return out


def main():
    args = [a for a in sys.argv[1:] if a not in ('--json', '--engine-derives')]
    if len(args) != 2: fail('usage: make_planner_rows2.py INPUTS.json QUANT [--json] [--engine-derives]')
    q = args[1]
    rows = derive(*classify(json.load(open(args[0]))), engine_derives='--engine-derives' in sys.argv)
    table = [(k, *rows[k]) for k in ALLOWED if k in rows and not (rows[k][1] == 'derived' and rows[k][3].endswith('no row'))]
    if '--json' in sys.argv:
        print(json.dumps({'quant': q, 'rows': [dict(key=k, value=v, kind=kind, cls=c, note=n) for k, v, kind, c, n in table],
                          'omitted': [dict(key=k, value=rows[k][0], why=rows[k][3]) for k in ALLOWED if k in rows and rows[k][3].endswith('no row')]}, indent=1))
        return
    count = {x: sum(1 for r in table if r[2] == x) for x in ('derived', 'measured', 'copied', 'policy')}
    print(f'# two-card threshold rows, {q} (B2: tool-generated; make_planner_rows2.py' +
          (' --engine-derives: for an engine with the planner_rows::derive fill only, claude-20261122-exitfix or later)' if '--engine-derives' in sys.argv
           else '; derived rows written: value-identical on claude-20261121-headp1 and on engines with the derive fill)'))
    print('# class A = path crossover, B = geometry, C = refill policy; kind: derived (formula) / measured (tool run) / copied / policy')
    print(f'# {len(table)} rows: ' + ', '.join(f'{v} {k}' for k, v in count.items()))
    for k in ALLOWED:
        if k in rows and rows[k][1] == 'derived' and rows[k][3].endswith('no row'):
            print(f'# {k} {rows[k][0]} {rows[k][3]}')
    for k, v, kind, c, n in table:
        print(f'row2 {k} {v}   # class {c} {kind}' + (f': {n}' if n else ''))


if __name__ == '__main__':
    main()
