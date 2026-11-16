#!/usr/bin/env python3
"""[claude] phase 3 Deploy A: the two-card threshold rows of a --planner table from a launcher's resolved environment.
usage: make_planner_rows2.py ENV.json QUANT > data/planner/qwen-<quant>-2gpu.txt
ENV.json: {"STRATA_KEY": "value", ...} - the two-card (layer) engine environment the rows replace.  Deploy A copies every
threshold key's value (parity by construction, review-fable-phase3 section 5); each row carries its class: A measured /
copied, B copied, C policy (review 7.13).  Rows move later one at a time, as table changes with no new binary."""
import json, sys
ALLOWED = ['STRATA_TP_PREFILL_MIN', 'STRATA_TP_PREFILL_MAX', 'STRATA_TP_PREFILL_CHUNK', 'STRATA_DP_MIN', 'STRATA_DP_PIPE_CHUNK',
           'STRATA_PIPE_HALF_MIN', 'STRATA_PIPE_HALF_MAX', 'STRATA_PREFILL_EQUAL_TAIL_MAX', 'STRATA_PREFILL_AUTO_MAX',
           'STRATA_TP_DEAL_BUDGET', 'STRATA_FUSED_ROUTED_MAX', 'STRATA_REFILL_HEAD', 'STRATA_REFILL_HEAD_MIN', 'STRATA_REFILL_HEAD2',
           'STRATA_REFILL_HEAD2_MIN', 'STRATA_REFILL_HEAD_BIG', 'STRATA_REFILL_HEAD_BIG_CHUNK', 'STRATA_REFILL_HEAD_TP_KEEP',
           'STRATA_REFILL_HEAD_TP_MIN']
CLASS = {'STRATA_TP_PREFILL_MIN': 'A', 'STRATA_TP_PREFILL_MAX': 'A', 'STRATA_DP_MIN': 'A'}
CLASS.update({k: 'C' for k in ALLOWED if k.startswith('STRATA_REFILL_')})
env = json.load(open(sys.argv[1])); q = sys.argv[2]
print(f'# two-card threshold rows, {q} (Deploy A: today\'s values, parity by construction; make_planner_rows2.py)')
print('# class A = path crossover (copied until a measured crossover moves it), B = geometry (copied), C = refill policy')
for k in ALLOWED:
    if env.get(k) not in (None, ''):
        c = CLASS.get(k, 'B'); kind = {'A': 'copied', 'B': 'copied', 'C': 'policy'}[c]
        print(f'row2 {k} {int(env[k])}   # class {c} {kind}')
