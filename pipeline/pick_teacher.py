"""Order the thinking teachers from the bake-off: mean score, a same-vocabulary teacher (soft targets) wins ties within MARGIN."""
import json, os, sys
from transformers import AutoTokenizer
CP = os.path.expanduser('~/cp'); MARGIN = 0.01
C = [('tc_t235think', 'q235think_fp8', 4, 2), ('tc_397on', 'q397fp8', 2, 4), ('tc_t24', 't24_nvfp4', 1, 8)]
base = AutoTokenizer.from_pretrained(f'{CP}/models/qwen3-8b').get_vocab()
rows = []
for run, m, g, tp in C:
    p = f'{CP}/runs/{run}/RESULT.json'
    if os.path.exists(p):
        same = AutoTokenizer.from_pretrained(f'{CP}/models/{m}').get_vocab() == base
        rows.append((json.load(open(p))['mean_cand'] + (MARGIN if same else 0), m, g, tp))
for _, m, g, tp in sorted(rows, reverse=True):
    print(m, g, tp)
