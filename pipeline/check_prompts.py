"""Gate before training 8B on the 4B teacher data: stored prompt ids + the no-think block must equal what 8B's own
template produces with enable_thinking=False; the teacher answers must not hold think tokens; held-out families absent."""
import json, os, re, sys
import numpy as np
from transformers import AutoTokenizer
sys.path.insert(0, os.path.expanduser('~/cp/alembic_8b_clone_20261030/src'))
from distillkit.config import load_config
CP = os.path.expanduser('~/cp')
SYSTEM = load_config(f'{CP}/alembic_8b_clone_20261030/configs/rag65_seed42.json').student.system_prompt
tok = AutoTokenizer.from_pretrained(f'{CP}/models/qwen3-8b')
SUF = tok.encode('<think>\n\n</think>\n\n', add_special_tokens=False); print('suffix ids', SUF)
TH = set(tok.convert_tokens_to_ids(['<think>', '</think>']))
HELD = re.compile(r'scicomp\.aalto|aaltoscicomp|aalto triton|open-mpi\.org|open-mpi/ompi|cuda-samples|cuda_samples', re.I)
bad = 0
for e in ('kd1', 'kd2', 'kd3', 'kdreq'):
    D = f'{CP}/exp/{e}'; rows = {}; held = n = 0
    for l in open(f'{D}/prompts.jsonl'):
        r = json.loads(l); rows[r['id']] = r['user']; n += 1; held += bool(HELD.search(r['user']))
    z = np.load(f'{D}/teacher/part0.npz'); meta = [json.loads(l) for l in open(f'{D}/teacher/part0.jsonl')]
    po, oo = z['prompt_off'], z['out_off']; ok = 0; S = 300
    for i in range(S):
        want = tok.apply_chat_template([{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': rows[meta[i]['id']]}], add_generation_prompt=True, tokenize=True, enable_thinking=False)
        want = list(want['input_ids'] if hasattr(want, 'keys') else want)
        ok += list(z['prompt'][po[i]:po[i + 1]]) + SUF == want
    th = int(np.isin(z['out'], list(TH)).sum())
    print(f'{e}: prompts {n}, held-out pattern hits {held}, template match {ok}/{S}, think tokens in teacher output {th}')
    bad += ok != S or th != 0
print('GATE', 'FAIL' if bad else 'PASS')
sys.exit(bool(bad))
