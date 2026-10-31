"""All 8B runs next to 1.3, with answer-quality guards. The primary result stays e8_all whatever else scored."""
import glob, json, os
CP = os.path.expanduser('~/cp')
NAMES = ['kdall', 'kdall6', 'e8_ctrl13', 'e8_van_off', 'e8_van_on', 'tc_t235think', 'tc_397on', 'tc_480', 'tc_t24', 'e8_k1', 'e8_k2', 'e8_all', 'e8_all_on', 'e8_both_on', 'e8_both_off', 'e8_all6', 'e8_x1_on', 'e8_x2_on', 'e8_x3_on', 'e8_all_on2', 'e8_x1_on2', 'e8_all_off2', 'e8_soup_on', 'e8_f_on', 'e8_soup_f_on', 'e8_f_on2']
print('| run | mean | v2 | v3 | d1 | no answer | hit limit |\n|---|---|---|---|---|---|---|')
res = {}
for n in NAMES:
    p = f'{CP}/runs/{n}/RESULT.json'
    if not os.path.exists(p):
        continue
    d = json.load(open(p)); m = d['majority']; res[n] = d['mean_cand']
    rows = [json.loads(line) for f in glob.glob(f'{CP}/runs/{n}/answers/*__cand.jsonl') for line in open(f)]
    na = sum(r['answer'].startswith('[no answer') for r in rows); ln = sum(r.get('finish_reason') == 'length' for r in rows)
    print(f"| {n} | {d['mean_cand']:.4f} | {m['v2']['a']:.4f} | {m['v3']['a']:.4f} | {m['d1']['a']:.4f} | {na} | {ln} |")
print('\nPRIMARY (fixed in advance): e8_all, thinking off:', res.get('e8_all'))
th = open(f'{CP}/exp8b/THINK_TEACHER').read().strip() if os.path.exists(f'{CP}/exp8b/THINK_TEACHER') else None
print('Thinking teacher:', th)
# The published weights: the better thinking-on score, so the thinking stage only replaces e8_all if it won.
on = [(res[n], p) for n, p in (('e8_both_on', 'e8_both'), ('e8_all_on', 'e8_all')) if n in res]
fall = [p for n, p in (('e8_all', 'e8_all'), ('e8_k2', 'e8_k2'), ('e8_k1', 'e8_k1')) if n in res]
best = max(on)[1] if on else (fall[0] if fall else None)
# Extra rounds replace the published weights only with a win above the run-to-run noise.
top = max(on)[0] if on else 0
for n in ('e8_x1', 'e8_x2', 'e8_x3', 'e8_soup'):
    if res.get(n + '_on', 0) >= top + 0.01:
        top, best = res[n + '_on'], n
if best:
    print('Published weights:', best)
    print('FINAL_MODEL', f'{CP}/exp8b/{best}/model')
