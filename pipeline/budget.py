"""MAX_STEPS for a training run to fit SECONDS, from the measured s/step and sequence length of a previous run."""
import glob, json, re, sys
prev_log, prev_data, data, seconds = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
mean = lambda d: (lambda v: sum(v) / len(v))([(r['n_prompt'] + r['n_out']) for f in glob.glob(f'{d}/part*.jsonl') for r in map(json.loads, open(f))])
sps = float(re.findall(r'([0-9.]+)s/step', open(prev_log).read())[-1])
print(int(0.9 * seconds / (sps * mean(data) / mean(prev_data))))
