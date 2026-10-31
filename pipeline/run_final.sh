#!/bin/bash
# Last checks: score the mentor cases (a reply with no answer counts as empty, the scorer stopped on it), then measure the
# published model and round 1 a second time: one score of the same weights moved by 0.0107 on the 4B.
export NGPU=4
source $HOME/cp/8b/bin/lib.sh; TASK=8b-final2; cd $CP
X=$CP/exp8b; P=$CP/8b/prod; S=$P/summary
python3 - $P/mentor <<'PY'
import json, sys
d = sys.argv[1]
for m in ('on_bare', 'on_app', 'off_bare', 'off_app'):
    rows = [json.loads(l) for l in open(f'{d}/{m}.jsonl')]
    empty = sum(not r['answer'] for r in rows)
    with open(f'{d}/{m}.jsonl', 'w') as f:
        for r in rows:
            r['answer'] = r['answer'] or ''
            f.write(json.dumps(r) + '\n')
    print(m, 'no answer', empty)
PY
serve 0,1,2,3 8103 $M/q235fp8 qwen3-235b --max-model-len 16384 --gpu-memory-utilization 0.85 || exit 1
timeout 1200 python $CP/mentor/score_mentor.py $CP/mentor/cases25.jsonl $P/mentor/scores.json v13_bare=$P/mentor/v13_bare.jsonl v13_app=$P/mentor/v13_app.jsonl off_bare=$P/mentor/off_bare.jsonl off_app=$P/mentor/off_app.jsonl on_bare=$P/mentor/on_bare.jsonl on_app=$P/mentor/on_app.jsonl 2>&1 | tee $P/mentor/score.out
stop_all
log "mentor 8B: $(tail -6 $P/mentor/score.out | tr '\n' ' ')"
( unset HF_HUB_OFFLINE; export HF_HOME=$CP/hf; cp -f $P/mentor/score.out $S/mentor.txt; hf upload VaibhavKKM/hpc-assistant-qwen3-8b $S product_checks --repo-type model >> $CP/8b/logs/final.log 2>&1 && log "product summary uploaded" )
ev() { [ -s $CP/runs/$1/RESULT.json ] || THINK=$3 bash $CP/8b/bin/eval.sh $1 $2 || log "eval $1 failed"; stop_all; }
ev e8_all_on2 $X/e8_all/model 1
ev e8_x1_on2 $X/e8_x1/model 1
ev e8_all_off2 $X/e8_all/model 0
log "8B REPEATS DONE"
