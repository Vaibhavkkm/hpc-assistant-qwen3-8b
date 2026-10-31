#!/bin/bash
# User, Oct 31: keep improving until the access ends. Each round starts from the best thinking-on model and is scored thinking on;
# it is published only if it wins by the run-to-run noise (report.py). No new training starts after CUT.
export NGPU=4
source $HOME/cp/8b/bin/lib.sh; TASK=8b; cd $REPO
X=$CP/exp8b; NT=$'<think>\n\n</think>\n\n'
CUT=$(date -d '2026-10-31 21:00 UTC' +%s)
best() { python3 -c "
import json,os
C=[('e8_all_on','e8_all'),('e8_both_on','e8_both'),('e8_x1_on','e8_x1'),('e8_x2_on','e8_x2'),('e8_x3_on','e8_x3')]
r=[(json.load(open(f'$CP/runs/{n}/RESULT.json'))['mean_cand'],m) for n,m in C if os.path.exists(f'$CP/runs/{n}/RESULT.json')]
print(max(r)[1])"; }
round_() { # round_ NAME SETS...
    local n=$1; shift; local left=$(( CUT - $(date +%s) - 2400 )); [ $left -gt 1500 ] || { log "no time for $n"; return; }
    local D=$X/$n/data S=$X/$(best)/model; mkdir -p $D
    for e in "$@"; do for f in $CP/exp/$e/teacher/part*.npz; do b=$(basename $f .npz); ln -sf $f $D/part_${e}_${b#part}.npz; ln -sf ${f%.npz}.jsonl $D/part_${e}_${b#part}.jsonl; done; done
    local steps=$(python $CP/8b/bin/budget.py $CP/8b/logs/e8_both.train.log $X/thk_q397fp8/data $D $left)
    log "round $n from $S on $*: at most $steps steps, $left s"
    if [ ! -s $X/$n/model/KD_TRAIN.json ]; then
        for t in 1 2; do
            CKPT_DIR=$X/$n/ckpt MAX_LEN=10000 MAX_STEPS=$steps PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROMPT_SUFFIX="$NT" ACCUM=16 timeout -k 120 $left torchrun --standalone --nproc_per_node 4 $CP/8b/bin/train_kd.py $S $D $X/$n/model >> $CP/8b/logs/$n.train.log 2>&1 && break
            stop_all; log "round $n stopped, try $t"; sleep 30
        done
        stop_all
    fi
    [ -s $X/$n/model/KD_TRAIN.json ] || return
    [ -s $CP/runs/${n}_on/RESULT.json ] || THINK=1 bash $CP/8b/bin/eval.sh ${n}_on $X/$n/model || log "eval $n failed"; stop_all
    log "round $n done"; bash $CP/8b/bin/finalize.sh
}
round_ e8_x1 kdreq kdmpi kdmpi2
round_ e8_x2 kdgpu
round_ e8_x3 kd4
log "8B EXTRA DONE"
