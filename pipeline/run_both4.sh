#!/bin/bash
# Thinking stage on 4 GPUs (user, Oct 31: free 4 of the team's 8 for a teammate). Thinking data is complete; 64 sequences per step as before.
export NGPU=4
source $HOME/cp/8b/bin/lib.sh; TASK=8b; cd $REPO
X=$CP/exp8b; U=$X/thk_q397fp8/data; NT=$'<think>\n\n</think>\n\n'
FREEZE=$(date -d '2026-10-31 17:30 UTC' +%s)
ev() { [ -s $CP/runs/$1/RESULT.json ] || THINK=${3:-0} bash $CP/8b/bin/eval.sh $1 $2 || log "eval $1 failed"; stop_all; }
if [ ! -s $X/e8_both/model/KD_TRAIN.json ]; then
    mkdir -p $X/e8_both
    for try in 1 2 3; do
        CKPT_DIR=$X/e8_both/ckpt MAX_LEN=10000 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROMPT_SUFFIX="$NT" ACCUM=16 timeout -k 120 $(( FREEZE - $(date +%s) )) torchrun --standalone --nproc_per_node 4 $CP/8b/bin/train_kd.py $X/e8_all/model $U $X/e8_both/model >> $CP/8b/logs/e8_both.train.log 2>&1 \
            && { log "trained e8_both: $(tr -d '\n ' < $X/e8_both/model/KD_TRAIN.json)"; break; } || log "training e8_both stopped, try $try"
        stop_all; sleep 30
    done
    stop_all
fi
if [ -s $X/e8_both/model/KD_TRAIN.json ]; then ev e8_both_on $X/e8_both/model 1; ev e8_both_off $X/e8_both/model; fi
bash $CP/8b/bin/finalize.sh
log "8B CHAIN DONE"
