#!/bin/bash
# Night rounds (user, Oct 31: access until maybe noon). Learning rate 1e-5 gave the largest single gain on the 4B and was never tried on 8B.
# A candidate needs >= 0.9191 thinking on, then a second measurement; it is published only if its mean beats the published mean by 0.01.
source $HOME/cp/8b/bin/lib.sh; TASK=8b-night; cd $REPO
X=$CP/exp8b; NT=$'<think>\n\n</think>\n\n'; BASE=$X/e8_k2/model
[ -s $CP/8b/PUBMEAN ] || echo 0.9124 > $CP/8b/PUBMEAN
tr_() { # tr_ NAME STUDENT DATA LR
    [ -s $X/$1/model/KD_TRAIN.json ] && return 0
    for t in 1 2 3; do
        CKPT_DIR=$X/$1/ckpt MAX_LEN=10000 LR=$4 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PROMPT_SUFFIX="$NT" ACCUM=8 torchrun --standalone --nproc_per_node 8 $CP/8b/bin/train_kd.py $2 $3 $X/$1/model >> $CP/8b/logs/$1.train.log 2>&1 && { log "trained $1: $(tr -d '\n ' < $X/$1/model/KD_TRAIN.json)"; stop_all; return 0; }
        stop_all; log "$1 stopped, try $t"; sleep 30
    done
    return 1
}
ev() { [ -s $CP/runs/$1/RESULT.json ] || THINK=1 bash $CP/8b/bin/eval.sh $1 $2 || log "eval $1 failed"; stop_all; }
score() { python3 $CP/8b/bin/score.py $CP/runs/$1/RESULT.json; }
results() { python3 $CP/8b/bin/report.py > $CP/8b/final/RESULTS.md; ( unset HF_HUB_OFFLINE; export HF_HOME=$CP/hf; hf upload VaibhavKKM/hpc-assistant-qwen3-8b $CP/8b/final/RESULTS.md RESULTS.md --repo-type model >> $CP/8b/logs/final.log 2>&1 && log "results uploaded" ); }
round_() { # round_ NAME STUDENT DATA LR
    tr_ e9_$1 $2 $3 $4 || { log "round $1 failed"; return; }
    ev e9_$1_on $X/e9_$1/model; local a=$(score e9_$1_on); log "round $1 first score $a"; results
    awk -v a=$a 'BEGIN{exit !(a >= 0.9191)}' || return
    ev e9_$1_on2 $X/e9_$1/model; local b=$(score e9_$1_on2); log "round $1 second score $b"; results
    local m=$(awk -v a=$a -v b=$b 'BEGIN{printf "%.4f", (a + b) / 2}')
    awk -v m=$m -v p=$(cat $CP/8b/PUBMEAN) 'BEGIN{exit !(m >= p + 0.01)}' || { log "round $1 mean $m: not published"; return; }
    echo e9_$1 > $CP/8b/WINNER; echo $m > $CP/8b/PUBMEAN; log "round $1 mean $m wins: publishing"; bash $CP/8b/bin/finalize.sh
}
best() { if [ -s $CP/8b/WINNER ]; then echo $X/$(cat $CP/8b/WINNER)/model; else echo "$1"; fi; }
round_ a $BASE $CP/exp/kdall/teacher 1e-5
# all 235B-Instruct soft-target sets, more data at the better rate
U=$X/night_u; mkdir -p $U
for e in kd1 kd2 kd3 kdreq kdmpi kdmpi2 kd4 kdgpu; do for f in $CP/exp/$e/teacher/part*.npz; do b=$(basename $f .npz); ln -sf $f $U/part_${e}_${b#part}.npz; ln -sf ${f%.npz}.jsonl $U/part_${e}_${b#part}.jsonl; done; done
round_ b $(best $BASE) $U 1e-5
log "8B NIGHT DONE"
