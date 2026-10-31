#!/bin/bash
# 8B student, unattended. Primary result fixed in advance: the 1.3 recipe (kd1 -> kd2 -> union kd1-3) from Qwen3-8B, thinking off.
# Then a thinking mode distilled from the bake-off's best thinking teacher. Resumable. A failed stage never blocks the final report.
source $HOME/cp/8b/bin/lib.sh; TASK=8b; cd $REPO
X=$CP/exp8b; B=$M/qwen3-8b-cp; NT=$'<think>\n\n</think>\n\n'; mkdir -p $X
FREEZE=$(date -d '2026-10-31 17:30 UTC' +%s)      # Sat 18:30 CET: training stops
TRAIN_END=$(date -d '2026-10-31 16:15 UTC' +%s)   # leaves time for the two thinking-model evals before the freeze
GEN_LAST=$(date -d '2026-10-31 10:30 UTC' +%s)
left() { echo $(( $1 - $(date +%s) )); }
ev() { [ -s $CP/runs/$1/RESULT.json ] || THINK=${3:-0} bash $CP/8b/bin/eval.sh $1 $2 || log "eval $1 failed"; stop_all; }
tr_() { # tr_ NAME STUDENT TEACHER_DIR [MAX_STEPS]
    [ -s $X/$1/model/KD_TRAIN.json ] && return 0
    [ "$(left $FREEZE)" -gt 600 ] || { log "no time for $1"; return 1; }
    mkdir -p $X/$1
    local L ok=0
    for L in 12288 10000 9000; do  # the first 8B run ran out of GPU memory on a rare long prompt; only over-long prompts lose their start
        if MAX_LEN=$L PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MAX_STEPS=$4 PROMPT_SUFFIX="$NT" ACCUM=8 timeout -k 120 $(left $FREEZE) torchrun --standalone --nproc_per_node 8 $CP/8b/bin/train_kd.py $2 $3 $X/$1/model >> $CP/8b/logs/$1.train.log 2>&1; then ok=1; break; fi
        stop_all; log "training $1 failed at MAX_LEN $L"; sleep 30
    done
    [ $ok = 1 ] || { log "training $1 failed or timed out"; return 1; }
    stop_all; log "trained $1: $(tr -d '\n ' < $X/$1/model/KD_TRAIN.json)"
}
primary() {
    tr_ e8_k1 $B $CP/exp/kd1/teacher || return; ev e8_k1 $X/e8_k1/model
    tr_ e8_k2 $X/e8_k1/model $CP/exp/kd2/teacher || return
    tr_ e8_all $X/e8_k2/model $CP/exp/kdall/teacher || return; ev e8_all $X/e8_all/model
    ev e8_all_on $X/e8_all/model 1
}
gen_teacher() { # gen_teacher MODEL G TP: 0 when enough answers exist
    local T=$X/thk_$1 N P rc=0 pids=(); mkdir -p $T/teacher
    [ -s $T/prompts.jsonl ] || cat $CP/exp/kd1/prompts.jsonl $CP/exp/kd2/prompts.jsonl | grep -viE 'scicomp\.aalto|aaltoscicomp|aalto triton' | shuf --random-source=<(yes) > $T/prompts.jsonl
    N=$(wc -l < $T/prompts.jsonl); P=$(( N / 500 ))   # small parts: a stopped run keeps a uniform sample
    local stop=$(( $(date +%s) + 8 * 3600 )); [ $stop -gt $GEN_LAST ] && stop=$GEN_LAST
    for g in $(seq 0 $(($2 - 1))); do
        gpus=$(seq -s, $((g * $3)) $((g * $3 + $3 - 1)))
        CUDA_VISIBLE_DEVICES=$gpus TP=$3 python $CP/8b/bin/teacher_gen2.py $T/prompts.jsonl $T/teacher $g $2 $P $M/$1 $M/r27 $stop >> $CP/8b/logs/thk_$1_$g.log 2>&1 &
        pids+=($!)
    done
    for p in "${pids[@]}"; do wait $p || rc=$?; done; stop_all
    local n=$(cat $T/teacher/part*.jsonl 2>/dev/null | wc -l)
    log "thinking teacher $1: $n answers, rc $rc"
    [ "$n" -ge 10000 ]
}
thinking() {
    [ -s $X/e8_all/model/KD_TRAIN.json ] || return
    CAND_GPUS=0,1,2,3,4,5,6,7 THINK_BUDGET=8192 ev tc_t235think $M/q235think_fp8 1   # lost in the bake-off to a replaced eval.sh
    local T=""
    while read -r m g tp; do
        if gen_teacher $m $g $tp; then T=$X/thk_$m; echo $m > $X/THINK_TEACHER; break; fi
        [ "$(left $GEN_LAST)" -gt 7200 ] || break
    done < <(python $CP/8b/bin/pick_teacher.py 2>/dev/null)
    [ -n "$T" ] || { log "no thinking teacher gave enough data"; return; }
    local U=$T/data; mkdir -p $U
    for f in $T/teacher/part*.npz; do b=$(basename $f .npz); ln -sf $f $U/part_thk_${b#part}.npz; ln -sf ${f%.npz}.jsonl $U/part_thk_${b#part}.jsonl; done
    for i in 0 7 14 21 28 35; do f=$(ls $CP/exp/kdall/teacher/*.npz | sed -n "$((i + 1))p"); b=$(basename $f .npz); ln -sf $(readlink -f $f) $U/$b.npz; ln -sf $(readlink -f ${f%.npz}.jsonl) $U/$b.jsonl; done
    local steps=$(python $CP/8b/bin/budget.py $CP/8b/logs/e8_all.train.log $CP/exp/kdall/teacher $U $(left $TRAIN_END))
    log "thinking training: at most $steps steps"
    tr_ e8_both $X/e8_all/model $U $steps || return
    ev e8_both_on $X/e8_both/model 1; ev e8_both_off $X/e8_both/model
}
python $CP/8b/bin/check_prompts.py >> $CP/8b/logs/gate.log 2>&1 || { log "STOP: data gate failed"; exit 1; }
primary
thinking
# extra candidate only with time left
[ "$(left $FREEZE)" -gt 25200 ] && [ -s $X/e8_k2/model/KD_TRAIN.json ] && { tr_ e8_all6 $X/e8_k2/model $CP/exp/kdall6/teacher && ev e8_all6 $X/e8_all6/model; }
bash $CP/8b/bin/finalize.sh
log "8B CHAIN DONE"
