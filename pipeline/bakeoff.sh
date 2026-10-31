#!/bin/bash
# Controls, then every teacher candidate on the same eval. Candidates hold all 8 GPUs while answering; judges unchanged.
source $HOME/cp/8b/bin/lib.sh; TASK=8b-bake; cd $REPO
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c | tee -a $CP/8b/logs/gpu.txt
python $CP/8b/bin/check_prompts.py >> $CP/8b/logs/gate.log 2>&1 || { log "STOP: data gate failed"; exit 1; }
B=$M/qwen3-8b-cp
ev() { [ -s $CP/runs/$1/RESULT.json ] || THINK=${3:-0} bash $CP/8b/bin/eval.sh $1 $2 || log "eval $1 failed, next"; stop_all; }
ev e8_ctrl13 $CP/exp/kdall/model
ev e8_van_off $B; ev e8_van_on $B 1
ev_t() { until [ -e $M/$2/.DL_OK ]; do sleep 60; done; CAND_GPUS=0,1,2,3,4,5,6,7 THINK_BUDGET=8192 ev $1 $M/$2 $3; }
ev_t tc_t235think q235think_fp8 1
ev_t tc_397on q397fp8 1
ev_t tc_480 q480fp8 0
ev_t tc_t24 t24_nvfp4 1
log "BAKEOFF DONE"
