#!/bin/bash
# Retriever c on the published weights, thinking on: general documentation excerpts for every question (evaluation sources excluded).
export NGPU=4
source $HOME/cp/8b/bin/lib.sh; TASK=8b-c; cd $REPO
[ -s $CP/runs/e8_c_on/RESULT.json ] || RAG_VARIANT=c THINK=1 bash $CP/8b/bin/eval.sh e8_c_on $CP/exp8b/e8_all/model; stop_all
python3 $CP/8b/bin/report.py > $CP/8b/final/RESULTS.md
( unset HF_HUB_OFFLINE; export HF_HOME=$CP/hf; hf upload VaibhavKKM/hpc-assistant-qwen3-8b $CP/8b/final/RESULTS.md RESULTS.md --repo-type model >> $CP/8b/logs/final.log 2>&1 && log "results uploaded" )
log "8B C DONE"
