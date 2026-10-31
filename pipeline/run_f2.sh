#!/bin/bash
# Second measurement of the published weights with retriever f: its first score was the best so far.
export NGPU=4
source $HOME/cp/8b/bin/lib.sh; TASK=8b-f2; cd $REPO
[ -s $CP/runs/e8_f_on2/RESULT.json ] || RAG_VARIANT=f THINK=1 bash $CP/8b/bin/eval.sh e8_f_on2 $CP/exp8b/e8_all/model; stop_all
python3 $CP/8b/bin/report.py > $CP/8b/final/RESULTS.md
( unset HF_HUB_OFFLINE; export HF_HOME=$CP/hf; hf upload VaibhavKKM/hpc-assistant-qwen3-8b $CP/8b/final/RESULTS.md RESULTS.md --repo-type model >> $CP/8b/logs/final.log 2>&1 && log "results uploaded" )
log "8B F2 DONE"
