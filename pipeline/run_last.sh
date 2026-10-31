#!/bin/bash
# Last honest levers before the access ends. A weight average of the 8B rounds (soup) needs no new training; retriever f adds the
# general documentation index (evaluation sources excluded) for non-Slurm questions. Published weights change only by the 0.01 rule.
export NGPU=4
source $HOME/cp/8b/bin/lib.sh; TASK=8b-last; cd $REPO
X=$CP/exp8b; D=$X/e8_soup/model
ev() { [ -s $CP/runs/$1/RESULT.json ] || THINK=1 bash $CP/8b/bin/eval.sh $1 $2 || log "eval $1 failed"; stop_all; }
[ -s $D/model.safetensors ] || python $CP/8b/bin/soup.py $D $X/e8_all/model $X/e8_x1/model $X/e8_x3/model >> $CP/8b/logs/soup.log 2>&1 || { log "soup failed"; }
[ -s $D/model.safetensors ] && { cp -n $X/e8_all/model/chat_template.jinja $D/ 2>/dev/null; ev e8_soup_on $D; }
bash $CP/8b/bin/finalize.sh
RAG_VARIANT=f ev e8_f_on $X/e8_all/model
[ -s $D/model.safetensors ] && RAG_VARIANT=f ev e8_soup_f_on $D
python3 $CP/8b/bin/report.py > $CP/8b/final/RESULTS.md
( unset HF_HUB_OFFLINE; export HF_HOME=$CP/hf; hf upload VaibhavKKM/hpc-assistant-qwen3-8b $CP/8b/final/RESULTS.md RESULTS.md --repo-type model >> $CP/8b/logs/final.log 2>&1 && log "results uploaded" )
log "8B LAST DONE"
