#!/bin/bash
# Product checks of the published 8B, thinking off and on: Simon's 25 mentor cases (bare and in the app) and the European QA sweep
# (same questions and reviewer as the 4B 1.3 sweep eu2, real tool calls against this cluster's Slurm, actions declined).
export NGPU=4
source $HOME/cp/8b/bin/lib.sh; TASK=8b-prod; cd $CP
BEST=$(sed -n 's/^FINAL_MODEL //p' $CP/8b/final/RESULTS.md | head -1); P=$CP/8b/prod; mkdir -p $P; log "product checks on $BEST"
push() { # summaries only: raw sweep rows hold live cluster details
    ( unset HF_HUB_OFFLINE; export HF_HOME=$CP/hf; mkdir -p $P/summary; cp -f $P/mentor/score.out $P/summary/mentor.txt 2>/dev/null
      for m in on off; do tail -3 $P/sweep_$m/out.txt > $P/summary/sweep_$m.txt 2>/dev/null; done
      printf 'Product checks of the published 8B. 4B 1.3 on the same European sweep: ok 32, minor 116, wrong 36 of 184. mentor.txt: 25 mentor cases, blind key-point scoring; v13 is the 4B 1.3.\n' > $P/summary/README.txt
      hf upload VaibhavKKM/hpc-assistant-qwen3-8b $P/summary product_checks --repo-type model >> $CP/8b/logs/final.log 2>&1 && log "product summary uploaded" || log "product summary upload failed" )
}
OFF=$P/model_off; rm -rf $OFF; mkdir -p $OFF; for f in "$BEST"/*; do ln -s "$f" $OFF/; done
rm $OFF/chat_template.jinja; sed 's/enable_thinking is defined and enable_thinking is false/true/' "$BEST"/chat_template.jinja > $OFF/chat_template.jinja
serve 0,1 8103 $M/q235fp8 qwen3-235b --max-model-len 16384 --gpu-memory-utilization 0.85 || exit 1
serve 2 8100 $OFF cand --max-model-len 16384 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes || exit 1
serve 3 8101 "$BEST" cand --max-model-len 16384 --gpu-memory-utilization 0.85 --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 || log "thinking-on server failed"
mkdir -p $P/mentor
timeout 1800 python $CP/cb/mentor_models.py $CP/mentor/cases25.jsonl $P/mentor off=http://127.0.0.1:8100/v1 on=http://127.0.0.1:8101/v1 2>&1 | tail -4
for m in off_bare off_app on_bare on_app; do cp -n $CP/mentor/cb/v13_app.jsonl $P/mentor/ 2>/dev/null; done; cp -n $CP/mentor/cb/v13_bare.jsonl $P/mentor/
timeout 1200 python $CP/mentor/score_mentor.py $CP/mentor/cases25.jsonl $P/mentor/scores.json v13_bare=$P/mentor/v13_bare.jsonl v13_app=$P/mentor/v13_app.jsonl off_bare=$P/mentor/off_bare.jsonl off_app=$P/mentor/off_app.jsonl on_bare=$P/mentor/on_bare.jsonl on_app=$P/mentor/on_app.jsonl 2>&1 | tee $P/mentor/score.out
log "mentor 8B: $(tail -6 $P/mentor/score.out | tr '\n' ' ')"
push
for mode in on off; do
    U=8100; [ $mode = on ] && U=8101
    O=$P/sweep_$mode; mkdir -p $O; cp $CP/qa_sweep/eu/questions.json $O/
    CAND=http://127.0.0.1:$U/v1 timeout 3000 python $CP/bin/qa_sweep_eu.py $O 12223 12266 2>&1 | tee $O/out.txt | tail -3
    log "sweep $mode: $(tail -2 $O/out.txt | tr '\n' ' ')"
    push
done
log "8B PRODUCT CHECKS DONE"; stop_all
