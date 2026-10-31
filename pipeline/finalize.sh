#!/bin/bash
# Report of every run; GGUF Q8 + Q4 of the best thinking-capable model and of the best overall; no upload.
source $HOME/cp/8b/bin/lib.sh; TASK=8b-final; R=$CP/8b/final; mkdir -p $R
python $CP/8b/bin/report.py > $R/RESULTS.md 2>> $CP/8b/logs/final.log
log "report written: $R/RESULTS.md"
BEST=$(sed -n 's/^FINAL_MODEL //p' $R/RESULTS.md | head -1)
[ -d "$BEST" ] || { log "no final model"; exit 0; }
cd $CP/llama.cpp
python convert_hf_to_gguf.py "$BEST" --outtype f16 --outfile $R/8b-f16.gguf >> $CP/8b/logs/final.log 2>&1 || log "gguf convert failed"
for q in Q8_0 Q4_K_M; do build/bin/llama-quantize $R/8b-f16.gguf $R/8b-$q.gguf $q >> $CP/8b/logs/final.log 2>&1 || log "quantize $q failed"; done
rm -f $R/8b-f16.gguf; ls -la $R >> $R/RESULTS.md
# Upload (user: VaibhavKKM/hpc-assistant-qwen3-8b). Rounds continue until access ends: the token file is removed after the last one.
unset HF_HUB_OFFLINE; export HF_HOME=$CP/hf; REPO_ID=VaibhavKKM/hpc-assistant-qwen3-8b; S=$R/upload; rm -rf $S; mkdir -p $S/training $S/gguf
cp -L "$BEST"/* $S/; mv $R/8b-*.gguf $S/gguf/ 2>/dev/null
cp $CP/8b/bin/*.py $CP/8b/bin/*.sh $CP/8b/bin/*.sbatch $S/training/; cp $R/RESULTS.md $S/RESULTS.md
PICK=$(basename $(dirname "$BEST")); TEACHER=$(cat $CP/exp8b/THINK_TEACHER 2>/dev/null || echo none)
{ printf -- '---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3-8B\nlibrary_name: transformers\npipeline_tag: text-generation\ntags:\n- hpc\n- slurm\n---\n\n'
  printf '# HPC assistant, Qwen3-8B student\n\nQwen3-8B distilled from larger Qwen models for HPC and Slurm questions by the recipe of the 4B release VaibhavKKM/hpc-assistant-qwen3-4b. Published run: %s. Answer data: Qwen3-235B-A22B-Instruct-2507; `e8_both` adds a thinking stage from %s (final text only, no soft targets). Thinking is switched with the chat template argument `enable_thinking` (on by default).\n\n' "$PICK" "$TEACHER"
  printf '## How to read the scores\n\nEach score is a win rate, not accuracy: three judge models compare the answer with the closed-book answer of the base Qwen3-4B, in both orders, with a majority vote; ties count one half. Answers use retrieved documentation excerpts. Thinking is stripped before judging, and a reply whose thinking does not finish counts as a non-answer. Run-to-run noise of the same weights was about 0.01 (the 4B release scored 0.8665 and 0.8558), so small gaps mean little. e8_all (thinking off) was fixed in advance as the primary result; the other runs are extra candidates, so the best of them is a best-of-several figure.\n\nKnown limits: prompts over 10,000 tokens lost their start in thinking-off training (46 of 136,452 sequences); over-long thinking answers were skipped. Training prompts hold a few traces of the held-out Aalto family (about 3 per 30,000 sampled), as the 4B data does. Tool calling and the product test suites were not re-run for this model.\n\n'
  sed '/^FINAL_MODEL/d' $R/RESULTS.md; printf '\nThe pipeline is in `training/`; GGUF files are in `gguf/`.\n'; } > $S/README.md
cp -L $CP/models/qwen3-8b/LICENSE $S/LICENSE
grep -rlE 'hf_[A-Za-z0-9]{20,}' $S/training $S/README.md && { log "UPLOAD STOPPED: token in files"; exit 1; }
ok=0; for i in 1 2 3; do
    hf repo create $REPO_ID --repo-type model --exist-ok >> $CP/8b/logs/final.log 2>&1
    hf upload $REPO_ID $S . --repo-type model >> $CP/8b/logs/final.log 2>&1 && { ok=1; break; }; sleep 120
done
if [ $ok = 1 ]; then log "uploaded to $REPO_ID: $(ls $S | wc -l) entries"; else log "UPLOAD FAILED (logs: 8b/logs/final.log)"; fi
