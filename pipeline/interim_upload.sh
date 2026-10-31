#!/bin/bash
# Safety upload of e8_all (user: VaibhavKKM/hpc-assistant-qwen3-8b), replaced later by the final model if a thinking stage wins.
W=$HOME/cp/8b/interim; rm -rf $W/upload; mkdir -p $W/upload/training; cd $W; source $HOME/cp/venv/bin/activate
python3 $HOME/cp/8b/bin/report.py > RESULTS.md
for f in $HOME/cp/exp8b/e8_all/model/*; do ln -s "$f" upload/; done; cp -L $HOME/cp/models/qwen3-8b/LICENSE upload/LICENSE
cp $HOME/cp/8b/bin/*.py $HOME/cp/8b/bin/*.sh $HOME/cp/8b/bin/*.sbatch upload/training/; sed '/^FINAL_MODEL/d' RESULTS.md > upload/RESULTS.md
{ printf -- '---\nlicense: apache-2.0\nbase_model: Qwen/Qwen3-8B\nlibrary_name: transformers\npipeline_tag: text-generation\ntags:\n- hpc\n- slurm\n---\n\n'
cat <<'CARD'
# HPC assistant, Qwen3-8B student (interim)

Qwen3-8B distilled from Qwen3-235B-A22B-Instruct-2507 for HPC and Slurm questions, built by the recipe of the 4B release VaibhavKKM/hpc-assistant-qwen3-4b. Interim upload: a thinking-mode stage is still in training and may replace these weights.

Thinking is switched with the chat template argument `enable_thinking` (on by default in this template). The thinking-on scores come from a model that was not trained on thinking data.

## How to read the scores

Each score is a win rate, not accuracy: three judge models compare the answer with the closed-book answer of the base Qwen3-4B, in both orders, with a majority vote; ties count one half. Answers use retrieved documentation excerpts. Thinking is stripped before judging, and a reply whose thinking does not finish counts as a non-answer. Run-to-run noise of the same weights was about 0.01 (the 4B release scored 0.8665 and 0.8558), so small gaps mean little. e8_all (thinking off) was fixed in advance as the primary result; the other runs are extra candidates.

Known limits: prompts over 10,000 tokens lost their start in training (46 of 136,452 sequences). Training prompts hold a few traces of the held-out Aalto family (about 3 per 30,000 sampled), as the 4B data does. Tool calling and the product test suites were not re-run for this model.

CARD
sed '/^FINAL_MODEL/d' RESULTS.md; printf '\nThe pipeline is in `training/`.\n'; } > upload/README.md
grep -rlE 'hf_[A-Za-z0-9]{20,}' upload/training upload/README.md && { echo "rc=token-in-files" > upload.log; exit 1; }
HF_HOME=$HOME/cp/hf hf upload VaibhavKKM/hpc-assistant-qwen3-8b upload . --repo-type model > upload.log 2>&1; echo "rc=$?" >> upload.log
