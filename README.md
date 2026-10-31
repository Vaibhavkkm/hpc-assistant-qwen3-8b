# HPC assistant, Qwen3-8B student: training pipeline

The scripts that trained and evaluated the Qwen3-8B student on the Curiosity B300 cluster (Oct 30-31 2026), copied as they
ran. They build on `research/curiosity_pipeline` in pparitoshh/alembic. Paths are the Curiosity ones.

Model and scores: https://huggingface.co/VaibhavKKM/hpc-assistant-qwen3-8b

| Script | What it does |
|---|---|
| `check_prompts.py` | Gate before training: template match, no think tokens in teacher output, held-out pattern count |
| `bakeoff.sh` | Controls and teacher candidates on the same evaluation |
| `run_8b.sh` | The 1.3 recipe from Qwen3-8B, then the thinking stage |
| `run_both4.sh` | The thinking stage on 4 GPUs |
| `run_extra.sh` | Extra rounds from the best model so far |
| `run_prod.sh` | Mentor cases and the European QA sweep, thinking off and on |
| `train_kd.py` | Full-parameter distillation, resumable, with a fit test on the longest sequence |
| `teacher_gen2.py` | Thinking-teacher answers, soft targets or re-tokenised text |
| `eval.sh`, `answer.py` | The evaluation, thinking off or on |
| `finalize.sh`, `report.py` | Results table, GGUF files, upload |
