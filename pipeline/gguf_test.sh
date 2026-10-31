#!/bin/bash
# CPU load and speed test of the published GGUFs, next to the 4B Q4 (16 threads, greedy, thinking on).
L=$HOME/cp/llama.cpp/build/bin/llama-cli; O=$HOME/cp/8b/final/gguf_test.txt; : > $O
P='Write a Slurm batch script for a job array of 10 tasks, at most 2 at a time, 1 GPU each.'
for m in $HOME/cp/8b/final/upload/gguf/8b-Q8_0.gguf $HOME/cp/8b/final/upload/gguf/8b-Q4_K_M.gguf $(ls $HOME/cp/models_gguf/*.gguf | head -1); do
    echo "=== $m" >> $O
    timeout 900 $L -m "$m" -t 16 -ngl 0 -c 8192 --temp 0 -n 700 --single-turn --simple-io -p "$P" < /dev/null >> $O 2> $O.err; echo "rc=$?" >> $O
    grep -E 'eval time|load time|error|Prompt:|Generation:' $O.err | cut -c1-130 >> $O
done
echo ALLDONE >> $O
