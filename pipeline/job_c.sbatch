#!/bin/bash
#SBATCH --job-name=cp-8bc
#SBATCH --partition=hackathon
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=112
#SBATCH --mem=900G
#SBATCH --time=01:00:00
#SBATCH --output=/home/USER/cp/8b/logs/jobc-%j.out
bash $HOME/cp/8b/bin/run_c.sh
