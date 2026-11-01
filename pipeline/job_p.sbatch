#!/bin/bash
#SBATCH --job-name=cp-8bp2
#SBATCH --partition=hackathon
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=224
#SBATCH --mem=0
#SBATCH --time=10:00:00
#SBATCH --output=/home/eai-jjppv/cp/8b/logs/jobq-%j.out
bash $HOME/cp/8b/bin/run_p.sh
