#!/bin/bash
#SBATCH --job-name=cp-8b2
#SBATCH --partition=hackathon
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=224
#SBATCH --mem=0
#SBATCH --time=1-06:00:00
#SBATCH --output=/home/USER/cp/8b/logs/job2-%j.out
bash $HOME/cp/8b/bin/run_8b.sh
