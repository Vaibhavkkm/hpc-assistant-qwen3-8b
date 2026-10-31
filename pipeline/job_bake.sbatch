#!/bin/bash
#SBATCH --job-name=cp-8b
#SBATCH --partition=hackathon
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=224
#SBATCH --mem=0
#SBATCH --time=1-00:00:00
#SBATCH --output=/home/eai-jjppv/cp/8b/logs/job-%j.out
bash $HOME/cp/8b/bin/bakeoff.sh
