#!/bin/bash
#SBATCH --job-name=cp-8bn
#SBATCH --partition=hackathon
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=224
#SBATCH --mem=0
#SBATCH --time=20:00:00
#SBATCH --output=/home/eai-jjppv/cp/8b/logs/jobn-%j.out
bash $HOME/cp/8b/bin/run_night.sh
