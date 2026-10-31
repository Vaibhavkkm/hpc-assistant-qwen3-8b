#!/bin/bash
#SBATCH --job-name=cp-8bf
#SBATCH --partition=hackathon
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=112
#SBATCH --mem=900G
#SBATCH --time=03:00:00
#SBATCH --output=/home/eai-jjppv/cp/8b/logs/jobf-%j.out
bash $HOME/cp/8b/bin/run_final.sh
