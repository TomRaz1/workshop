#!/bin/bash
#SBATCH --job-name=run_all_comb
#SBATCH --output=../logs/output_%j.txt
#SBATCH --error=../logs/error_%j.txt
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G

echo "Start run: $(date)"
python3 /home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/code/run_all_comb.py
echo "End run: $(date)"
