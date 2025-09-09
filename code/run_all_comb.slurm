#!/bin/bash

#SBATCH --job-name=run_comb
#SBATCH --output=run_comb.out
#SBATCH --error=run_comb.err
#SBATCH --partition=studentkillable
#SBATCH --time=60
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=8000
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1

source /home/yandex/0368352201_BrainWS2025b/tomraz/anaconda3/etc/profile.d/conda.sh
conda activate /home/yandex/0368352201_BrainWS2025b/tomraz/my_env

cd /home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/code


echo "==== GPU INFO ===="
nvidia-smi


echo "==== START RUN ===="
/usr/bin/time -v python3 run_all_comb.py
echo "==== END RUN ===="

