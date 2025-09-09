#!/bin/bash

#SBATCH --job-name=try_combs          # job name
#SBATCH --output=try_combs.out        # standard output file
#SBATCH --error=try_combs.err         # error output file
#SBATCH --partition=studentkillable   # partition to run on
#SBATCH --time=1-00:00:00             # 24 hours
                    
#SBATCH --nodes=1                     # number of nodes
#SBATCH --ntasks=1                    # number of tasks
#SBATCH --mem=8000                    # memory (MB)
#SBATCH --cpus-per-task=4             # number of CPU cores
#SBATCH --gres=gpu:1                  # number of GPUs

# activate conda environment
source /home/yandex/0368352201_BrainWS2025b/tomraz/anaconda3/etc/profile.d/conda.sh
conda activate /home/yandex/0368352201_BrainWS2025b/tomraz/my_env

# move to code directory
cd /home/yandex/0368352201_BrainWS2025b/tomraz/dFC_DimReduction/code

echo "==== GPU INFO ===="
nvidia-smi

echo "==== START RUN ===="
/usr/bin/time -v python3 try_combs.py
echo "==== END RUN ===="

