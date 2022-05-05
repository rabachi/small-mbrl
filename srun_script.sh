#!/bin/bash

#SBATCH -t 180:00:00               # max runtime is 180 hours (9 days)
#SBATCH --mem=10GB
#SBATCH --gres=cpu:0
#SBATCH -p cpu
#SBATCH --output=slurm_out/slurm_%A_%a.out
#SBATCH --array=0-9

#export LD_LIBRARY_PATH=/pkgs/cuda-9.2/lib64:$LD_LIBRARY_PATH
#export PATH=/pkgs/anaconda3/bin:$PATH
#

#conda activate py37

module load pytorch1.0-cuda9.0-python3.6

echo "${SLURM_ARRAY_TASK_ID}"

python deploy_slurm.py --deploy_num "${SLURM_ARRAY_TASK_ID}"
