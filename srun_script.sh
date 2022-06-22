#!/bin/bash
#SBATCH -t 180:00:00             # max runtime is 180 hours (9 days)
#SBATCH --mem=8GB
#SBATCH --gres=cpu:0
#SBATCH -p gpu
#SBATCH --output=slurm_out/slurm_%A_%a.out
#SBATCH --array=0-4

#export LD_LIBRARY_PATH=/pkgs/cuda-9.2/lib64:$LD_LIBRARY_PATH
# export PATH=/pkgs/anaconda3/bin:$PATH

conda activate viper
module load cuda-11.1
# module load pytorch1.7.1-cuda11.0-python3.6

echo "${SLURM_ARRAY_TASK_ID}"

python main2.py train_type='upper-cvar' env='FrozenLake' hydra_dir_end=${SLURM_JOB_ID} seed=${SLURM_ARRAY_TASK_ID}