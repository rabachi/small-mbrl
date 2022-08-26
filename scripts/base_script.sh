#!/bin/bash
#SBATCH -t 180:00:00             # max runtime is 180 hours (9 days)
#SBATCH --mem=8GB
#SBATCH --gres=cpu:0
#SBATCH -p gpu
#SBATCH --output=../slurm_out/slurm_%A_%a.out

source ~/.bashrc
conda activate viper

export PATH=/pkgs/cuda-11.3/bin:/h/abachiro/.conda/envs/viper/bin:/h/abachiro/.conda/envs/latplan/bin:/h/abachiro/.local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/usr/games:/usr/local/games:/snap/bin:/h/abachiro/anaconda3/bin:/h/abachiro/.roswell/bin

export LD_LIBRARY_PATH=/pkgs/cuda-11.3/lib64::/h/abachiro/.mujoco/mujoco/bin:/usr/lib/nvidia

echo "${SLURM_ARRAY_TASK_ID}"

python ../main2.py \
    train_type=$1 \
    env=$2 \
    env.env_id=$3 \
    seed=$4 \
    hydra.run.dir=/scratch/gobi1/abachiro/small_mbrl_results/exp/$SLURM_JOB_ID
