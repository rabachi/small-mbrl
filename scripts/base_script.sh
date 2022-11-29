#!/bin/bash
#SBATCH -t 180:00:00             # max runtime is 180 hours (9 days)
#SBATCH -p cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G
#SBATCH --output=../slurm_out/aug-28-slurm_%A_%a.out

source ~/.bashrc
conda activate viper

export PATH=/pkgs/cuda-11.3/bin:$PATH

export LD_LIBRARY_PATH=/pkgs/cuda-11.3/lib64:$LD_LIBRARY_PATH

echo "${SLURM_ARRAY_TASK_ID}"

python ../main2.py \
    train_type=$1 \
    env=$2 \
    env.env_id=$3 \
    seed=$4 \
    use_incorrect_priors=True \
    hydra.run.dir=/scratch/gobi1/abachiro/small_mbrl_results/exp/aug-28/$SLURM_JOB_ID
