#!/bin/sh

# Algs="pg pg-CE CVaR upper-cvar-opt-cvar max-opt-cvar pg-cvar upper-cvar max-opt"
# Algs="pg CVaR"
Algs="pg"


# echo "FrozenLake $alg"
# for i in {1..8}; do
#     sbatch base_script.sh "pg-CE" FrozenLake FrozenLake4x4 $i
# done

# for i in {1..8}; do
#     sbatch base_script.sh "pg-CE" SafetyGrid DistributionalShift-v0 $i
# done

# for alg in $Algs; do
#     echo "FrozenLake $alg"
#     for i in {1..8}; do
#         sbatch base_script.sh $alg FrozenLake FrozenLake4x4 $i
#     done
# done

# for alg in $Algs; do
#     echo "Chain $alg"
#     for i in {1..8}; do
#         sbatch base_script.sh $alg chain chain $i
#     done
# done

# for alg in $Algs; do
#     echo "CliffWalking $alg"
#     for i in {1..8}; do
#         sbatch base_script.sh $alg CliffWalking CliffWalking-v0 $i
#     done
# done

# for alg in $Algs; do
#     echo "DoubleLoop $alg"
#     for i in {1..8}; do
#         sbatch base_script.sh $alg DoubleLoop doubleloop $i
#     done
# done

# for alg in $Algs; do
#     echo "SafetyGrid DistributionalShift-v0 $alg"
#     for i in {1..1}; do
#         sbatch base_script.sh $alg SafetyGrid DistributionalShift-v0 False 0.1 reward
#     done
# done

for alg in $Algs; do
    echo "SafetyGrid DistributionalShift-v0 $alg"
    for i in {1..1}; do
        sbatch base_script.sh $alg SafetyGrid DistributionalShift-v0 False 0.1 reward_lr0.1
    done
done

# for alg in $Algs; do
#     echo "SafetyGrid IslandNavigation $alg"
#     for i in {1..8}; do
#         sbatch base_script.sh $alg SafetyGrid IslandNavigation-v0 $i
#     done
# done