import argparse
from deploy import deploy_losses
from main2 import experiment
from envs.basic_envs import GarnetMDP, State2MDP
from envs.chain import Chain
from envs.ring import Ring
from envs.cliffwalking import CliffWalkingEnv
from envs.randomfrozenlake import RandomFrozenLakeEnv

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Slurm deployment')
    parser.add_argument('--deploy_num', type=int, default=0, help='The deployment number')
    args = parser.parse_args()

    deploy_argss = deploy_losses()
    assert args.deploy_num < len(deploy_argss), f"Invalid deployment number: {args.deploy_num}"

    deploy_args = deploy_argss[args.deploy_num]

    print(f"Launching {args.deploy_num}, {deploy_args}")

    env = RandomFrozenLakeEnv(deploy_args.seed, map_name=None) 
    # env = CliffWalkingEnv(deploy_args.seed)
    # env = Ring(deploy_args.seed)
    # env = Chain(5, 0.2, 0.9, deploy_args.seed)
    # env = State2MDP(deploy_args.seed)
    experiment(deploy_args, env)

    print(f"Finished {args.deploy_num}, {deploy_args}")
