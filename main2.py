from torch import initial_seed
# from src.logger import graph_single, graph_seeds
from src.training import MBRLLoop
from src.model import DirichletModel
from envs.basic_envs import GarnetMDP, State2MDP
from envs.chain import Chain
from envs.ring import Ring
from envs.randomfrozenlake import RandomFrozenLakeEnv
from envs.cliffwalking import CliffWalkingEnv
# from src.logger import CSVLogger
# from deploy import deploy_losses, deploy_MC2PS
import numpy as np
import hydra
import gym
import pdb
# from memory_profiler import profile
# SEED = 0

# @hydra.main(config_path="config/FrozenLake.yaml")#, config_name="chain")
@hydra.main(config_path="config", config_name="run_test")
# @profile
def experiment(args):

    env = hydra.utils.instantiate(args.env.env_setup)
    print(env)

    # if args.env_name == 'chain':
    #     env = Chain(5, 0.2, 0.99, args.seed)
    # elif args.env_name == 'FrozenLake4x4':
    #     env = RandomFrozenLakeEnv(args.seed, map_name=None) 
    # elif args.env_name == 'CliffWalking':
    #     env = CliffWalkingEnv(args.seed)

    nState = env.nState
    nAction = env.nAction

    print(f'Training {args.train_type}')
    
    if hasattr(env, 'initial_distribution'):
        initial_distribution = env.initial_distribution
    else:
        initial_distribution = np.zeros(nState)
        initial_distribution[0] = 1.
    
    if hasattr(env, 'discount'):
        discount = env.discount
    else:
        discount = 0.99

    print(args.train_type)
    data_dir = f'seed_{args.seed}'

    agent = DirichletModel(
        nState, 
        nAction, 
        int(args.seed),
        discount, 
        initial_distribution, 
        args.init_lambda,
        args.train_type.lambda_lr,
        args.train_type.policy_lr
        )

    trainer = MBRLLoop(env, agent, nState, nAction, initial_distribution, data_dir)

    if args.train_type.type == 'MC2PS': #this one is only offline
        trainer.training_then_sample(args)
        return

    trainer.training_loop(args)
    # trainer.training_then_sample()

if __name__=="__main__":
    experiment()