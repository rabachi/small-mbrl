from torch import initial_seed
from src.logger import graph_single, graph_seeds
from src.training import MBRLLoop
from src.model import DirichletModel
# from envs.basic_envs import GarnetMDP, State2MDP
import envs
# from src.logger import CSVLogger
from deploy import deploy_losses, deploy_MC2PS
import numpy as np
import gym
# SEED = 0

def experiment(args):
    # env setup
    nState = 2
    nAction = 2
    # Gparams = (2,2)
    # env = GarnetMDP(nState, nAction, Gparams)
    env = envs.basic_envs.State2MDP(args.seed)
    # env_name = 'FrozenLake-v1'
    # env = gym.make(env_name)
    # nState = env.observation_space.n
    # nAction = env.action_space.n

    print(f'Training {args.train_type}')
    
    initial_distribution = env.initial_distribution
    discount = env.discount

    agent = DirichletModel(nState, nAction, int(args.seed), discount, initial_distribution)

    #fix this mess after!
    args_ = {
        'nState' : nState,
        'nAction': nAction,
        'initial_distribution' : initial_distribution, #this might be a bad idea for all envs ... 
        'model_lr' : args.model_lr, 
        'policy_lr' : args.policy_lr,
        'num_samples_plan' : args.num_samples_plan,
        'num_eps' : args.num_eps,
        'train_type' : args.train_type,
        'traj_len' : args.traj_len,
        'risk_threshold' : args.risk_threshold,
        'k_value' : args.k_value,
        'log_freq' : args.log_freq,
        'num_eps_eval' : args.num_eps_eval,
        'save_sub_dir' : args.save_sub_dir,
        'seed' : int(args.seed),
        #MC2PS args
        'batch_size' : int(args.batch_size),
        'num_models' : int(args.num_models),
        'num_discounts' : int(args.num_discounts),
        'sigma' : args.sigma,
        'eps_rel' : args.eps_rel,
        'significance_level' : args.significance_level
        }
    trainer = MBRLLoop(env, agent, **args_)
    # trainer.training_loop()
    trainer.training_then_sample()

def make_plots(deploy_argss):
    # deploy_argss = deploy_losses()
    graph_seeds(deploy_argss, 'av-V-env-pi')
    graph_seeds(deploy_argss, 'av-V-model-pi')
    graph_seeds(deploy_argss, 'v-alpha-quantile')
    graph_seeds(deploy_argss, 'cvar-alpha')

def experiment_manual(args):
    # env setup
    nState = 2
    nAction = 2
    # Gparams = (2,2)
    # env = GarnetMDP(nState, nAction, Gparams)
    env = State2MDP(args.seed)

    train_types = ['CVaR', 'k-of-N', 'pg-CE', 'pg', 'VaR-sigmoid']#['CVaR', 'k-of-N', 'VaR-sigmoid', 'VaR-delta', 'pg']
    train = True
    # logger = CSVLogger(
    #     fieldnames={'av_V_p_pi':0},
    #     filename='./log.csv'
    # )
    if train:
        for train_type in train_types:
            print(f'Training {train_type}')
            agent = DirichletModel(nState, nAction, args.seed)

            args = {
                'model_lr' : 0.01, 
                'policy_lr' : 0.1,
                'num_samples_plan' : 40,
                'num_eps' : 10 if train_type=='VaR-delta' else 100,
                'train_type' : train_type,
                'traj_len' : 10,
                'risk_threshold' : 0.1,
                'k_value' : 4,
                'log_freq' : 1,
                'num_eps_eval' : 10,
                }
            trainer = MBRLLoop(env, agent, **args)

            # trainer.training_loop()
            trainer.training_then_sample()

    # graph_single(train_types, 'av-V-env-pi')
    # graph_single(train_types, 'av-V-model-pi')
    # V_p_pi = agent.policy_evaluation((P_samp, R_samp), p_params)
    # logger.plot_histogram()#V_p_pi, p_params) #put in logger
    # logger.plot_performance()#training_values) #put in logger

if __name__=="__main__":
    deploy_argss = deploy_losses()
    # deploy_argss = deploy_MC2PS()

    # deploy_args = deploy_argss[0]
    # experiment(deploy_args)
    make_plots(deploy_argss)