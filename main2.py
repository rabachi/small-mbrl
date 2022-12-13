# from src.logger import graph_single, graph_seeds
from src.training import MBRLLoop
from src.model import DirichletModel

from envs.env import setup_environment
# from src.logger import CSVLogger
# from deploy import deploy_losses, deploy_MC2PS
import numpy as np
import hydra
import jax

import copy
import os


@hydra.main(config_path="config", config_name="run_test")
def experiment(args):

    env = setup_environment(
        args.env.env_setup,
        args.env.env_type,
        args.env.env_id,
        args.env.norm_reward,
        args.seed,
    )
    # env = hydra.utils.instantiate(args.env.env_setup)
    print(env)

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
    data_dir = f'{args.env.env_id}_{args.train_type.type}_{args.optimization_type}_incorrectpriors{args.use_incorrect_priors}'

    agent = DirichletModel(
        nState, 
        nAction, 
        int(args.seed),
        discount, 
        initial_distribution, 
        args.init_lambda,
        args.train_type.lambda_lr,
        args.train_type.policy_lr,
        args.use_incorrect_priors
    )
    
    p_params_baseline, baseline_true_perf = 0, 0
    if args.train_type.type in ["upper-cvar-opt-cvar"]:
        p_params_baseline, baseline_true_perf = get_baseline_policy(
            env, 
            args, 
            nState, 
            nAction, 
            discount, 
            initial_distribution
        )
    
    trainer = MBRLLoop(
        env, 
        agent, 
        nState, 
        nAction, 
        initial_distribution, 
        data_dir,
        p_params_baseline,
        baseline_true_perf,
        seed=int(args.seed),
        wandb_entity=args.wandb_entity,
    )

    if args.train_type.type == 'MC2PS': #this one is only offline
        trainer.training_then_sample(args)
        return

    if args.train_type.type == "Q-learning":
        trainer.Q_learning(args, discount)
        return

    trainer.training_loop(args)
    # trainer.training_then_sample()

def get_baseline_policy(env, args, nState, nAction, discount, initial_distribution):
    #TODO:have to figure out how to do this with hydra
    #check if baseline file exists for env in directory
    filepath = f"baseline_policies_{args.env.env_id}.npy"
    if os.path.exists(filepath):
        # if yes, read file and return those params
        p_params_baseline = np.load(filepath)
        baseline_true_perf = np.load(f'{filepath[:-4]}_true_perf.npy')
    else:
        # if no, train policy to a mid-point on env, then save policy_params to file and return it as well (can run a new agent and MBRLLoop just for PG or something?)
        agent = DirichletModel(
            nState, 
            nAction, 
            int(args.seed),
            discount, 
            initial_distribution, 
            args.init_lambda,
            args.train_type.lambda_lr,
            args.train_type.policy_lr,
            args.use_incorrect_priors
        )
        
        data_dir = f'baseline_{args.env.env_id}_{args.train_type.type}'

        trainer = MBRLLoop(
            env, 
            agent, 
            nState, 
            nAction, 
            initial_distribution, 
            data_dir,
            None,
            None
        )

        new_args = copy.deepcopy(args)
        new_args.num_eps = 200
        new_args.train_type.type = 'pg'
        new_args.train_type.policy_lr = 0.1
        new_args.train_type.mid_train_steps = 50

        print('training_loop')
        baseline_true_perf = trainer.training_loop(new_args)

        p_params_baseline = agent.policy.get_params()
        np.save(filepath, p_params_baseline)
        np.save(f'{filepath[:-4]}_true_perf.npy', baseline_true_perf)
    
    return p_params_baseline, baseline_true_perf

if __name__=="__main__":
    jax.config.update('jax_platform_name', 'cpu')
    experiment()