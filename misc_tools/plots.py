from FL_deploy import deploy_losses, deploy_MC2PS
from src.logger import graph_single, graph_seeds

def make_plots(deploy_argss, env_name):
    # deploy_argss = deploy_losses()
    graph_seeds(deploy_argss, env_name, 'av-V-env-pi')
    # graph_seeds(deploy_argss, env_name, 'av-V-model-pi')
    # graph_seeds(deploy_argss, env_name, 'v-alpha-quantile')
    # graph_seeds(deploy_argss, env_name, 'grad-norm')
    graph_seeds(deploy_argss, env_name, 'cvar-alpha')
    graph_seeds(deploy_argss, env_name, 'cvar-constraint-lambda')

if __name__=="__main__":
    do_experiment = False
    env_names = ['FL']#['FL', 'CW']
    for env_name in env_names:
        deploy_argss = deploy_losses(env_name, -1.)
        # # deploy_argss = deploy_MC2PS()
        make_plots(deploy_argss, env_name)
        
        # deploy_args = deploy_argss[0]

        # if env_name == 'chain':
        #     env = Chain(5, 0.2, 0.99, deploy_args.seed)
        # elif env_name == 'FL':
        #     env = RandomFrozenLakeEnv(deploy_args.seed, map_name=None) 
        # elif env_name == 'CW':
        #     env = CliffWalkingEnv(deploy_args.seed)
        # # env = CliffWalkingEnv(deploy_args.seed)
        # # env = Ring(deploy_args.seed)
        # # env = Chain(5, 0.2, 0.9, deploy_args.seed)
        # # env = State2MDP(deploy_args.seed)
        # if do_experiment:
        #     experiment(deploy_args, env, 30.)

        # env_name = env.get_name()
        # make_plots_single(deploy_argss[0], env_name, 'reset')