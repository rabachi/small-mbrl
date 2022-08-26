from src.logger import graph_single, graph_seeds, graph_seeds2
import os

def make_plots(deploy_argss, env_name):
    # deploy_argss = deploy_losses()
    graph_seeds(deploy_argss, env_name, 'av-V-env-pi')
    # graph_seeds(deploy_argss, env_name, 'av-V-model-pi')
    # graph_seeds(deploy_argss, env_name, 'v-alpha-quantile')
    # graph_seeds(deploy_argss, env_name, 'grad-norm')
    graph_seeds(deploy_argss, env_name, 'cvar-alpha')
    graph_seeds(deploy_argss, env_name, 'cvar-constraint-lambda')

def make_plots_single(fid, env_name, train_type):
    # deploy_argss = deploy_losses()
    graph_single('av-V-env-pi', fid, env_name, train_type)
    graph_single('av-V-model-pi', fid, env_name, train_type)
    # graph_single(deploy_argss, env_name, 'v-alpha-quantile', fid)
    graph_single('grad-norm', fid, env_name, train_type)
    graph_single('cvar-alpha', fid, env_name, train_type)
    graph_single('cvar-constraint-lambda', fid, env_name, train_type)

def make_plots_seeds(main_dirs, num_seeds):
    graph_seeds2(main_dirs, num_seeds, 'av-V-env-pi')
    graph_seeds2(main_dirs, num_seeds, 'av-V-model-pi')
    graph_seeds2(main_dirs, num_seeds, 'grad-norm')
    graph_seeds2(main_dirs, num_seeds, 'cvar-alpha')
    graph_seeds2(main_dirs, num_seeds, 'cvar-constraint-lambda')

if __name__ == "__main__":
    path_to_exp = '/scratch/gobi1/abachiro/small_mbrl_results/exp/'
    # exps = ['max-opt', 'max-opt-cvar', 'upper-cvar','upper-cvar-opt-cvar', 'pg', 'psrl', 'CVaR']
    exps = ['pg', 'CVaR']#['upper-cvar-opt-cvar', 'CVaR', 'max-opt-cvar']
    env = 'FrozenLake4x4_cvarfirst'
    main_dirs = []
    for exp in exps:
        if exp in ['upper-cvar-opt-cvar', 'max-opt-cvar']:
            env = 'FrozenLake4x4_cvarfirst'
        else:
            env = 'FrozenLake4x4'
        main_dirs.append(os.path.join(path_to_exp, f'{exp}_{env}'))
    num_seeds = 5
    make_plots_seeds(
        main_dirs,
        num_seeds
    )
    # print(main_dirs)
    make_plots_single(
        [
            # '/scratch/gobi1/abachiro/small_mbrl_results/exp/default_1/upper-cvar-opt-cvar_FrozenLake4x4_constraint-15.0_midtrainsteps500',
            # '/scratch/gobi1/abachiro/small_mbrl_results/exp/default_1/psrl-opt-cvar_FrozenLake4x4_constraint-20.0_midtrainsteps1',
            # '/scratch/gobi1/abachiro/small_mbrl_results/exp/default_1/max-opt-cvar_FrozenLake4x4_constraint-20.0_midtrainsteps200',
            # '/scratch/gobi1/abachiro/small_mbrl_results/exp/default_1/max-opt_FrozenLake4x4_constraint-20.0_midtrainsteps200',
            '/scratch/gobi1/abachiro/small_mbrl_results/exp/upper-cvar-opt-cvar_FrozenLake4x4_cvarfirst/seed_1',
            '/scratch/gobi1/abachiro/small_mbrl_results/exp/pg_FrozenLake4x4_cvarfirst/seed_1',
            '/scratch/gobi1/abachiro/small_mbrl_results/exp/CVaR_FrozenLake4x4_cvarfirst/seed_1'
        ], 
        'FL', 
        [
            'upper-cvar-opt-cvar',
            # 'psrl-opt-cvar',
            # 'max-opt-cvar',
            # 'max-opt',
            # 'upper-cvar',
            'pg',
            'CVaR'
        ])
    # make_plots_single('/scratch/gobi1/abachiro/small_mbrl_results/exp/default_1/upper-cvar_FrozenLake4x4_constraint-15_midtrainsteps200', 'FL', 'upper-cvar')