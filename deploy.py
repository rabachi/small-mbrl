import argparse  # For args
import copy  # For copying new args
import pdb

def make_parser():
    """Make an argument parser for controlling experiments via command line.
    :return: An argument parser.
    """
    parser = argparse.ArgumentParser(description='todo')
    # General arguments that shouldn't affect the results
    parser.add_argument('--verbose', default=False,
                        help='If we should print diagnostic info.')
    parser.add_argument('--save_data', default=True,
                        help='If we should store diagnostic info.')
    parser.add_argument('--save_dir', default='/scratch/gobi1/abachiro/small-mbrl/data',
                        help='Where to store diagnostic info.')

    parser.add_argument('--policy_lr', default=0.1,
                        help='Policy learning rate.')
    parser.add_argument('--model_lr', default=0.01,
                        help='Model learning rate.')
    parser.add_argument('--num_samples_plan', default=40,
                        help='')
    parser.add_argument('--num_eps_eval', default=40,
                        help='')
    parser.add_argument('--num_eps', default=100,
                        help='')
    parser.add_argument('--train_type', default='pg',
                        help='')
    parser.add_argument('--traj_len', default=10,
                        help='')
    parser.add_argument('--risk_threshold', default=0.1,
                        help='')
    parser.add_argument('--k_value', default=4,
                        help='')              
    parser.add_argument('--log_freq', default=1,
                        help='How often to evaluate the policy on true env.')   

    #MC2PS specific args
    parser.add_argument('--batch_size', default=100)
    parser.add_argument('--num_models', default=5)
    parser.add_argument('--num_discounts', default=9)
    parser.add_argument('--sigma', default='CVaR')
    parser.add_argument('--eps_rel', default=0.1)
    parser.add_argument('--significance_level', default=0.1)

    # Other arguments
    parser.add_argument('--seed', default=0, help='The seed to use.')
    parser.add_argument('--temperature', default=1.0, help='The temperature.')  # TODO: Separate one for model_softmax?
    return parser


def deploy_test():
    """

    :return:
    """
    parser = make_parser()
    args, _ = parser.parse_known_args()
    argss = []
    epochs = [i for i in range(10)]
    for epoch in epochs:
        new_args = copy.deepcopy(args)
        new_args.num_points = epoch
        new_args.save_sub_dir = new_args.save_dir + '/' + get_id(new_args)
        argss += [new_args]
    return argss


def deploy_single():
    """

    :return:
    """
    parser = make_parser()
    args, _ = parser.parse_known_args()
    argss = []
    epochs = [100]
    for epoch in epochs:
        new_args = copy.deepcopy(args)
        new_args.num_points = epoch
        new_args.save_sub_dir = new_args.save_dir + '/' + get_id(new_args)
        argss += [new_args]
    return argss

def get_id(args):
    """
    :param args:
    :return:
    """
    # TODO: Make these in a sensible order, so readable
    args_id = ''
    args_id += f'train_type={args.train_type}_'
    args_id += f'seed={args.seed}'
    args_id += f'risk_threshold={args.risk_threshold}_'
    args_id += f'policy_lr={args.policy_lr}_'
    args_id += f'num_samples_plan={args.num_samples_plan}_'
    # args_id += f'num_eps={args.num_eps}_'
    args_id += f'traj_len={args.traj_len}_'
    args_id += f'k_value={args.k_value}_'
    args_id += f'log_freq={args.log_freq}_'
    args_id += f'num_eps_eval={args.num_eps_eval}_'
    return args_id


def deploy_losses():
    """

    :return:
    """
    parser = make_parser()
    args, _ = parser.parse_known_args()
    argss = []
    train_types = [
        # 'VaR-sigmoid',
        # 'VaR-delta',
        # 'max-opt', #opt only
        'upper-cvar', #opt only
        # 'psrl-opt-cvar', #opt + constraint
        # 'max-opt-cvar', #opt + constraint
        # 'optimistic-psrl-opt-cvar', #incorrect
        'upper-cvar-opt-cvar', #opt + constraint
        # 'CVaR', #risk only
        # 'grad-risk-eval',
        # 'regret-CVaR',
        # 'robust-DP',
        # 'k-of-N',
        # 'pg', 
        # 'pg-CE',
        # 'psrl' #opt only
    ]
    # train_types = [
    #     'max-opt', #opt only
    #     'upper-cvar', #opt only
    #     'psrl-opt-cvar', #opt + constraint
    #     'max-opt-cvar', #opt + constraint
    #     'upper-cvar-opt-cvar', #opt + constraint
    #     # 'CVaR', #risk only
    #     # 'pg', 
    #     # 'pg-CE',
    #     'psrl' #opt only
    # ]
    seeds = range(5)#[1,2,4]##range(15) #[17, 3, 11, 19, 23]#13, 2, 0, 5, 123,  #[23]#698793, 47, 4139,  48784127, 41]#, 17, 13, 698793, 47, 4139]
    for train_type in train_types:
        for seed in seeds:
            new_args = copy.deepcopy(args)
            new_args.model_lr = 0.01
            new_args.policy_lr = 0.01 if train_type in ['max-opt-cvar', 'upper-cvar-opt-cvar'] else 0.1
        #     new_args.policy_lr = 0.01 if train_type in ['psrl-opt-cvar',
        # 'max-opt-cvar',
        # 'optimistic-psrl-opt-cvar',
        # 'upper-cvar-opt-cvar'] else 0.1
            new_args.num_samples_plan = 10
            new_args.num_eps = 10 if train_type=='VaR-delta' else 150
            new_args.train_type = train_type
            new_args.traj_len = 100
            new_args.risk_threshold = 0.1
            new_args.k_value = 4
            new_args.log_freq = 1
            new_args.num_eps_eval = 10
            new_args.seed = seed
            # new_args.save_sub_dir = new_args.save_dir + '/' + 'state2mdp-' + get_id(new_args)
            # new_args.save_sub_dir = new_args.save_dir + '/' + 'cliffwalking-' + get_id(new_args)
            # new_args.save_sub_dir = new_args.save_dir + '/' + 'Frozenlake-constraint-1-midtrainsteps-100-' + get_id(new_args)
            new_args.save_sub_dir = new_args.save_dir + '/' + 'Frozenlake-constraint-1-midtrainsteps-100-upper0.4-' + get_id(new_args)
            argss += [new_args]
    return argss


def deploy_MC2PS():
    parser = make_parser()
    args, _ = parser.parse_known_args()
    argss = []
    train_types = ['MC2PS']
    seeds = [17]#, 13, 0, 5, 123456789] #[23]#698793, 47, 4139,  48784127, 41]#, 17, 13, 698793, 47, 4139]
    for train_type in train_types:
        for seed in seeds:
            new_args = copy.deepcopy(args)
            new_args.batch_size = 100
            new_args.num_models = 5
            new_args.num_discounts = 9
            new_args.sigma = 'CVaR'
            new_args.eps_rel = 0.1
            new_args.significance_level = 0.1
            new_args.risk_threshold = 0.1
            new_args.train_type = train_type
            new_args.seed = seed
            new_args.save_sub_dir = new_args.save_dir + '/' + get_id(new_args)
            argss += [new_args]
    return argss