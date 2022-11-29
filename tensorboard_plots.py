from src.logger import graph_single, graph_seeds, graph_seeds2
import os
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib as mpl
import matplotlib.pyplot as plt
import itertools
import argparse
import numpy as np
import pickle
import math
from tensorboard.backend.event_processing import event_accumulator
mpl.rcParams.update({'font.size': 12})
# mpl.rcParams['text.usetex'] = True

def get_data(path, prefix, envs_algs):
    output = ''
    slurm_dirs = next(os.walk(path))[1]
    for sd in slurm_dirs:
        if sd[:1] in '19':
            curr_dir = os.path.join(path, sd)
            hydra_dir = next(os.walk(curr_dir))[1]
            for hd in hydra_dir:
                if hd != '.hydra':
                    eventfile = os.listdir(os.path.join(curr_dir, hd))[0]
                    full_path = os.path.join(curr_dir, hd, eventfile)
                    # print(full_path)
                    ea = event_accumulator.EventAccumulator(full_path, size_guidance={event_accumulator.SCALARS: 0},)
                    ea.Reload()

                    names = hd.split('_')
                    env = names[0]
                    alg = names[1]
                    try:
                        output_dict = {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}
                        envs_algs[env][alg].append(output_dict)

                        if hd == "chain_pg-CE":
                            print(f"{sd}", end=' ')
                    except:
                        output += f"{sd} {hd} was not successful \n"
    print(output)
    with open(f'{prefix}saved_dictionary.pkl', 'wb') as f:
        pickle.dump(envs_algs, f)

def put_data_together(envs_algs, envs_algs_, envs_algs_aug26, envs_algs_aug27, envs_algs_aug28, envs, algs):
    for e in envs:
        for alg in algs:
            if e == 'CliffWalking-v0':
                if alg in ['pg', 'max-opt-cvar', 'max-opt', 'pg-cvar', 'CVaR']:
                    envs_algs[e][alg] = envs_algs_[e][alg] 
                elif alg in ['pg-CE', 'upper-cvar', 'upper-cvar-opt-cvar']:
                    envs_algs[e][alg] = envs_algs_aug28[e][alg] 
            elif e == 'doubleloop':
                if alg in ['pg', 'max-opt-cvar', 'max-opt', 'pg-cvar', 'CVaR']:
                    envs_algs[e][alg] = envs_algs_[e][alg]
                elif alg in ['pg-CE', 'upper-cvar', 'upper-cvar-opt-cvar']:
                    envs_algs[e][alg] = envs_algs_aug28[e][alg]
            elif e == 'chain':
                if alg in ['pg', 'max-opt-cvar', 'max-opt', 'pg-cvar', 'CVaR']:
                    envs_algs[e][alg] = envs_algs_[e][alg]
                elif alg in ['pg-CE', 'upper-cvar', 'upper-cvar-opt-cvar']:
                    envs_algs[e][alg] = envs_algs_aug28[e][alg]
            elif e == 'DistributionalShift-v0':
                if alg in ['pg', 'CVaR']:
                    envs_algs[e][alg] = envs_algs_[e][alg]
                elif alg in ['max-opt']:
                    envs_algs[e][alg] = envs_algs_aug26[e][alg]
                elif alg in ['max-opt-cvar','upper-cvar','pg-cvar']:
                    envs_algs[e][alg] = envs_algs_aug27[e][alg]
                elif alg in ['upper-cvar-opt-cvar', 'pg-CE']:
                    envs_algs[e][alg] = envs_algs_aug28[e][alg]
            elif e == 'FrozenLake4x4':
                if alg in ['pg', 'CVaR']:
                    envs_algs[e][alg] = envs_algs_[e][alg]
                elif alg in ['max-opt', 'upper-cvar']:
                    envs_algs[e][alg] = envs_algs_aug26[e][alg]
                elif alg in ['pg-cvar']:
                    envs_algs[e][alg] = envs_algs_aug27[e][alg]
                elif alg == 'upper-cvar-opt-cvar':
                    envs_algs[e][alg][:4] = envs_algs_aug27[e][alg][:4]
                    envs_algs[e][alg][4:8] = envs_algs_aug28[e][alg][4:8]
                elif alg in ['max-opt-cvar', 'upper-cvar-opt-cvar', 'pg-CE']:
                    envs_algs[e][alg] = envs_algs_aug28[e][alg]
            elif e == 'IslandNavigation-v0':
                if alg in ['max-opt', 'pg-CE', 'upper-cvar']:
                    envs_algs[e][alg] = envs_algs_aug26[e][alg]
                elif alg in ['max-opt-cvar','upper-cvar-opt-cvar','pg-cvar']:
                    envs_algs[e][alg] = envs_algs_aug27[e][alg]
                elif alg in ['pg', 'CVaR']:
                    envs_algs[e][alg] = envs_algs_aug28[e][alg]


if __name__ == "__main__":

    # get all runs 
    prefix = 'aug-28'
    # prefix26 = 'aug-26'
    # prefix27 = 'aug-27'
    # prefix28 = 'aug-28'
    # prefixes = ['', prefix26, prefix27, prefix28]
    path=f"/scratch/gobi1/abachiro/small_mbrl_results/exp/{prefix}"
    envs = [
        'CliffWalking-v0',
        'doubleloop',
        'chain',
        'DistributionalShift-v0',
        'FrozenLake4x4',
        'IslandNavigation-v0'
        ]

    algs = [
        'pg',
        'max-opt-cvar', 
        'max-opt',
        'pg-CE',
        'upper-cvar-opt-cvar',
        'upper-cvar',
        'pg-cvar',
        'CVaR'
    ]
    label_algs = {
        'pg': 'PG',
        'max-opt-cvar': 'Max Opt + constraint', 
        'max-opt': 'Max Opt',
        'pg-CE': 'PG with CE',
        'upper-cvar-opt-cvar': 'Upper CVaR + constraint',
        'upper-cvar': 'Upper CVaR',
        'pg-cvar': 'PG + constraint',
        'CVaR' : 'CVaR'
    }
    num_runs = 8
    scalars = [
        # 'av-V-model-pi', 
        'av-V-env-pi', 
        # 'v-alpha-quantile', 
        'cvar-alpha', 
        # 'cvar-constraint-lambda', 
        # 'grad-norm', 
        # 'best_iter', 
        # 'samples_taken'
    ]
    label_scalars = {
        # 'av-V-model-pi': 'Model Performance', 
        'av-V-env-pi': 'Environment Returns',
        'doubleloop-av-V-env-pi': 'Average Rewards',
        'chain-av-V-env-pi':'Average Rewards',
        # 'v-alpha-quantile', 
        'cvar-alpha': 'CVaR on Posterior', 
        # 'cvar-constraint-lambda': 'CVaR constraint', 
        # 'grad-norm': 'Gradient norm', 
        # 'best_iter': 'Best Iteration', 
        # 'samples_taken'
    }

    # envs_algs = {env: {alg: [] for alg in algs} for env in envs}
    envs_algs_ = {env: {alg: [] for alg in algs} for env in envs}
    envs_algs_aug26 = {env: {alg: [] for alg in algs} for env in envs}
    envs_algs_aug27 = {env: {alg: [] for alg in algs} for env in envs}
    envs_algs_aug28 = {env: {alg: [] for alg in algs} for env in envs}

    envs_plots = {env: {
                        'pg': plt.subplots(2, figsize=(6,8)), #for pg, pg-ce, cvar
                        'pg-cvar': plt.subplots(2, figsize=(6,8)), #for cvar, pg-cvar
                        'upper-cvar': plt.subplots(2, figsize=(6,8)), #for upperCvar, upperCvar+constraint
                        'max-opt': plt.subplots(2, figsize=(6,8)), #for maxopt, maxopt+constraint
                        }
                        for env in envs}
    # print(envs_algs)
    output = ''
    
    # get_data(path, '', envs_algs_)
    #get_data(path, 'aug26', envs_algs_aug26)
    # get_data(path, 'aug27', envs_algs_aug27)
    get_data(path, 'aug-28', envs_algs_aug28)
    
    with open(f'saved_dictionary.pkl', 'rb') as f:
        envs_algs_ = pickle.load(f)

    with open(f'aug-26saved_dictionary.pkl', 'rb') as f:
        envs_algs_aug26 = pickle.load(f)

    with open(f'aug-27saved_dictionary.pkl', 'rb') as f:
        envs_algs_aug27 = pickle.load(f)

    with open(f'aug-28saved_dictionary.pkl', 'rb') as f:
        envs_algs_aug28 = pickle.load(f)

    # put all the runs together
    envs_algs = {env: {alg: [] for alg in algs} for env in envs}
    put_data_together(envs_algs, envs_algs_, envs_algs_aug26, envs_algs_aug27, envs_algs_aug28, envs, algs)

    # print(envs_algs)
    colors_list =itertools.cycle(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    colors = {
        alg: next(colors_list) for alg in algs
    }
    print(colors)
    for env in envs:
        for alg in algs:
            if envs_algs[env][alg] == []:
                continue
            scalar_idx = 0
            for scalar in scalars:
                data_y = envs_algs[env][alg][0][scalar]
                min_idx = data_y.shape[0]
                for i in range(0, num_runs):
                    try:
                        data_y = envs_algs[env][alg][i][scalar]
                    except:
                        # print(seed)
                        print(i, env, alg, "didn't work")
                        # print(envs_algs[env][alg])
                    print(i, env, alg, data_y.shape[0])
                    if data_y.shape[0] < min_idx:
                        min_idx = data_y.shape[0]
                #calculate seed data for each "scalar"
                # print(env,alg, min_idx)
                if env == "FrozenLake4x4":
                    min_idx = min(100, min_idx)    
                else:
                    min_idx = min(100, min_idx)
                seed_vals = np.zeros((num_runs, min_idx))
                for seed in range(num_runs):
                    try:
                        seed_vals[seed] = envs_algs[env][alg][seed][scalar].value[:min_idx]
                    except:
                        # print(env, alg)
                        # print(seed)
                        print(seed, env, alg, " didn't work")
                scalar_mean = np.mean(seed_vals, axis=0)
                scalar_stderr = np.std(seed_vals, axis=0)/math.sqrt(num_runs)
                if alg in ['pg']:
                    dict_idx = ['pg', 'pg-cvar']
                elif alg in ['pg-CE']:
                    dict_idx = ['pg']
                elif alg in ['upper-cvar', 'upper-cvar-opt-cvar']:
                    dict_idx = ['upper-cvar']
                elif alg in ['max-opt', 'max-opt-cvar']:
                    dict_idx = ['max-opt']
                elif alg in ['pg-cvar']:
                    dict_idx = ['pg-cvar']
                elif alg in ['CVaR']:
                    dict_idx = ['pg'] #, 'pg-cvar']
                
                for didx in dict_idx:
                    envs_plots[env][didx][1][scalar_idx].plot(range(len(scalar_mean)), scalar_mean, label=label_algs[alg], color=colors[alg])
                    # envs_plots[env][1][scalar_idx].errorbar(range(len(scalar_mean)), scalar_mean, yerr=scalar_stderr, label=label_algs[alg])
                    envs_plots[env][didx][1][scalar_idx].fill_between(
                        range(len(scalar_mean)), 
                        scalar_mean - scalar_stderr, 
                        scalar_mean + scalar_stderr,
                        alpha=0.25,
                        color=colors[alg]
                        )
                    if env in ['chain', 'doubleloop'] and (scalar == 'av-V-env-pi'):
                        envs_plots[env][didx][1][scalar_idx].set_ylabel(label_scalars[f'{env}-{scalar}'])
                    else:
                        envs_plots[env][didx][1][scalar_idx].set_ylabel(label_scalars[scalar])
                        # envs_plots[env][didx][1][scalar_idx].set_xlabel()
                    if env in ['chain', 'doubleloop', 'CliffWalking-v0']:
                        envs_plots[env][didx][1][scalar_idx].set_xlabel('Environment Steps (x100)')
                    else:
                        envs_plots[env][didx][1][scalar_idx].set_xlabel('Environment Steps (x1000)')
                scalar_idx += 1

    for e in envs_plots.keys():
        for dict_idx in envs_plots[e].keys():
            envs_plots[e][dict_idx][1][0].legend(borderaxespad=1) #bbox_to_anchor=(1,0.5),
            envs_plots[e][dict_idx][1][0].grid()
            envs_plots[e][dict_idx][1][1].grid()
            if not os.path.isdir(f'images/{e}/'):
                os.mkdir(f'images/{e}/')
            envs_plots[e][dict_idx][0].savefig(f'images/{e}/{e}_{dict_idx}.pdf', bbox_inches='tight')
            
