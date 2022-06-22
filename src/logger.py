import csv
import os
from re import A, L
import matplotlib as mpl
from matplotlib.font_manager import FontProperties  # For changing matplotlib arguments
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams.update({'font.size': 15})
import matplotlib.pyplot as plt  # For graphing
import numpy as np
import copy
from FL_deploy import get_id
import ast
import itertools
from io import StringIO

class CSVLogger():
    def __init__(self, fieldnames, filename='log.csv'):
        self.filename = filename
        self.csv_file = open(filename, 'w')

        self.writer = csv.DictWriter(self.csv_file, fieldnames=fieldnames)
        self.writer.writeheader()

        self.csv_file.flush()

    def writerow(self, row):
        self.writer.writerow(row)
        self.csv_file.flush()

    def close(self):
        self.csv_file.close()


def load_from_csv(path):
    """
    :param path:
    :param do_train:
    :param do_val:
    :param do_test:
    :return:
    """
    # log_loc = 'log.csv'
    print(path)
    with open(path) as csvfile:
        # data = csvfile.read()
        # data = data.replace('\x00','?')
        # reader = csv.DictReader(StringIO(data), skipinitialspace=True)
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        data = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                data[name].append(row[name])
    index = 1500
    # data['av-V-model-pi'] = np.array(
    #     list(
    #         [ast.literal_eval(data['av-V-model-pi'][i])[-1] for i in range(len(data['av-V-model-pi']
    #         ))]))
    # data['grad-norm'] = np.array(
    #     list(itertools.chain.from_iterable(
    #         [ast.literal_eval(data['grad-norm'][i]) for i in range(len(data['grad-norm']
    #         ))]))
    #         )
    # data['grad-norm'] = np.array(
    #     list([ast.literal_eval(data['grad-norm'][i])[-1] for i in range(len(data['grad-norm']
    #         ))]))
    # data['av-V-model-pi'] = np.array([float(i.strip()) for i in data['av-V-model-pi']])[:index]
    new_data = np.zeros(len(data['av-V-model-pi']))
    
    for i in range(len(data['av-V-model-pi'])):
        s = data['av-V-model-pi'][i]
        try:
            new_data[i] = float(s)
        except:
            continue
            # print(str.split(s, '-'))

    data['av-V-model-pi'] = new_data
    
    new_data = np.zeros(len(data['av-V-env-pi']))
    
    for i in range(len(data['av-V-env-pi'])):
        s = data['av-V-env-pi'][i]
        try:
            new_data[i] = float(s)
        except:
            continue
            # print(str.split(s, '-'))

    data['av-V-env-pi'] = new_data
    
    new_data = np.zeros(len(data['cvar-alpha']))
    
    for i in range(len(data['cvar-alpha'])):
        s = data['cvar-alpha'][i]
        try:
            new_data[i] = float(s)
        except:
            continue
            # print(str.split(s, '-'))

    data['cvar-alpha'] = new_data
    
    #np.array([float(i.strip()) for i in data['av-V-env-pi']])[:index]

    data['ep'] = np.array([int(i.strip()) for i in data['ep']])
    # data['v-alpha-quantile'] = np.array([float(i.strip()) for i in data['v-alpha-quantile']])[:index]
    # data['cvar-alpha'] = np.array([float(i.strip()) for i in data['cvar-alpha']])[:index]
    # try:
    #     # print(data['cvar-alpha'])
    #     if 'psrl' in path:
    #         lst = []
    #         for i in range(len(data['cvar-alpha'])):
    #             try:
    #                 lst.append(ast.literal_eval(data['cvar-alpha'][i])[-1])
    #             except:
    #                 print(ast.literal_eval(data['cvar-alpha'][i]))

    #         data['cvar-alpha'] = np.array(lst)
    #         # data['cvar-alpha'] = np.array(
    #         # list(
    #         #     [ast.literal_eval(data['cvar-alpha'][i]) for i in range(len(data['cvar-alpha']
    #         #     ))]))
    #         # print(data['cvar-alpha'])
    #     else:
    #         data['cvar-alpha'] = np.array(
    #         list(
    #             [ast.literal_eval(data['cvar-alpha'][i])[-1] for i in range(len(data['cvar-alpha']
    #             ))]))
    # except:
    #     data['cvar-alpha'] = np.array(
    #     list(itertools.chain.from_iterable(
    #         [ast.literal_eval(data['cvar-alpha'][i]) for i in range(len(data['cvar-alpha']
    #         ))])))
            
    # data['cvar-alpha'] = np.array(
    #     list([ast.literal_eval(data['cvar-alpha'][i])[-1] for i in range(len(data['cvar-alpha']
    #         ))]))
    # new_data = np.zeros(len(data['cvar-alpha']))
    # for i in range(len(data['cvar-alpha'])):
    #     s = data['cvar-alpha'][i]
    #     try:
    #         new_data[i] = float(s)
    #     except:
    #         print(str.split(s, '-'))
    # data['cvar-alpha'] = new_data
    # data['cvar-constraint-lambda'] = np.array(
    #     list(itertools.chain.from_iterable(
    #         [ast.literal_eval(data['cvar-constraint-lambda'][i]) for i in range(len(data['cvar-constraint-lambda']
    #         ))])))
    new_data = np.zeros(len(data['cvar-constraint-lambda']))
    for i in range(len(data['cvar-constraint-lambda'])):
        s = data['cvar-constraint-lambda'][i]
        try:
            new_data[i] = float(s)
        except:
            print(str.split(s, '-'))
    data['cvar-constraint-lambda'] = new_data

    new_data = np.zeros(len(data['grad-norm']))
    for i in range(len(data['grad-norm'])):
        s = data['grad-norm'][i]
        try:
            new_data[i] = float(s)
        except:
            print(str.split(s, '-'))
    data['grad-norm'] = new_data
    # data['cvar-constraint-lambda'] = np.array([float(i.strip()) for i in data['cvar-constraint-lambda']])[:index]
    return data

def init_ax(fontsize=12, nrows=1, ncols=1):
    """

    :param fontsize:
    :param nrows:
    :param ncols:
    :return:
    """
    font = {'family': 'Times New Roman'}
    mpl.rc('font', **font)
    # mpl.rcParams['legend.fontsize'] = fontsize
    mpl.rcParams['axes.labelsize'] = fontsize
    mpl.rcParams['xtick.labelsize'] = fontsize
    mpl.rcParams['ytick.labelsize'] = fontsize
    mpl.rcParams['axes.grid'] = False

    # fig = plt.figure(figsize=(6.4, 4.8))  # / np.sqrt(nrows), 4.8 * nrows / np.sqrt(nrows)))
    # fig = plt.figure()
    fig = plt.figure()#figsize=(4.0, 6.0))
    axs = [fig.add_subplot(nrows, ncols, i + 1) for i in range(nrows * ncols)]
    for ax in axs:
        ax.tick_params(axis='x', which='both', bottom=False, top=False)
        ax.tick_params(axis='y', which='both', left=False, right=False)
        ax.grid(True)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
    return fig, axs

def setup_ax(ax, do_legend=True, alpha=0.0, fontsize=12, legend_loc=None, handlelength=None):
    """

    :param ax:
    :param do_legend:
    :param alpha:
    :param fontsize:
    :param legend_loc:
    :param handlelength:
    :return:
    """
    if do_legend:
        ax.legend(fancybox=True, borderaxespad=0.5, framealpha=alpha, fontsize=fontsize,
                  loc=legend_loc, handlelength=handlelength)
    # ax.tick_params(axis='x', which='both', bottom=False, top=False)
    # ax.tick_params(axis='y', which='both', left=False, right=False)
    # ax.grid(False)
    # ax.spines['right'].set_visible(False)
    # ax.spines['top'].set_visible(False)
    # plt.tight_layout()
    return ax


def graph_single(y_type, fids, env_name, train_types):
    """
    :param args:
    :param x_type:
    :param y_type:
    :return:
    """
    fig, axs = init_ax()
    idx = 0
    for fid in fids:
        data = load_from_csv(fid)

        axs[0].plot(data['ep'], data[y_type], label=f'{train_types[idx]}')
            
        axs[0].set_xlabel('Eps'), axs[0].set_ylabel(f'Policy performance {y_type}')    
        plt.legend(bbox_to_anchor=(1,0.5))#,fontsize='large')
        # axs[0].legend()
        idx +=1
        plt.grid(True)
    # plt.show()
    fig.savefig(f'images/{env_name}_graph_seeds_{y_type}.pdf', bbox_inches='tight')
    print(f'images/{env_name}_graph_seeds_{y_type}.pdf')
    plt.close(fig)


def get_data_by_seed(argss):
    """

    :param argss:
    :return:
    """
    data_by_seed = {}
    for args in argss:
        temp_args = copy.deepcopy(args)
        temp_args.seed = 0
        args_id = get_id(temp_args)
        if args_id not in data_by_seed:  # Create a dict for each group of seeds
            data_by_seed[args_id] = {'args': args}
        data = load_from_csv(args.save_sub_dir)
        for key, val in data.items():
            if key not in data_by_seed[args_id]:  # Create an array for data over seeds for each dict entry.
                data_by_seed[args_id][key] = []
            data_by_seed[args_id][key] += [val]
    return data_by_seed

def graph_seeds2(main_dirs, num_seeds, y_type):
    print(f"Graphing seeds...")
    fig, axs = init_ax(nrows=1, ncols=1)

    data_by_seed = {}
    for dir in main_dirs:
        train_type, env = dir.split('/')[-1].split('_')
        print(train_type)
        for seed in range(num_seeds):
            if train_type not in data_by_seed:
                data_by_seed[train_type] = {}
            data = load_from_csv(os.path.join(dir, f'seed_{seed}'))
            for key, val in data.items():
                if key not in data_by_seed[train_type]:
                    data_by_seed[train_type][key] = []
                data_by_seed[train_type][key] += [val]
    idx = 0
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    print(colors)
    
    print(data_by_seed.keys())
    for seed_key, seed_data in data_by_seed.items():
        print(seed_data.keys())
        label = str(seed_key)
        data_y = seed_data[y_type]
        min_idx = data_y[0].shape[0]
        for i in range(len(data_y)):
            print(data_y[i].shape[0])
            if data_y[i].shape[0] < min_idx:
                min_idx = data_y[i].shape[0]
        for i in range(len(data_y)):
            data_y[i] = data_y[i][:min_idx]
        # print(data_y)
        data_means_y = np.mean(data_y, axis=0)
        ps = axs[0].plot(data_means_y, label=label, color=colors[2+idx])
        # print(len(data_by_seed.keys()))
        std_err = np.std(data_y, axis=0)/np.sqrt(len(data_by_seed.keys()))
        axs[0].fill_between(
            range(len(data_means_y)), 
            data_means_y - std_err, 
            data_means_y + std_err,
            # stds = onp.std(data_y, axis=0)
            # axs[ax_index].fill_between(data_means_x, data_means_y - stds, data_means_y + stds,
            color=ps[0].get_color(), alpha=0.25
            )
        idx += 1
    if y_type == 'av-V-env-pi':
        y_label = 'True Policy Performance'
    elif y_type == 'cvar-alpha':
        y_label = 'CVaR'
    else:
        y_label = y_type
    axs[0].set_ylabel(y_label)

    axs[0].set_xlabel('Environment steps (x100)')
    [setup_ax(ax, fontsize=12) for ax in axs]
    fig.tight_layout(pad=1.0)
    # env_name = 'FrozenLake'
    # env_name = 'MDP2State'
    # env_name = 'Chain5StateSlip0.2'
    fig.savefig(f'images/finalperfs/{env}_graph_seeds_{y_type}.pdf')#, bbox_inches='tight')
    plt.close(fig)

def graph_seeds(argss, env_name, y_type):
    """

    :param argss:
    :param x_type:
    :return:
    """
    print(f"Graphing seeds...")
    fig, axs = init_ax(nrows=1, ncols=1)
    # train_types = ['regret-CVaR', 'CVaR', 'k-of-N', 'pg-CE', 'pg', 'VaR-sigmoid']
    data_by_seed = get_data_by_seed(argss)
    idx = 0
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    print(colors)
    
    for seed_key, seed_data in data_by_seed.items():
    
        label = str(seed_data['args'].train_type)
        data_y = seed_data[y_type]
        # if len(data_y[0]) < 149:
        #     continue
        # else:
        # print(data_y[0])
        min_idx = data_y[0].shape[0]
        for i in range(len(data_y)):
            print(data_y[i].shape[0])
            if data_y[i].shape[0] < min_idx:
                min_idx = data_y[i].shape[0]
        for i in range(len(data_y)):
            data_y[i] = data_y[i][:min_idx]
        # print(data_y)
        data_means_y = np.mean(data_y, axis=0)
        ps = axs[0].plot(data_means_y, label=label, color=colors[2+idx])
        # print(len(data_by_seed.keys()))
        std_err = np.std(data_y, axis=0)/np.sqrt(len(data_by_seed.keys()))
        axs[0].fill_between(
            range(len(data_means_y)), 
            data_means_y - std_err, 
            data_means_y + std_err,
            # stds = onp.std(data_y, axis=0)
            # axs[ax_index].fill_between(data_means_x, data_means_y - stds, data_means_y + stds,
            color=ps[0].get_color(), alpha=0.25
            )
        idx += 1
    if y_type == 'av-V-env-pi':
        y_label = 'True Policy Performance'
    elif y_type == 'cvar-alpha':
        y_label = 'CVaR'
    else:
        y_label = y_type
    axs[0].set_ylabel(y_label)

    axs[0].set_xlabel('Environment steps (x100)')
    [setup_ax(ax, fontsize=12) for ax in axs]
    fig.tight_layout(pad=1.0)
    # env_name = 'FrozenLake'
    # env_name = 'MDP2State'
    # env_name = 'Chain5StateSlip0.2'
    fig.savefig(f'images/finalperfs/{env_name}_graph_seeds_{y_type}.pdf')#, bbox_inches='tight')
    plt.close(fig)
