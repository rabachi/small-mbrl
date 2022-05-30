import csv
import os
from re import L
import matplotlib as mpl  # For changing matplotlib arguments
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
mpl.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt  # For graphing
import numpy as np
import copy
from deploy import get_id

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
    with open(path) as csvfile:
        reader = csv.DictReader(csvfile, skipinitialspace=True)
        data = {name: [] for name in reader.fieldnames}
        for row in reader:
            for name in reader.fieldnames:
                data[name].append(row[name])
    index = 150
    data['av-V-model-pi'] = np.array([float(i.strip()) for i in data['av-V-model-pi']])[:index]
    data['av-V-env-pi'] = np.array([float(i.strip()) for i in data['av-V-env-pi']])[:index]
    data['v-alpha-quantile'] = np.array([float(i.strip()) for i in data['v-alpha-quantile']])[:index]
    data['cvar-alpha'] = np.array([float(i.strip()) for i in data['cvar-alpha']])[:index]
    data['cvar-constraint-lambda'] = np.array([float(i.strip()) for i in data['cvar-constraint-lambda']])[:index]
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
    mpl.rcParams['legend.fontsize'] = fontsize
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

def setup_ax(ax, do_legend=True, alpha=0.0, fontsize=6,         legend_loc=None, handlelength=None):
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


def graph_single(train_types, y_type):
    """
    :param args:
    :param x_type:
    :param y_type:
    :return:
    """
    fig, axs = init_ax()
    for train_type in train_types:
        data = load_from_csv(args.save_sub_dir)

        axs[0].plot(range(len(data[y_type])), data[y_type], label=f'{train_type}')
        
    axs[0].set_xlabel('Iters'), axs[0].set_ylabel(f'Policy performance {y_type}')    
    axs[0].legend(bbox_to_anchor=(1,0.5))

    plt.grid(True)
    # plt.show()
    fig.savefig(f'images/graph_performances_{y_type}.pdf', bbox_inches='tight')
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
    for seed_key, seed_data in data_by_seed.items():
        label = str(seed_data['args'].train_type)
        data_y = seed_data[y_type]
        data_means_y = np.mean(data_y, axis=0)
        ps = axs[0].plot(data_means_y, label=label)
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
    axs[0].set_ylabel(y_type)
    axs[0].set_xlabel('Policy updates')
    [setup_ax(ax, fontsize=6) for ax in axs]
    fig.tight_layout(pad=1.0)
    # env_name = 'FrozenLake'
    # env_name = 'MDP2State'
    # env_name = 'Chain5StateSlip0.2'
    fig.savefig(f'images/{env_name}_graph_seeds_{y_type}.pdf')#, bbox_inches='tight')
    plt.close(fig)
