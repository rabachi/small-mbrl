# imports
import gym
import numpy as np
from regex import R
from scipy.special import softmax
import matplotlib.pyplot as plt
# from replay_buffer import ReplayBuffer
from src.model import DirichletModel
from envs.envs import State2MDP, GarnetMDP
from src.utils import collect_data

#IMPLEMENT for unknown env.R, env.P, make sure value iteration is replaced with some other way to get optimal policy

SEED = 0
rng = np.random.RandomState()
# SEED = rng.randint(1)


def plot_env_model_distribution(agent, env_name, num_samples, data_p_type='uniform'):
    nState = agent.nState
    nAction = agent.nAction
    # num_samples = 1000
    policy_types = ['deterministic', 'uniform', 'optimal']
    policies = dict.fromkeys(policy_types, [None] * len(policy_types))
    # v = dict(zip(policy_types, [np.zeros((num_samples, 1))] * len(policy_types)))
    v = {p_type: np.zeros((num_samples, 1)) for p_type in policy_types}

    initial_dist = np.ones(nState) * 1./nState

    policies['deterministic'] = np.zeros((nState, nAction))
    policies['deterministic'][:, 0] = 1.0
    policies['deterministic'][:, -1] = 0.0

    policies['uniform'] = np.ones((nState, nAction)) * 1./nAction

    for p_type in policy_types:
        for sample in range(num_samples):
            R_samp, P_samp = agent.sample_mdp()
            # agent.update_CE_model(R_samp, P_samp, sample + 1)

            if p_type == 'optimal':
                value_opt, _ = \
                    agent.value_iteration(R_samp, P_samp)
                v['optimal'][sample] = initial_dist @ value_opt
            else:
                v[p_type][sample] = \
                    initial_dist @ agent.policy_evaluation(
                        (R_samp, P_samp), policies[p_type])
    num_bins = 50
    fig = plt.figure()
    for p_type in policy_types:
        plt.hist(v[p_type], num_bins, alpha=0.5, label=f'{p_type}')

    R_ce, P_ce = agent.get_CE_model()
    ce_value_opt, _ = agent.value_iteration(R_ce, P_ce)
    ce_perf = initial_dist @ ce_value_opt
    plt.axvline(ce_perf, color='k', linestyle='dashed',
                linewidth=1, label='Certainty equivalent optimal')

    plt.xlabel('V_p^pi')
    plt.ylabel('P_p(V_p^pi)')
    plt.legend()
    fig.savefig(f'images/dists/V_P_dist_{env_name}_data_{data_p_type}_policy.pdf', bbox_inches='tight')
    # fig.savefig(f'images/env_dist_{env.get_name()}.pdf', bbox_inches='tight')
    plt.close(fig)

def update_posterior(agent, batch):
    obses, actions, rewards, next_obses, not_dones, _ = batch
    for state, action, reward, next_state, not_done in zip(obses, actions, rewards, next_obses, not_dones):
        agent.update_obs(int(state), int(action), reward, int(next_state), not not_done)

def experiment():
    # env setup
    # env_name = 'Garnet(2,2,(2,2))'
    # env_name = 'State2MDP'
    # env_name = 'FrozenLake-v1' 
    # env_name = 'Taxi-v3'
    # env = gym.make(env_name)
    env = State2MDP(SEED)
    # env = GarnetMDP(2, 2, (2,2))
    state = env.reset()
    nState = env.nState
    nAction = env.nAction
    # nState = env.observation_space.n
    # nAction = env.action_space.n

    nEps = 100000
    nSteps = 10
    batch_size = 1000
    initial_dist = np.ones(nState) * 1./nState

    # rng = np.random.RandomState(SEED)
    agent = DirichletModel(nState, nAction, SEED)

    #collect multiple data batches (or just a lot of data that you can sample batches from)
    policy_types = ['deterministic', 'uniform']#, 'optimal']
    policies = dict.fromkeys(policy_types, [None] * len(policy_types))
    
    policies['deterministic'] = np.zeros((nState, nAction))
    policies['deterministic'][:, 0] = 1.0
    policies['deterministic'][:, -1] = 0.0
    policies['uniform'] = np.ones((nState, nAction)) * 1./nAction
    # v_opt, pi_opt = agent.value_iteration(env.R, env.P)
    # policies['optimal'] = softmax(rng.standard_normal(size=(nState, nAction)))
    v_pi_env = {}
    
    data = {}
    for name, policy in policies.items():
        print(name)
        data[name] = collect_data(rng, env, nEps, nSteps, policy)
        #sample N batches from data, (should be disjoint?), plot value distributions 
        nBatches = 1000
        v_pi_env[name] = np.zeros((nBatches, 1))
        for n in range(nBatches):
            batch_env = data[name].sample(batch_size)
            #for each policy, do policy evaluation on batch_for_env?
            v_pi_env[name][n] = initial_dist @ agent.sample_policy_evaluation(batch_env) #is this function correct?

    num_bins = 50
    fig = plt.figure()
    for p_type in policies.keys():
        plt.hist(v_pi_env[p_type], num_bins, alpha=0.5, label=f'{p_type}')
    plt.xlabel('V_env^pi')
    plt.ylabel('P_env(V_env^pi)')
    plt.legend()
    fig.savefig(f'images/dists/V_env_dist_{env_name}.pdf', bbox_inches='tight')
    plt.close(fig)

    #sample a batch, update posterior on batch, plot value distributions on N mdp samples from posterior
    for p_type in policy_types:
        batch_for_posterior = data[p_type].sample(batch_size)
        update_posterior(agent, batch_for_posterior)
        plot_env_model_distribution(agent, env_name, nBatches, data_p_type=p_type)

if __name__ == '__main__':
    experiment()

    #Test sample_policy_evaluation vs. policy_performance here on same policies -- look close
    # env = GarnetMDP(2, 2, (2,2))
    # state = env.reset()
    # nState = env.nState
    # nAction = env.nAction
    # agent = DirichletModel(nState, nAction, SEED)
    # p_params = np.ones((nState, nAction)) * 1/nAction

    # nEps = 10
    # nSteps = 10
    # batch_size = 100
    # policy = softmax(p_params)

    # data = collect_data(rng, env, nEps, nSteps, policy)
    # initial_dist = np.ones(nState) * 1./nState
    # v = initial_dist @ agent.sample_policy_evaluation(data.sample(batch_size))
    # v_true = agent.policy_performance((env.R, env.P), p_params)

    # print(v, v_true)


