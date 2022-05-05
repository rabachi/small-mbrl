# imports
import gym
import numpy as np
from regex import R
from scipy.special import softmax
import matplotlib.pyplot as plt
from replay_buffer import ReplayBuffer
from src.model import DirichletModel
from envs.envs import State2MDP, GarnetMDP
from src.utils import collect_data

#IMPLEMENT for unknown env.R, env.P, make sure value iteration is replaced with some other way to get optimal policy

SEED = 0
rng = np.random.RandomState()
# SEED = rng.randint(1)

def collect_transitions(env, nState, nAction, num_points_per_transition):
    replay_buffer = ReplayBuffer([1], [1], num_points_per_transition * (nState*nAction*nState))
    count_transitions = np.zeros((nState, nAction, nState))
    #need to keep track of which transitions we have data points for, and how many points we have points for
    # need to figure out which transition we have collected data for : use state and action and next_state to index an array and increment a counter in that array
    collected_all_transitions = False if num_points_per_transition > 0 else True
    while not collected_all_transitions:
        done = False
        state = env.reset()
        while not collected_all_transitions:
            action = rng.choice(nAction)
            next_state, reward, done, _ = env.step(action)
            if count_transitions[state, action, next_state] < num_points_per_transition:
                count_transitions[state, action, next_state] += 1
                replay_buffer.add(state, action, reward, next_state, done, done)
            state = next_state
            collected_all_transitions = (count_transitions >= num_points_per_transition).all()
    print(count_transitions)
    print('finished data collection')
    return replay_buffer

def plot_env_model_distribution(agent, env_name, num_samples, num_points_per_transition, alpha0, policies):
    nState = agent.nState
    nAction = agent.nAction
    policy_types = policies.keys()
    v = {p_type: np.zeros((num_samples, 1)) for p_type in policy_types}

    initial_dist = np.ones(nState) * 1./nState

    for p_type in policy_types:
        for sample in range(num_samples):
            R_samp, P_samp = agent.sample_mdp()
            v[p_type][sample] = \
                    initial_dist @ agent.policy_evaluation(
                        (R_samp, P_samp), policies[p_type])
    
    num_bins = 50
    # mean = np.zeros((len(policy_types)))
    var = {}
    fig = plt.figure()
    i = 0
    for p_type in policy_types:
        n, bins = np.histogram(v[p_type])
        mids = 0.5*(bins[1:] + bins[:-1])
        mean = np.average(mids, weights=n)
        print(mean)
        var[p_type] = np.average((mids - mean)**2, weights=n)
        plt.hist(v[p_type], num_bins, alpha=0.5, label=f'{p_type}')
        plt.text(mean, 12 - i, f'Var {p_type}: {var[p_type]:.2f}, Mean: {mean:.2f}')
        i += 3

    R_ce, P_ce = agent.get_CE_model()
    ce_value_opt, _ = agent.value_iteration(R_ce, P_ce)
    ce_perf = initial_dist @ ce_value_opt
    plt.axvline(ce_perf, color='k', linestyle='dashed',
                linewidth=1, label='Certainty equivalent optimal')

    plt.xlabel('V_p^pi')
    plt.ylabel('P_p(V_p^pi)')
    plt.title(f'{env_name}, alpha0: {alpha0}, Number of points per transition: {num_points_per_transition}')
    plt.legend()
    fig.savefig(f'images/dists/V_P_dist_{env_name}_minnumpointspertransition_{num_points_per_transition}_alpha0_{alpha0}.pdf', bbox_inches='tight')
    # fig.savefig(f'images/env_dist_{env.get_name()}.pdf', bbox_inches='tight')
    plt.close(fig)

def update_posterior(agent, batch):
    obses, actions, rewards, next_obses, not_dones, _ = batch
    for state, action, reward, next_state, not_done in zip(obses, actions, rewards, next_obses, not_dones):
        agent.update_obs(int(state), int(action), reward, int(next_state), not not_done)

def update_mle(nState, nAction, batch):
    transition_counts = np.zeros((nState, nAction, nState))
    rewards_estimate = np.zeros((nState, nAction))
    obses, actions, rewards, next_obses, not_dones, _ = batch
    for state, action, reward, next_state, not_done in zip(obses, actions, rewards, next_obses, not_dones):
        transition_counts[state, action, next_state] += 1
        rewards_estimate[state, action] = reward
    
    transition_counts /= obses.shape[0]
    return transition_counts, rewards_estimate

def plot_env_env_distribution(agent, env, env_name, num_samples, policies):
    nState = agent.nState
    nAction = agent.nAction
    policy_types = policies.keys() 

    initial_dist = np.ones(nState) * 1./nState
    batch_size = 2000
    data = {}
    v_pi_env = {}
    nEps = 100000
    nSteps = 100
    for name, policy in policies.items():
        print(name)
        data[name] = collect_data(rng, env, nEps, nSteps, policy)
        #sample N batches from data, (should be disjoint?), plot value distributions 
        nBatches = 500
        v_pi_env[name] = np.zeros((nBatches, 1))
        for n in range(nBatches):
            batch_env = data[name].sample(batch_size)
            #for each policy, do policy evaluation on batch_for_env?
            v_pi_env[name][n] = initial_dist @ agent.sample_policy_evaluation(batch_env) #is this function correct?
    num_bins = 50
    var = {}
    fig = plt.figure()
    i = 0
    for p_type in policy_types:
        n, bins = np.histogram(v_pi_env[p_type])
        mids = 0.5*(bins[1:] + bins[:-1])
        mean = np.average(mids, weights=n)
        print(mean)
        var[p_type] = np.average((mids - mean)**2, weights=n)
        plt.hist(v_pi_env[p_type], num_bins, alpha=0.5, label=f'{p_type}')
        plt.text(mean, 12 - i, f'Var {p_type}: {var[p_type]:.2f}, Mean: {mean:.2f}')
        i += 3

    R_ce, P_ce = agent.get_CE_model()
    ce_value_opt, _ = agent.value_iteration(R_ce, P_ce)
    ce_perf = initial_dist @ ce_value_opt
    plt.axvline(ce_perf, color='k', linestyle='dashed',
                linewidth=1, label='Certainty equivalent optimal')

    plt.xlabel('V_env^pi')
    plt.ylabel('P_env(V_env^pi)')
    plt.title(f'{env_name}, vlarge data')
    plt.legend()
    fig.savefig(f'images/dists/V_env_dist_{env_name}_policy_vlarge_data.pdf', bbox_inches='tight')
    plt.close(fig)

def experiment():
    # env setup
    # env_name = 'Garnet(2,2,(2,2))'
    env_name = 'State2MDP'
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
    alpha0 = 0.1

    batch_size = 1000
    initial_dist = np.ones(nState) * 1./nState

    # rng = np.random.RandomState(SEED)
    agent1 = DirichletModel(nState, nAction, SEED)

    #collect multiple data batches (or just a lot of data that you can sample batches from)
    policy_types = ['uniform', 'true-optimal']
    policies = dict.fromkeys(policy_types, [None] * len(policy_types))
    
    # policies['deterministic'] = np.zeros((nState, nAction))
    # policies['deterministic'][:, 0] = 1.0
    # policies['deterministic'][:, -1] = 0.0
    policies['uniform'] = np.ones((nState, nAction)) * 1./nAction
    v_opt, pi_opt = agent1.value_iteration(env.R, env.P)
    # policies['optimal'] = softmax(rng.standard_normal(size=(nState, nAction)))
    policies['true-optimal'] = pi_opt
    num_points_ablation = [500] #10, 50, 100, 
    num_model_samples = 500
    # nEps = 10000
    # nSteps = 100

    for num_points_per_transition in num_points_ablation:
        print(f'num points per transition: {num_points_per_transition}')
        agent = DirichletModel(nState, nAction, SEED, alpha0=alpha0)
        data = collect_transitions(env, nState, nAction, num_points_per_transition)
        # data = collect_data(rng, env, nEps, nSteps, policies['uniform'])

        if len(data) > 0: 
            batch_data = data.sample(len(data))
            # Bayesian estimate
            update_posterior(agent, batch_data)
        plot_env_model_distribution(agent, env_name, num_model_samples, num_points_per_transition, alpha0, policies)
        del agent, data, batch_data
    
    # true env
    # transition_counts, rewards_estimate = update_mle(nState, nAction, batch_data)
    # plot_env_env_distribution(agent1, env, env_name, num_model_samples, policies)

if __name__ == '__main__':
    experiment()

    # #Test sample_policy_evaluation vs. policy_performance here on same policies -- look close
    # env = GarnetMDP(2, 2, (2,2))
    # state = env.reset()
    # nState = env.nState
    # nAction = env.nAction
    # agent = DirichletModel(nState, nAction, SEED)
    # p_params = np.ones((nState, nAction)) * 1/nAction

    # nEps = 1000
    # nSteps = 100
    # batch_size = 100000
    # policy = softmax(p_params)

    # data = collect_data(env, nEps, nSteps, policy)
    # initial_dist = np.ones(nState) * 1./nState
    # v = initial_dist @ agent.sample_policy_evaluation(data.sample(batch_size))
    # v_true = agent.policy_performance((env.R, env.P), p_params)

    # print(v, v_true)


