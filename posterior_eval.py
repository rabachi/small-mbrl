# imports
import gym
import numpy as np
from regex import R
from scipy.special import softmax
import matplotlib.pyplot as plt
from torch import initial_seed
from replay_buffer import ReplayBuffer
from src.model import DirichletModel
from envs.chain import Chain
from envs.randomfrozenlake import RandomFrozenLakeEnv
from envs.cliffwalking import CliffWalkingEnv
from envs.ring import Ring
from src.utils import collect_data, discount, collect_sa

#IMPLEMENT for unknown env.R, env.P, make sure value iteration is replaced with some other way to get optimal policy

SEED = 0
rng = np.random.RandomState()
# SEED = rng.randint(1)

def plot_env_model_distribution(agent, env_name, num_samples, num_points_per_sa, alpha0, policies, initial_dist):
    nState = agent.nState
    nAction = agent.nAction
    policy_types = policies.keys()
    v = {p_type: np.zeros((num_samples, 1)) for p_type in policy_types}

    for p_type in policy_types:
        for sample in range(num_samples):
            R_samp, P_samp = agent.sample_mdp()
            v_samp,pi_samp = agent.value_iteration(R_samp, P_samp)
            v[p_type][sample] = initial_dist @ v_samp
            # v[p_type][sample] = \
            #         initial_dist @ agent.policy_evaluation(
            #             (R_samp, P_samp), policies[p_type])
    num_bins = 50
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
        plt.text(300, 4.5 - i, f'Var {p_type}: {var[p_type]:.2f}, Mean: {mean:.2f}', horizontalalignment='center', verticalalignment='center')
        i += 3

    R_ce, P_ce = agent.get_CE_model()
    ce_value_opt, _ = agent.value_iteration(R_ce, P_ce)
    ce_perf = initial_dist @ ce_value_opt
    plt.axvline(ce_perf, color='k', linestyle='dashed',
                linewidth=1, label='Certainty equivalent optimal')

    plt.xlabel('V_p^pi')
    plt.ylabel('P_p(V_p^pi)')
    plt.title(f'{env_name}, alpha0: {alpha0}, Number of points per transition: {num_points_per_sa}')
    plt.legend()
    fig.savefig(f'images/dists/V_P_dist_{env_name}_per_sa_{num_points_per_sa}_alpha0_{alpha0}.pdf', bbox_inches='tight')
    # fig.savefig(f'images/env_dist_{env.get_name()}.pdf', bbox_inches='tight')
    plt.close(fig)

def update_posterior(agent, batch):
    obses, actions, rewards, next_obses, not_dones, _ = batch
    for state, action, reward, next_state, not_done in zip(obses, actions, rewards, next_obses, not_dones):
        agent.update_obs(int(state), int(action), reward, int(next_state), not not_done)

def MC_policy_evaluation(env, policy, num_rollouts, num_steps, init_distr, discount_factor):
    nState = init_distr.shape[0]
    count_init_rollouts = np.ones((nState, 1)) * num_rollouts
    init_states = init_distr.nonzero()
    count_init_rollouts[init_states] = 0
    returns_all = np.zeros((nState, num_rollouts))

    while (count_init_rollouts < num_rollouts).any():
        print('env', count_init_rollouts)
        for r in range(num_rollouts):
            state = env.reset()
            init_state = state
            count_init_rollouts[init_state] += 1
            ep_rewards = np.zeros(num_steps)
            step = 0
            done = False
            while step < num_steps:
                action = rng.multinomial(1, policy[state]).nonzero()[0][0]
                next_state, reward, done, _ = env.step(action)
                # if reward < -1:
                    # print(step, next_state, reward, done)
                ep_rewards[step] = reward
                step += 1
                state = next_state
            
            returns_all[init_state] = discount(ep_rewards, discount_factor)[0]

    v_pi = np.mean(returns_all, 1)
    return init_distr @ v_pi

def experiment():
    # env setup
    # env_name = 'Garnet(2,2,(2,2))'
    # env_name = 'State2MDP'
    # env_name = 'FrozenLake-v1' 
    # env_name = 'Taxi-v3'
    # env = gym.make(env_name)
    # env = State2MDP(SEED)
    nState = 5
    prob_slip = 0.2
    discount = 0.99
    # env = Ring(SEED)
    env = CliffWalkingEnv(SEED)
    # env = RandomFrozenLakeEnv(SEED, map_name=None) #this seed is not setting all the randomness here (map generated is using a diff number generator)
    # env = Chain(nState, prob_slip, discount, SEED)
    env_name = env.get_name()
    # env = GarnetMDP(2, 2, (2,2))
    nState = env.nState
    nAction = env.nAction
    # nState = env.observation_space.n
    # nAction = env.action_space.n
    alpha0 = 0.1

    initial_dist = env.initial_distribution

    # rng = np.random.RandomState(SEED)
    agent1 = DirichletModel(nState, nAction, SEED, discount, initial_dist)

    #collect multiple data batches (or just a lot of data that you can sample batches from)
    policy_types = ['sample-optimal']#, 'true-optimal']
    policies = dict.fromkeys(policy_types, [None] * len(policy_types))
    
    # policies['deterministic'] = np.zeros((nState, nAction))
    # policies['deterministic'][:, 0] = 1.0
    # policies['deterministic'][:, -1] = 0.0
    # policies['uniform'] = np.ones((nState, nAction)) * 1./nAction
    
    # v_opt, pi_opt = agent1.value_iteration(env.R, env.P)
    # # policies['optimal'] = softmax(rng.standard_normal(size=(nState, nAction)))
    # policies['true-optimal'] = pi_opt
    
    num_points_ablation = [10000] #1, 10, 100, 1000
    num_model_samples = 200
    # nEps = 10000
    num_points_per_sa = num_points_ablation[0]

    for num_points_per_sa in num_points_ablation:
        print(f'num points per transition: {num_points_per_sa}')
        agent = DirichletModel(nState, nAction, SEED, discount, initial_dist, alpha0=alpha0)
        data = collect_sa(rng, env, nState, nAction, num_points_per_sa)
    
        # data = collect_data(rng, env, nEps, nSteps, policies['uniform'])

        if len(data) > 0: 
            batch_data = data.sample(len(data))
            # Bayesian estimate
            update_posterior(agent, batch_data)
        plot_env_model_distribution(agent, env_name, num_model_samples, num_points_per_sa, alpha0, policies, initial_dist)
        del agent, data


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


