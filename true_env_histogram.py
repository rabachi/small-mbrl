# imports
import gym
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from torch import initial_seed
from replay_buffer import ReplayBuffer
from src.model import DirichletModel
from envs.chain import Chain
from envs.randomfrozenlake import RandomFrozenLakeEnv
from envs.cliffwalking import CliffWalkingEnv
from envs.ring import Ring
from src.utils import collect_data, discount, collect_sa, update_mle

#IMPLEMENT for unknown env.R, env.P, make sure value iteration is replaced with some other way to get optimal policy

SEED = 0
rng = np.random.RandomState()
# SEED = rng.randint(1)

def value_iteration(nState: int, nAction: int, discount_factor, r_matrix: np.ndarray, p_matrix: np.ndarray):
    epsilon = 1e-5
    V = np.zeros(nState)
    delta = np.infty
    i = 0
    while delta > epsilon:
        for s in range(nState):
            prev_v = V[s]
            V[s] = np.max(r_matrix[s] + np.einsum('at,t->a', p_matrix[s], discount_factor*V))
            delta = np.abs(prev_v - V[s])
        i += 1
    pi = np.zeros((nState, nAction))
    for s in range(nState):
        pi[s][np.argmax(r_matrix[s] + np.einsum('at,t->a', p_matrix[s], discount_factor * V))] = 1
    return V, pi

def policy_evaluation(env, policy, num_episodes_eval, num_steps, discount_factor):
    nState = env.nState
    v = np.zeros(nState)
    for s in range(nState):
        v_eps = np.zeros(num_episodes_eval)
        env.reset_to_state(s)
        for ep in range(num_episodes_eval):
            rewards = []
            for step in range(num_steps):
                action = np.argmax(policy[s]) #only for optimal policy
                next_state, reward, done, _ = env.step(action)
                state = next_state
                rewards.append(reward) 
            returns = discount(rewards, discount_factor)#env.discount)
            v_eps[ep] = returns[0]
        v[s] = np.mean(v_eps)
    return v

def experiment():
    prob_slip = 0.2
    nState = 5
    discount_factor = 0.99

    # env = Chain(nState, prob_slip, discount_factor, SEED)
    env = CliffWalkingEnv(SEED)
    # env = Ring(SEED)
    # env = State2MDP(SEED)
    # env = GarnetMDP(2, 2, (2,2))
    # env = RandomFrozenLakeEnv(SEED, map_name=None) 

    num_points_per_sa = 1000
    num_repeats = 200
    num_episodes_eval = 10

    nState = env.nState
    nAction = env.nAction

    values = np.zeros(num_repeats)
    for m in range(num_repeats):
        print(m)
        data_m = collect_sa(rng, env, nState, nAction, num_points_per_sa)
        mle_transitions, mle_rewards = update_mle(nState, nAction, data_m.sample(len(data_m)))
        _, pi_star = value_iteration(nState, nAction, discount_factor, mle_rewards, mle_transitions)
        v_pi_star = policy_evaluation(env, pi_star, num_episodes_eval, int(1/(1 - discount_factor)), discount_factor)#env.discount)))
        values[m] = env.initial_distribution @ v_pi_star

    file = f'data/values_{env.get_name()}_numpoints{num_points_per_sa}_numrepeats{num_repeats}_actionprob_0.4.npy'
    np.save(file, values)
    num_bins = 50
    fig = plt.figure()

    n, bins = np.histogram(values)
    mids = 0.5*(bins[1:] + bins[:-1])
    mean = np.average(mids, weights=n)
    print(mean)
    var = np.average((mids - mean)**2, weights=n)
    plt.hist(values, num_bins, alpha=0.5, label=f'optimal')
    print(f'variance: {var}')
    print(f'mean: {mean}')
    # plt.text(0.5, 4.5, f'Variance: {var:.2f}, Mean: {mean:.2f}', horizontalalignment='center', verticalalignment='center')

    plt.xlabel('V_env^pi')
    plt.ylabel('P_env(V_env^pi)')
    plt.title(f'{env.get_name()}, Number of points per transition: {num_points_per_sa}')
    plt.legend()
    fig.savefig(f'images/env_dists/V_env_dist_{env.get_name()}_per_sa_{num_points_per_sa}.pdf', bbox_inches='tight')
    # fig.savefig(f'images/env_dist_{env.get_name()}.pdf', bbox_inches='tight')
    plt.close(fig)

if __name__ == '__main__':
    experiment()