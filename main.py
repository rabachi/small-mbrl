# imports
import gym
import numpy as np
from scipy.special import softmax
import matplotlib.pyplot as plt
from src.model import DirichletModel
from src.envs import State2MDP, GarnetMDP

def experiment():
    # env setup
    # env = gym.make('FrozenLake-v0')
    # env = State2MDP()
    env = GarnetMDP(5, 5, (5,2))
    state = env.reset()
    nState = 5#env.observation_space.n
    nAction = 5#env.action_space.n
    nEps = 20

    agent = DirichletModel(nState, nAction)

    for ep in range(nEps):
        #generate data
        done = False

        # while not done:
        for step in range(10):
            action = np.random.choice(nAction)
            next_state, reward, done, _ = env.step(action)#, state)

            agent.update_obs(state, action, reward, next_state, done)
            print(agent.R_prior, agent.P_prior)

    policies = [None] * 4
    num_samples = 1000
    #v state, pi, num_samples
    # v = np.zeros((num_samples, len(policies), nState))
    v = np.zeros((num_samples, len(policies), 1))
    # v_progress = np.zeros((num_samples, 3, nState))
    initial_dist = np.ones(nState) * 1./nState
    for pi_idx in range(len(policies) - 1):
        policies[pi_idx] = softmax(np.random. rand(nState,
                                                     nAction), axis=1)
        # policies[2] = softmax(np.random.power(3, (nState,
                                                  # nAction)), axis=1)
        # policies[2] = softmax(np.random.geometric(0.3, size=(nState,
        #                                           nAction)), axis=1)
        policies[0] = np.zeros((nState, nAction))
        policies[0][:, 0] = 1.0
        policies[0][:, -1] = 0.0

        policies[1] = np.ones_like(policies[1]) * 1./nAction
        # print(np.sum(policies[pi_idx], 1))

        for sample in range(num_samples):
            R_samp, P_samp = agent.sample_mdp()
            v[sample, pi_idx] = \
                initial_dist @ agent.policy_evaluation(
                    R_samp,
                                                        P_samp,
                                                    policies[pi_idx])
            value_opt, _ = \
                agent.value_iteration(R_samp, P_samp)
            v[sample, 3] = initial_dist @ value_opt
            # v_progress[sample] = v_progress_[:3]
            # print(v[sample, pi_idx])
    num_bins = 50
    fig = plt.figure()
    # for s in range(nState):
    for i in range(len(policies)):
        labels = {0: 'deterministic', 1: 'uniform',
                  # 2:'geometric',
                  2:'random',
                  3:'optimal'}
        plt.hist(v[:, i], num_bins, alpha=0.5, label=f''
                                                        f''
                                                        f''
                                                        f''
                                                        f'{labels[i]}')
        # if i < 3:
        #     plt.hist(v_progress[:, i, s], num_bins, alpha=0.5,
        #          label=f'in progress {i}')
    plt.xlabel('V_p^pi')
    plt.ylabel('P_p(V_p^pi)')
    plt.legend()
    plt.show()
    # policies, values, mdps are tabular (should also think about doing function approximation easily and how it affects)

    # RL part
    # prior over env
    # sample mdp
    # policy evaluation for multiple policies
    # plot distribution of value functions over mdps


    # Bandit part
    # bandits:
    # setup MAB
    # sample from MAB using some policies
    # evaluation of policies for multiple policies
    # plot distribution of values over MAB distributions?

    # Risk evaluation part

if __name__ == '__main__':
    experiment()

