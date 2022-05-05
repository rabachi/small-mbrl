import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt

from envs.envs import State2MDP, GarnetMDP
from src.model import DirichletModel

from src.exact_pg import policy_performance, get_policy, \
    policy_evaluation, iterative_policy_evaluation, mdp_3states, model
from src.utils import sigmoid
from jax.experimental import optimizers
from jax.lax import stop_gradient
from src.logger import CSVLogger
from src.utils import *

def experiment_var():
    # env setup
    env = GarnetMDP(5, 5, (5,2))
    state = env.reset()
    nState = 5 #env.observation_space.n
    nAction = 5 #env.action_space.n
    nEps = 20

    logger = CSVLogger(
        fieldnames={'av_V_p_pi':0},
        filename='./log.csv'
    )

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

    initial_dist = np.ones(nState) * 1./nState
    #Take policy gradients
    #gradient of V
    #samples from posterior
    #delta: if V == alpha
    #sigmoid(V-alpha)
    #regret version
    #CVaR version

    # Fix policy_evaluation to work with jax, and the matrix issue
    # Fix sigmoid
    # add logger object

    num_samples = 100
    num_iters = 100
    p_lr = 0.01
    p_params = jnp.ones((nState, nAction)) * 0.01
    alpha = 0.1 #Placeholder
    for i in range(num_iters):
        to_log_v = 0
        p_grad = jnp.zeros_like(p_params)
        for sample in range(num_samples):
            R_samp, P_samp = agent.sample_mdp() #Can I sample in
            # parallel? Do many samples here, then do
            # policy_evaluation in parallel (using vmap eg)?
            V_p_pi, V_p_pi_grad = jax.value_and_grad(
                agent.policy_performance, 1)(
                (P_samp, R_samp),
                p_params
            )
            # delta way
            if V_p_pi == alpha:
                pass

            to_log_v += V_p_pi
            p_grad += jax.grad(sigmoid)(V_p_pi - alpha) * V_p_pi_grad

        p_params += p_lr * p_grad/num_samples
        logger.writerow({'av_V_p_pi': to_log_v/num_samples})
        print(f'Value fn: {to_log_v/num_samples}')

    V_p_pi = agent.policy_evaluation((P_samp, R_samp), p_params)
    logger.plot_histogram()#V_p_pi, p_params) #put in logger
    logger.plot_performance()#training_values) #put in logger


if __name__=="__main__":
    experiment_var()