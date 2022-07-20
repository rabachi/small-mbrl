import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax
from jax.nn import sigmoid
from itertools import product
from replay_buffer import ReplayBuffer
# from memory_profiler import profile

def similar(L):
    return all(np.isclose(x, y, rtol=1e-1) for x,y in zip(L[-10:], L[-9:]))
    
def non_decreasing(L):
    return all((x<y) or np.isclose(x,y,rtol=1.) for x, y in zip(L, L[1:]))

def update_mle(nState, nAction, batch):
    transition_counts = np.zeros((nState, nAction, nState))
    rewards_estimate = np.zeros((nState, nAction))
    obses, actions, rewards, next_obses, not_dones, _ = batch
    for state, action, reward, next_state, not_done in zip(obses, actions, rewards, next_obses, not_dones):
        transition_counts[int(state), int(action), int(next_state)] += 1
        rewards_estimate[int(state), int(action)] = reward

    transition_counts = np.nan_to_num(transition_counts/np.sum(transition_counts, axis=2, keepdims=True))
    return transition_counts, rewards_estimate

def collect_sa(rng, env, nState, nAction, num_points_per_sa):
    replay_buffer = ReplayBuffer([1], [1], num_points_per_sa * (env.nState*env.nAction))
    count_sa = np.ones((env.nState, env.nAction)) * 0.#num_points_per_sa
    # init_states = init_distr.nonzero()
    # count_sa[init_states] = 0
    # print(env.nState, env.nAction)
    collected_all_sa = False if num_points_per_sa > 0 else True
    # while not collected_all_sa:
    done = False
    for state, action in product(range(env.nState), range(env.nAction)):
        while count_sa[state, action] < num_points_per_sa:
            env.reset_to_state(state)
            next_state, reward, done, _ = env.step(action)
            if count_sa[state, action] < num_points_per_sa:
                count_sa[state, action] += 1
                replay_buffer.add(state, action, reward, next_state, done, done)
            state = next_state
            # collected_all_sa = (count_sa >= num_points_per_sa).all()
            # print(count_sa)
    # print(count_sa)
    print('finished data collection')
    return replay_buffer

# def collect_sa(rng, env, nState, nAction, num_points_per_sa):
#     replay_buffer = ReplayBuffer([1], [1], num_points_per_sa * (nState*nAction))
#     count_sa = np.ones((nState, nAction)) * 0.#num_points_per_sa
#     # init_states = init_distr.nonzero()
#     # count_sa[init_states] = 0
    
#     collected_all_sa = False if num_points_per_sa > 0 else True
#     trials = 0
#     while not collected_all_sa and trials < 1e4:
#         trials += 1
#         done = False
#         state = env.reset()
#         step = 0
#         while step < 100: #not collected_all_sa:
#             action = int(rng.choice(nAction))
#             next_state, reward, done, _ = env.step(action)
#             if count_sa[state, action] < num_points_per_sa:
#                 count_sa[state, action] += 1
#                 replay_buffer.add(state, action, reward, next_state, done, done)
#             state = next_state
#             collected_all_sa = (count_sa >= num_points_per_sa).all()
#             # print(count_sa)
#             step += 1
#     # print(count_sa)
#     print('finished data collection')
#     return replay_buffer

def collect_data(rng, env, nEps, nSteps, policy):
    replay_buffer = ReplayBuffer([1], [1], nEps * nSteps)
    for ep in range(nEps):
        # print(ep)
        done = False
        step = 0
        state = env.reset()
        while not done and step < nSteps:
        # for step in range(10):
            # action = rng.choice(action_space)
            if policy[state].shape == (): #optimal policy
                action = int(policy[state])
            else:
                action = rng.multinomial(1, policy[state]).nonzero()[0][0]
            next_state, reward, done, _ = env.step(action)
            replay_buffer.add(state, action, reward, next_state, done, done)
            state = next_state
            step += 1
    return replay_buffer

def discount(rewards, discount_factor) -> np.ndarray:
    rewards_np = np.asarray(rewards)
    t_steps = np.arange(len(rewards))
    r = rewards_np * discount_factor**t_steps
    r = r[::-1].cumsum()[::-1] / discount_factor**t_steps
    return r

# def sigmoid(x):
#     "Numerically stable sigmoid function."
#     if x >= 0:
#         z = jnp.exp(-x)
#         return 1 / (1 + z)
#     else:
#         # if x is less than zero then z will be small, denom can't be
#         # zero because it's 1+z.
#         z = jnp.exp(x)
#         return z / (1 + z)

def vmap_sigmoid(x):
    return jax.vmap(sigmoid, in_axes=0, out_axes=0)(x)

def get_log_policy(p_params, n_states, n_actions, temp):
    """

    :param p_params:
    :return:
    """
    return log_softmax(p_params.reshape(n_states, n_actions), temp)#.reshape(-1,)
    
def get_policy(p_params, nState, nAction):
    return softmax(p_params.reshape(nState, nAction), 1.0)

def log_softmax(vals, temp=1.):
    """Same function as softmax but not exp'd
        Args:
            vals : S x A. Applied row-wise
            temp (float, optional): Defaults to 1.. Temperature parameter
        Returns:
        """
    return (1. / temp) * vals - logsumexp((1. / temp) * vals, axis=1, keepdims=True)

def softmax(vals, temp=1.):
    """Batch softmax
    Args:
     vals : S x A. Applied row-wise
     temp (float, optional): Defaults to 1.. Temperature parameter
    Returns:
    """
    z = vals - jnp.max(vals, axis=1, keepdims=True)
    numerator = jnp.exp(z)
    denom = jnp.sum(numerator, axis=1, keepdims=True)
    softmax = numerator/denom
    # test = jnp.exp((1. / temp) * vals - logsumexp((1. / temp) *
                                                #   vals, axis=1, keepdims=True))
    # print(jnp.isclose(test, softmax))
    return softmax

if __name__ == "__main__":
    rewards = [1, 2, 3, 4]
    discount_factor = 0.9
    answer = [1+2*0.9+3*(0.9**2)+4*(0.9**3), 2+3*0.9+4*(0.9**2), 3+4*0.9, 4]
    print(answer)
    print(discount(rewards, discount_factor))
    print(discount(rewards, discount_factor) == np.asarray(answer))