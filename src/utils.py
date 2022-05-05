import numpy as np
import jax.numpy as jnp
from jax.scipy.special import logsumexp
import jax
from jax.nn import sigmoid
from replay_buffer import ReplayBuffer

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


def softmax(vals, temp=1.):
    """Batch softmax
    Args:
     vals : S x A. Applied row-wise
     temp (float, optional): Defaults to 1.. Temperature parameter
    Returns:
    """
    return jnp.exp((1. / temp) * vals - logsumexp((1. / temp) *
                                                  vals, axis=1, keepdims=True))

if __name__ == "__main__":
    rewards = [1, 2, 3, 4]
    discount_factor = 0.9
    answer = [1+2*0.9+3*(0.9**2)+4*(0.9**3), 2+3*0.9+4*(0.9**2), 3+4*0.9, 4]
    print(answer)
    print(discount(rewards, discount_factor))
    print(discount(rewards, discount_factor) == np.asarray(answer))