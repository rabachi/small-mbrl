import jax.numpy as jnp

def sigmoid(x):
    return jnp.exp(x - jnp.log(jnp.exp(x) + 1))