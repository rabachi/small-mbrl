import numpy as np
import jax.numpy as jnp
from src.utils import softmax

class Policy:
    def __init__(self,
        nState,
        nAction,
        temp,
        seed,
        p_params=None):

        self.nState = nState
        self.nAction = nAction
        self.temp = temp

        self.rng = np.random.RandomState(int(seed))
        if not p_params:
            p_params = jnp.ones((self.nState * self.nAction)) * self.rng.normal(0, 1, size=(self.nState * self.nAction))
        self.p_params = p_params

    def __call__(self, curr_state):
        action_probs = softmax(self.p_params.reshape(self.nState, self.nAction), self.temp)[curr_state]
        # rng = np.random.RandomState(SEED)
        # print(sum(action_probs))
        # print(action_probs)
        return self.rng.multinomial(1, action_probs).nonzero()[0][0] 
    
    def get_params(self):
        return self.p_params
    
    def update_params(self, p_params):
        self.p_params = np.clip(p_params, -5., 5.0)
    
    def reset_params(self):
        self.p_params = jnp.ones((self.nState * self.nAction)) * self.rng.normal(0, 1, size=(self.nState * self.nAction))

    def get_policy(self, p_params):
        return softmax(p_params.reshape(self.nState, self.nAction), self.temp)