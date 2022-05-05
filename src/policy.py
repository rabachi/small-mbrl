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

        if not p_params:
            p_params = jnp.ones((self.nState, self.nAction)) * 0.01
        self.p_params = p_params
        self.rng = np.random.RandomState(int(seed))

    def __call__(self, curr_state):
        action_probs = softmax(self.p_params.reshape(self.nState, self.nAction), self.temp)[curr_state]
        # rng = np.random.RandomState(SEED)
        return self.rng.multinomial(1, action_probs).nonzero()[0][0] 
    
    def get_params(self):
        return self.p_params
    
    def update_params(self, p_params):
        self.p_params = p_params
    
    def get_policy(self, p_params):
        return softmax(p_params.reshape(self.nState, self.nAction), self.temp)