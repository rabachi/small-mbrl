import numpy as np
import jax.numpy as jnp
import pdb
from src.utils import softmax, project_onto_simplex
from scipy import stats
import jax

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
            p_params = np.clip(jnp.ones((self.nState * self.nAction)) * self.rng.normal(0, 1, size=(self.nState * self.nAction)), 1e-6, 1.-1e-6)
            # p_params = jnp.ones((self.nState * self.nAction)) * self.rng.normal(0, 1, size=(self.nState * self.nAction))
            
        self.p_params = p_params
        # jax.lax.stop_gradient(project_onto_simplex(p_params.reshape(self.nState, self.nAction), axis=1))
        # print(self.p_params)

    def __call__(self, curr_state):
        # action_probs = softmax(self.p_params.reshape(self.nState, self.nAction), 1.0)#[curr_state]
        # action_probs = jax.lax.stop_gradient(project_onto_simplex(self.p_params.reshape(self.nState, self.nAction), axis=1))
        # print(action_probs[curr_state])
        p_params = self.p_params.reshape(self.nState, self.nAction)
        action_probs = p_params/np.sum(p_params, axis=1, keepdims=True)
        # pdb.set_trace()
        # print(action_probs)
        # rng = np.random.RandomState(SEED)
        # print(curr_state) #action_probs[curr_state])
        # if action_probs[curr_state].sum() > 1.0:
        #     print(action_probs[curr_state])
        # print(curr_state)
        return self.rng.multinomial(1, action_probs[curr_state]).nonzero()[0][0] 
    
    def get_params(self):
        return self.p_params
    
    def update_params(self, p_params):
        self.p_params = np.clip(p_params, 1.e-6, 1.0-1e-6)
        # self.p_params = p_params
    
    def reset_params(self):
        # self.p_params = jnp.ones((self.nState * self.nAction)) * self.rng.normal(0, 1, size=(self.nState * self.nAction))
        # self.p_params = jax.lax.stop_gradient(project_onto_simplex(p_params.reshape(self.nState, self.nAction), axis=1))
        self.p_params = np.clip(jnp.ones((self.nState * self.nAction)) * self.rng.normal(0, 1, size=(self.nState * self.nAction)), 1e-6, 1.-1e-6)

    def entropy(self, state):
        p_params = self.p_params.reshape(self.nState, self.nAction)
        action_probs = (p_params/np.sum(p_params, axis=1, keepdims=True))[state]
        # action_probs = jax.lax.stop_gradient(project_onto_simplex(self.p_params.reshape(self.nState, self.nAction), axis=1))[state]
        # action_probs = softmax(self.p_params.reshape(self.nState, self.nAction), 1.0)[state]
        return stats.entropy(action_probs)

    # def get_policy(self, p_params):
        # return softmax(p_params.reshape(self.nState, self.nAction), self.temp)