import numpy as np

class Chain:
    def __init__(self, nState, prob_slip, discount, SEED):
        self.rng = np.random.RandomState(SEED)
        self.nState = nState
        self.nAction = 2
        self.prob_slip = prob_slip
        self.initial_distribution = np.zeros(self.nState)
        self.initial_distribution[0] = 1.
        self.P = np.zeros((self.nState, self.nAction, self.nState))# + 1e-6
        self.R = np.zeros((self.nState, self.nAction))
        
        for s in range(self.nState):
            self.P[s, 0, 0] = self.prob_slip #move to state 0 with action 0 if "slip"
            self.P[s, 1, 0] = 1. - self.prob_slip #move to state 0 with action 1 if not slip
            self.R[s, 1] = 2.
            if s == self.nState - 1:
                self.P[s, 0, s] = 1. - self.prob_slip #stay in rightmost state with action 0 if not slip
                self.P[s, 1, s] = self.prob_slip #with action 1, if slip, stay in rightmost state
                self.R[s, 0] = 10.
            else:
                self.P[s, 0, s+1] = 1. - self.prob_slip 
                self.P[s, 1, s+1] = self.prob_slip
                # self.R[s, 0] = 0
        print(np.sum(self.P, axis=2))
        self.discount = discount
        self.state = 0

    def get_name(self):
        return f'Chain{self.nState}StateSlip{self.prob_slip}'

    def reset(self):
        self.state = 0
        return self.state

    def reset_to_state(self, state):
        self.state = state
        return int(self.state)
    
    def step(self, action):
        next_state = self.rng.multinomial(1, self.P[self.state, action]).nonzero()[0][0]
        reward = self.R[self.state, action]
        self.state = next_state
        terminal = False
        return (int(next_state), reward, terminal, {})
