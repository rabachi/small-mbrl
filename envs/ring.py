import numpy as np

class Ring:
    def __init__(self, SEED) -> None:
        self.nState = 5
        self.nAction = 3
        self.state = 0
        self.P = np.zeros((self.nState, self.nAction, self.nState))
        self.R = np.zeros((self.nState, self.nAction, self.nState))
        
        self.R[1, :, 2] = 0.5
        self.R[3, :, 2] = 0.5
        self.R[2, :, 2] = 1.0
        for s in range(self.nState):
            if (s == 0) or (s == 1) or (s == 3):
                #action a
                self.P[s, 0, s - 1] = 1.0
                #action b
                self.P[s, 1, s] = 0.8
                self.P[s, 1, s + 1] = 0.1
                self.P[s, 1, s - 1] = 0.1
                #action c
                self.P[s, 2, s + 1] = 0.9
                self.P[s, 2, s] = 0.1
            else:
                #action a
                self.P[s, 0, s - 1] = 0.5
                self.P[s, 0, s] = 0.5
                #action b
                self.P[s, 1, s] = 1.0
                #action c
                if s == 4: #loop around
                    self.P[s, 2, 0] = 0.5
                else:
                    self.P[s, 2, s + 1] = 0.5
                self.P[s, 2, s] = 0.5

        self.initial_distribution = np.zeros(self.nState)
        self.initial_distribution[0] = 1.
        self.rng = np.random.RandomState(SEED)

    def get_name(self):
        return 'Ring'
    
    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        prob_next_state = self.P[self.state, action]
        next_state = self.rng.multinomial(1, prob_next_state).nonzero()[0][0]
        reward = self.R[self.state, action, next_state]
        self.state = next_state
        return next_state, reward, False, {}