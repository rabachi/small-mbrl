import numpy as np

class DoubleLoop:
    def __init__(self, seed) -> None:
        self.nState = 9 
        self.nAction = 2
        dist = np.zeros(self.nState)
        dist[0] = 1.0
        self.initial_distribution = dist
        self.state = 0   
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        if self.state == 0:
            if action == 0:
                next_state = 1
                reward = 0
            else:
                next_state = 5
                reward = 0
        elif self.state in [1, 2, 3]:
            next_state = self.state + 1
            reward = 0
        elif self.state == 4:
            next_state = 0
            reward = 1
        elif self.state in [5, 6, 7]:
            if action == 0:
                next_state = 0
                reward = 0
            elif action == 1:
                next_state = self.state + 1 
                reward = 0
        elif self.state == 8:
            next_state = 0
            reward = 2
        else:
            raise ValueError(f'State {self.state} is not defined')
        self.state = next_state
        return next_state, reward, False, {}

    def get_name(self):
        return "DoubleLoop"