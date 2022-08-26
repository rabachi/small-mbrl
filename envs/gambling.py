import numpy as np

class Gambling:
    def __init__(self, SEED) -> None:
        self.nState = 1
        self.nAction = 3
        self.state = 0

        self.initial_distribution = np.zeros(self.nState)
        self.initial_distribution[0] = 1.
        self.rng = np.random.RandomState(SEED)

    def get_name(self):
        return 'Gambling'
    
    def reset(self):
        self.state = 0
        return self.state

    def step(self, action):
        prob_next_state = self.P[self.state, action]
        if action == 0:
            reward = self.rng.choice([5., -15.])
        elif action == 1:
            reward = self.rng.choice([1., -6.])
        elif action == 2:
            reward = 1.
        else:
            raise ValueError(f'Invalid action {action}')

        next_state = self.state
        self.state = next_state
        return next_state, reward, True, {}