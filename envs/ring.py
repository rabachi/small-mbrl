class Ring(object):
    def __init__(self, SEED) -> None:
        # super(Ring, self).__init__()
        self.nState = 5
        self.nAction = 3
        self.state = 0
        self.P = []
        self.R = []
        self.initial_distribution = np.ones(self.nState) * 1e-6
        self.initial_distribution[0] = 1. - 1e-6*(self.nState - 1)
        self.rng = np.random.RandomState(SEED)

    def get_name(self):
        return 'Ring'
    
    def reset(self):
        self.state = 0

    def step(self, action):
        pass