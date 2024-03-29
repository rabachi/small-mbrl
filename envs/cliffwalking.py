import sys
from contextlib import closing
from io import StringIO
from typing import Optional

import numpy as np

from gym import Env, spaces

UP = 0
RIGHT = 1
DOWN = 2
LEFT = 3


class CliffWalkingEnv(Env):
    """
    This is a simple implementation of the Gridworld Cliff
    reinforcement learning task.

    Adapted from Example 6.6 (page 106) from [Reinforcement Learning: An Introduction
    by Sutton and Barto](http://incompleteideas.net/book/bookdraft2018jan1.pdf).

    With inspiration from:
    https://github.com/dennybritz/reinforcement-learning/blob/master/lib/envs/cliff_walking.py

    ### Description
    The board is a 4x12 matrix, with (using NumPy matrix indexing):
    - [3, 0] as the start at bottom-left
    - [3, 11] as the goal at bottom-right
    - [3, 1..10] as the cliff at bottom-center

    If the agent steps on the cliff it returns to the start.
    An episode terminates when the agent reaches the goal.

    ### Actions
    There are 4 discrete deterministic actions:
    - 0: move up
    - 1: move right
    - 2: move down
    - 3: move left

    ### Observations
    There are 3x12 + 1 possible states. In fact, the agent cannot be at the cliff, nor at the goal (as this results the end of episode). They remain all the positions of the first 3 rows plus the bottom-left cell.
    The observation is simply the current position encoded as [flattened index](https://numpy.org/doc/stable/reference/generated/numpy.unravel_index.html).

    ### Reward
    Each time step incurs -1 reward, and stepping into the cliff incurs -100 reward.

    ### Arguments

    ```
    gym.make('CliffWalking-v0')
    ```

    ### Version History
    - v0: Initial version release
    """

    metadata = {"render_modes": ["human", "ansi"], "render_fps": 4}

    def __init__(self, seed):
        self.shape = (4, 12)
        self.start_state_index = np.ravel_multi_index((3, 0), self.shape)

        self.nS = np.prod(self.shape)
        self.nA = 4

        # Cliff Location
        self._cliff = np.zeros(self.shape, dtype=bool)
        self._cliff[3, 1:-1] = True

        # Calculate transition probabilities and rewards
        self.P = {}
        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            self.P[s] = {a: [] for a in range(self.nA)}
            self.P[s][UP] = self._calculate_transition_prob(position, [-1, 0])
            self.P[s][RIGHT] = self._calculate_transition_prob(position, [0, 1])
            self.P[s][DOWN] = self._calculate_transition_prob(position, [1, 0])
            self.P[s][LEFT] = self._calculate_transition_prob(position, [0, -1])

        # Calculate initial state distribution
        # We always start in state (3, 0)
        self.initial_state_distrib = np.zeros(self.nS)
        self.initial_state_distrib[self.start_state_index] = 1.0

        self.observation_space = spaces.Discrete(self.nS)
        self.action_space = spaces.Discrete(self.nA)
        self.rng = np.random.RandomState(seed)

    def _limit_coordinates(self, coord):
        """
        Prevent the agent from falling out of the grid world
        :param coord:
        :return:
        """
        coord[0] = min(coord[0], self.shape[0] - 1)
        coord[0] = max(coord[0], 0)
        coord[1] = min(coord[1], self.shape[1] - 1)
        coord[1] = max(coord[1], 0)
        return coord

    def _calculate_transition_prob(self, current, delta):
        """
        Determine the outcome for an action. Transition Prob is always 1.0.
        :param current: Current position on the grid as (row, col)
        :param delta: Change in position for transition
        :return: (1.0, new_state, reward, done)
        """
        terminal_state = (self.shape[0] - 1, self.shape[1] - 1)
        if current == terminal_state:
            new_position = current
        else:
            new_position = np.array(current) + np.array(delta)
            new_position = self._limit_coordinates(new_position).astype(int)
        new_state = np.ravel_multi_index(tuple(new_position), self.shape)
        if self._cliff[tuple(new_position)]:
            # return [(1.0, self.start_state_index, -100, False)]
            return [(0.8, self.start_state_index, -100, False)]

        is_done = False#tuple(new_position) == terminal_state
        r = 1 if tuple(new_position) == terminal_state else -1

        return [(1.0, new_state, r, is_done)]
        # return [(0.8, new_state, r, is_done)]

    def get_name(self):
        return "CliffWalking"

    @property
    def nState(self):
        return self.observation_space.n
    
    @property
    def nAction(self):
        return self.action_space.n

    @property
    def initial_distribution(self):
        return self.initial_state_distrib

    def step(self, a):
        # transitions = self.P[self.s][a]
        action_prob = 0.4
        a_probs = np.ones(self.nAction) * (1-action_prob)/(self.nAction-1)
        a_probs[a] = action_prob
        a_real = self.rng.multinomial(1, a_probs).nonzero()[0][0]
        transitions = self.P[self.s][a_real]
        i = self.rng.multinomial(1, [t[0] for t in transitions]).nonzero()[0][0]
        p, s, r, d = transitions[i]
        self.s = s
        self.lastaction = a
        return (int(s), r, d, {"prob": p})

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None
    ):
        # super().reset(seed=seed)
        self.s = self.rng.multinomial(1, self.initial_state_distrib).nonzero()[0][0]
        self.lastaction = None
        if not return_info:
            return int(self.s)
        else:
            return int(self.s), {"prob": 1}

    def reset_to_state(
        self,
        state
    ):
        self.s = state
        # next_state, _, _, _ = self.step(action)
        # self.s = next_state
        return int(self.s)

    def render(self, mode="human"):
        outfile = StringIO() if mode == "ansi" else sys.stdout

        for s in range(self.nS):
            position = np.unravel_index(s, self.shape)
            if self.s == s:
                output = " x "
            # Print terminal state
            elif position == (3, 11):
                output = " T "
            elif self._cliff[position]:
                output = " C "
            else:
                output = " o "

            if position[1] == 0:
                output = output.lstrip()
            if position[1] == self.shape[1] - 1:
                output = output.rstrip()
                output += "\n"

            outfile.write(output)
        outfile.write("\n")

        # No need to return anything for human
        if mode != "human":
            with closing(outfile):
                return outfile.getvalue()