import numpy as np
from envs.chain import Chain
from envs.basic_envs import State2MDP, GarnetMDP
from envs.ring import Ring
from envs.randomfrozenlake import RandomFrozenLakeEnv
from envs.cliffwalking import CliffWalkingEnv

def find_optimal_policies(envs_list):
    pass

if __name__ == "__main__":
    seed = 0
    Gparams = (2,2)
    nState = 2
    nAction = 2
    prob_slip = 0.1
    discount = 0.99 
    envs_list = [
        GarnetMDP(nState, nAction, Gparams),
        State2MDP(seed),
        Chain(nState, prob_slip, discount, seed),
        Ring(seed),
        RandomFrozenLakeEnv(seed, map_name=None)
    ]
    find_optimal_policies(envs_list)