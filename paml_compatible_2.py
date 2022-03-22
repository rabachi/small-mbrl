import jax.numpy as np
import jax
import numpy as onp
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator)

import signal
import sys

# from envs import *
from utils import *
from plotting_exact import *
from deploy_exact import *
from src.exact_pg import policy_performance, get_policy, \
    policy_evaluation, iterative_policy_evaluation, mdp_3states, model
from jax.experimental import optimizers
from jax.lax import stop_gradient


def get_log_policy(p_params, n_states, n_actions, temp):
    """

    :param p_params:
    :return:
    """
    return log_softmax(p_params.reshape(n_states, n_actions), temp)#.reshape(-1,)


def Garnet(seed, StateSize=5, ActionSize=2, GarnetParam=(1,1)):
    onp.random.seed(seed)

    P = [onp.matrix(onp.zeros( (StateSize,StateSize))) + 1e-6 for act in range(ActionSize)]
    R = onp.zeros( (StateSize,1) )
    b_P = GarnetParam[0] # branching factor
    b_R = GarnetParam[1] # number of non-zero rewards
    for act in range(ActionSize):
        for ind in range(StateSize):
            pVec = onp.zeros(StateSize) + 1e-6
            p_vec = onp.append(onp.random.uniform(0,1,b_P - 1),[0,1])
            p_vec = onp.diff(onp.sort(p_vec))
            pVec[onp.random.choice(StateSize, b_P, replace=False)] = p_vec#[:,np.newaxis]
            pVec /= sum(pVec)
            P[act][ind,:] = pVec

    R[onp.random.choice(StateSize,b_R, replace = False)] = onp.random.uniform(0,1,b_R)[:,onp.newaxis]
    R = onp.tile(onp.array(R), (1, ActionSize))
    return onp.array(P), R, 0.9


def chain(n_states):
    # n_states-state chain, 3 actions
    # P has shape 3 x n_states x n_states
    # R has shape n_states x 3
    prob_a0 = 1.0 - 1e-5
    prob_a1 = 1.0 - 1e-5
    prob_a2 = 1.0 - 1e-5

    P_a0 = onp.eye(n_states) + 1e-5
    for i in range(n_states):
        P_a0[i, i] = 1. - ((n_states-1)*1e-5)

    P_a1_eye = onp.eye(n_states - 1) + 1e-5
    for i in range(n_states - 1):
        P_a1_eye[i, i] = 1. - ((n_states-1)*1e-5)

    P_a1_column = np.hstack((np.ones((n_states - 1, 1)) * 1e-5, P_a1_eye))
    last_state_e = onp.ones((1, n_states)) * 1e-5
    last_state_e[:,-1] = 1. - ((n_states-1)*1e-5)
    P_a1 = np.vstack((P_a1_column, last_state_e))

    P_a2 = onp.ones((n_states, n_states)) * 1e-5
    P_a2[0,0] = 1. - ((n_states-1)*1e-5)
    for i in range(n_states - 2):
        P_a2[i + 1, i] = 1. - ((n_states-1)*1e-5)
    P_a2[n_states - 1, n_states - 1] = 1. - ((n_states-1)*1e-5)

    P = np.stack((P_a0, P_a1, P_a2), axis=0)
                 
    R_0 = np.array([[1.0, 1.0, 1.0]])
    R_last = np.array([[.5, .5, 0.5]])
    R_middle = np.array([[1.0,1.0,1.0]])
    R_zeros = np.zeros(((n_states - 3)//2, 3))
    R = np.vstack((R_0, R_zeros, R_middle, R_zeros, R_last))


    # print(np.ones((n_states, 1)).shape, np.zeros((n_states, n_states - 1)).shape)
    # P_a = np.hstack((np.zeros((n_states, n_states - 1)), np.ones((n_states, 1))))
    # P = np.stack((P_a, P_a, P_a), axis=0)
    print(P)
    return P, R, 0.9


def BiasedGridworld(n_states_wall):
    '''
    n_states_wall (int): number of states on each wall of the grid
    '''
    down = 0
    right = 1
    up = 2
    left = 3 
    
    actions = [down, right, up, left] #the action names aren't exactly correct because I planned the names based on pixel configuration but these are arrays... The "real" thing is towards right and down but I planned towards right and up. (so probably on up and down are switched in the name)
    P_actions = {
                    down: onp.zeros((n_states_wall, n_states_wall, n_states_wall, n_states_wall)),
                    right: onp.zeros((n_states_wall, n_states_wall, n_states_wall, n_states_wall)),
                    up: onp.zeros((n_states_wall, n_states_wall, n_states_wall, n_states_wall)),
                    left: onp.zeros((n_states_wall, n_states_wall, n_states_wall, n_states_wall))
                    } #s,s'

    R_a = onp.ones((n_states_wall, n_states_wall, len(actions))) * -1.0 #for now all actions are the same 
    R_a[n_states_wall - 1, n_states_wall - 1] = -10.0
    R_a[n_states_wall - 1, n_states_wall - 2] = 1.0
    R_a[n_states_wall - 2, n_states_wall - 1] = -10.0
    R_a[n_states_wall - 2, n_states_wall - 2] = 1.0

    for action in actions:
        for j in range(n_states_wall):
            for i in range(n_states_wall):
                if action == left:
                    if i == 0:
                        P_actions[action][i, j, i + 1, j] = 1.0
                    else:    
                        P_actions[action][i, j, i - 1, j] = 0.999
                        P_actions[action][i, j, i, j] = 0.001
                elif action == down:
                    if j == 0:
                        P_actions[action][i, j, i, j + 1] = 1.0
                    else:
                        P_actions[action][i, j, i, j - 1] = 0.999
                        P_actions[action][i, j, i, j] = 0.001
                elif action == right:
                    if i == n_states_wall - 1:
                        P_actions[action][i, j, i - 1, j] = 1.0
                    else:
                        P_actions[action][i, j, i + 1, j] = 0.999
                        P_actions[action][i, j, i, j] = 0.001
                elif action == up:
                    if (j == n_states_wall - 1): # the second condition is for the two "attractor" states
                        P_actions[action][i, j, n_states_wall - 1, n_states_wall - 2] = 0.8
                        P_actions[action][i, j, i, j] = 0.2

                    elif (j == n_states_wall - 2) and (i == n_states_wall - 1):
                        P_actions[action][i, j, i, n_states_wall - 1] = 0.8
                        P_actions[action][i, j, i, j] = 0.2
                    else:
                        P_actions[action][i, j, i, j + 1] = 0.999
                        P_actions[action][i, j, i, j] = 0.001

        print(P_actions[action].reshape((16, 16)).sum(axis=1))       
        P_actions[action] = P_actions[action].reshape((16, 16))
    
    P = np.stack(list((P_actions[a] for a in P_actions.keys())))
    # print(R_a)
    # R = onp.repeat(R_a.reshape(n_states_wall * n_states_wall, 1), len(actions), axis=1)
    R = R_a.reshape((n_states_wall*n_states_wall, len(actions)))
    # print(P_actions)
    # print(R[:,2].reshape(4,4))
    print(P)
    print(R)
    return P, R, 0.9


def NotaExample3():
    n_states = 6
    n_actions = 2

    P = onp.zeros((n_actions, n_states, n_states)) #ass'
    P[0, 0, 1] = 1.
    P[1, 0, 1] = 1.

    P[0, 1, 2] = 1.
    P[1, 1, 4] = 1.

    P[0, 2, 3] = 1.
    P[1, 2, 3] = 1.

    P[0, 4, 5] = 1.
    P[1, 4, 5] = 1.

    #terminal states
    P[0, 3, 3] = 1.
    P[1, 3, 3] = 1.
    P[0, 5, 5] = 1.
    P[1, 5, 5] = 1.

    R = onp.zeros((n_states, n_actions))
    R[0, 0] = 1.
    R[0, 1] = 1.
    
    R[1, 0] = 2.
    R[1, 1] = -2.

    R[2, 0] = 100.
    R[2, 1] = 100.
    
    R[3, 0] = -100.
    R[3, 1] = -100.
    return P, R, 0.9


def amfexample(seed, n_states):
    # n_states = 20
    onp.random.seed(seed)
    n_actions = n_states - 1

    P = onp.ones((n_states, n_states)) * 1e-12#ss'
    P[0,0] = 0 + 1e-12
    P[1:,0] = 1. - 1e-12 * n_states

    # logprobs = onp.flip(onp.sort(onp.random.exponential(size=(1, n_states - 1))), axis=1)

    # logprobs = onp.arange(n_states - 1, 0, -0.5)[:n_states-1].reshape((1, n_states - 1))
    # logprobs = onp.arange(n_states-1, np.floor((n_states-1)/2) + 0.5, -0.5).reshape((1, n_states - 1))
    # print(logprobs.shape)
    # # logprobs[:, 7] = 5.
    # # zeroth = logprobs[:,0].copy()
    # # logprobs[:,0] = logprobs[:,1]
    # # logprobs[:,1] = zeroth
    # # logprobs[:, 3] = 10.
    # # logprobs = onp.ones((1,n_states-1)) * (1.0 / len(P[0, 1:]))
    # # P[0, 1:] = (1.0 - 1e-6)/len(P[0, 1:])#softmax(logprobs)
    
    # P[0, 1:] = softmax(logprobs) #- (1e-6 / len(P[0, 1:]))
    make_zero = len(P[0,:] - 3)
    P[0, 1] = 0.49 - (1e-12)*make_zero
    halfpoint = n_states//2 - 2
    P[0, halfpoint] = 0.51 - (1e-12)*make_zero


    # P[0, -1] = 1.
    # P[0, 1:] = (1. - 1e-6) / (n_states - 1)
    # P[0, 2:] = (1 - P[0,1])/len(P[0,2:])

    # probs = onp.ones((1, n_states - 1)) * 1e-6
    # probs[:,0] = 0.41
    # probs[:,1] = 0.39
    # probs[:,2:] = (1. - onp.sum(probs[0,:2])) / len(probs[0,2:])
    # P[0, 1:] = np.array(probs)

    # probs[:,0] = 0.4 - 1e-6
    # probs[:,1] = 0.5 - 1e-6
    # probs[:,2] = 0.1 - 1e-6
    # probs[:,3:] = (1. - onp.sum(probs[0,:3])) / len(probs[0,3:])
    # P[0, 1:] = np.array(probs)

    P = onp.tile(P, (n_actions, 1, 1))
    print(onp.sum(P,axis=2))
    print(P[0])
    sorted_rewards = (onp.arange(n_states - 1) + 1) * -1.
    onp.random.shuffle(sorted_rewards[1:])
    R_row = onp.tile(sorted_rewards, (n_actions, 1))
    R_no_x0 = onp.transpose((1 - onp.eye(n_states - 1)) * R_row + onp.eye(n_states - 1))
    # R_no_x0 = onp.eye(n_states - 1)*2. - onp.ones((n_states - 1, n_actions))
    R = onp.vstack((onp.zeros((1, n_states - 1)), R_no_x0))
    # R = onp.hstack((onp.zeros((n_states, 1)), R))
    print(R)
    return P, R, 0.9


def aggexample():
    P_per_action = onp.transpose(onp.array([
        [0.98         , 0.496 - 1e-12, 0.0025 - 1e-12, 0.0  + 1e-12],
        [0.005        , 0.01 - 1e-12, 0.0   + 1e-12 , 0.0  + 1e-12],
        [0.015 - 1e-12, 0.494  + 1e-12, 0.995        , 0.99 - 1e-12],
        [0.    + 1e-12, 0.0  + 1e-12, 0.0025         , 0.01 - 1e-12]
    ]))
    P = onp.tile(P_per_action, (2, 1, 1))

    R = onp.array([
        [0.5, .15],
        [0.005, 0.005],
        [-0.005, -0.5],
        [-0.5, -0.5],
        ])
    return P, R, 0.9


def raexample():
    # 2(a) x 7(s) x 7(s')
    P_right = onp.zeros((7,7)) + 1e-12
    P_right[0, 2] = 0.8 - 5*1e-12
    P_right[0, 1] = 0.2 #- 5*1e-6

    # P_right[1, 2] = 0.05
    P_right[1, 3] = 0.6 - 5*1e-12
    P_right[1, 4] = 0.4

    P_right[2, 5] = 0.1 - 5*1e-12
    P_right[2, 6] = 0.9

    P_right[3, 0] = 1.0 - 6*1e-12
    P_right[4, 0] = 1.0 - 6*1e-12
    P_right[5, 0] = 1.0 - 6*1e-12
    P_right[6, 0] = 1.0 - 6*1e-12

    P_left = onp.zeros((7,7)) + 1e-12
    P_left[0, 1] = 0.8 - 5*1e-12
    P_left[0, 2] = 0.2 #- 5*1e-6

    # P_left[1, 2] = 0.05
    P_left[1, 4] = 0.7 - 5*1e-12
    P_left[1, 3] = 0.3

    P_left[2, 6] = 0.1 - 5*1e-12
    P_left[2, 5] = 0.9

    P_left[3, 0] = 1.0 - 6*1e-12
    P_left[4, 0] = 1.0 - 6*1e-12
    P_left[5, 0] = 1.0 - 6*1e-12
    P_left[6, 0] = 1.0 - 6*1e-12

    P = onp.stack((P_right, P_left), axis=0)
    print(P.sum(axis=2))

    R = onp.zeros((7,2))
    R[1, 0] = 0.6
    R[1, 1] = -0.1

    R[2, 0] = -0.1
    R[2, 1] = 0.45

    R[3, 0] = 0.1
    R[3, 1] = -0.1
    R[4, 0] = R[4, 1] = -0.05

    R[5, 0] = -0.1
    R[5, 1] = 0.2
    R[6, 0] = R[6, 1] = 0.

    return P, R, 0.9


limit_q = False
def get_q(q, limit_q=False, agg_q_vector=None):
    if limit_q:
        # ret_q = onp.zeros_like(onp.array(q))
        # ret_q[1, :] = onp.array(q[1,:])
        # ret_q[10, :] = onp.array(q[10,:])
        # return np.array(ret_q)
        print(q.shape)
        print(q[agg_q_vector[0]].shape)
        q[agg_q_vector[0]]
        aggregated = q[agg_q_vector[0], :].reshape(-1,1)
        for idx in range(1, agg_q_vector.shape[0]):
            bin_idx += (bin_idx == agg_q_vector[idx])
            aggregated = np.concatenate((
                aggregated,
                q[agg_q_vector[idx], :].reshape(-1,1)
                ))
        return np.array(aggregated)
    else:
        # print('Q function', q)
        return q

def get_m_params(params, env, env_name, n_states, n_actions, agg_vector=None):
    # known_params = env[0][0,1:,:]
    # model_part = np.array(params[:-n_states*n_actions].reshape(1,), copy=True)
    # first_num = (n_states - 2) // 2 
    # second_num = n_states - 2 - first_num
    
    # first_part = np.tile((1 - model_part)/(n_states - 2), first_num)
    # second_part = np.tile((1 - model_part)/(n_states - 2), second_num)

    # aggregated = np.concatenate((first_part, model_part, second_part))
    # # print(aggregated)

    # plus_1 = np.concatenate((np.ones(1)*1e-6, aggregated)).reshape(1,-1)
    # per_action = np.vstack((plus_1, known_params))
    # model = np.tile(per_action, (n_actions, 1, 1)).reshape(-1)
    # new_params = np.concatenate((model, params[-n_states*n_actions:]))
    # return new_params
    if agg_vector is not None and env_name == 'amfexample':
        known_params = env[0][0,1:,:]
        rewards = env[1].reshape(-1)
        # aggregated = params[:-n_states*n_actions][agg_vector[0]].reshape(1,) 
        aggregated = params[agg_vector[0]].reshape(1,) 
        for idx in range(1, agg_vector.shape[0]):
            aggregated = np.concatenate((
                aggregated,
                # params[:-n_states*n_actions][agg_vector[idx]].reshape(1,)
                params[agg_vector[idx]].reshape(1,)
                ))
        plus_1 = np.concatenate((np.ones(1)*1e-6, aggregated)).reshape(1,-1)
        per_action = np.vstack((plus_1, known_params))
        model = np.tile(per_action, (n_actions, 1, 1)).reshape(-1)
        # new_params = np.concatenate((model, params[-n_states*n_actions:]))
        new_params = np.concatenate((model, rewards))
        return new_params
    elif env_name == 'aggexample':
        params = params.reshape(1, 1, 2)
        unagg_x = np.repeat(params/2., 4, axis=1)
        unagg_xprime = np.repeat(unagg_x, 2, axis=2)
        model = np.tile(unagg_xprime, (2, 1, 1)).reshape(-1)
        rewards = env[1].reshape(-1)
        new_params = np.concatenate((model, rewards)).reshape(-1)
        # print(new_params)
        return new_params
    elif env_name == 'raexample':
        params = params.reshape(n_actions, n_states - 1, n_states)
        # print(params[:,0].reshape(n_actions,-1, n_states).shape, params[:,1].reshape(n_actions,-1, n_states).shape, params[:,1:].shape)
        model = np.concatenate((params[:,0].reshape(n_actions, -1, n_states)/2, params[:,0].reshape(n_actions, -1, n_states)/2, params[:,1:]), axis=1).reshape(-1)
        # model = np.concatenate((
        #     params[:,0].reshape(n_actions, -1, n_states), 
        #     params[:,1].reshape(n_actions, -1, n_states), 
        #     params[:,0].reshape(n_actions, -1, n_states), 
        #     params[:,2:]
        #     ), axis=1).reshape(-1)
        rewards = env[1].reshape(-1)
        new_params = np.concatenate((model, rewards)).reshape(-1)
        return new_params
    else:
        rewards = env[1].reshape(-1)
        new_params = np.concatenate((params, rewards)).reshape(-1)
        return new_params

def get_policy_params(_policy_params):
    policy_params_1 = onp.repeat(_policy_params[1].reshape(1,-1), 2, axis=0)
    policy_params_2 = onp.repeat(_policy_params[2].reshape(1,-1), 4, axis=0)
    policy_params = onp.concatenate(
        (
        _policy_params[0].reshape(1,-1), 
        policy_params_1, 
        policy_params_2, 
        ), axis=0).reshape(-1)
    return policy_params

def get_pg(pg_, n_states, n_actions):
    pg_ = pg_.reshape(n_states, n_actions)
    # print(pg_)
    pg = np.concatenate((
        pg_[0,:], pg_[1,:], pg_[3,:]
        ))
    return pg

def get_pi_log_grad(pi_log_grad, n_states, n_actions):
    pi_log_grad_a_1 = onp.repeat(pi_log_grad[1].reshape(1, n_actions, -1, n_actions), 2, axis=0).reshape(2, n_actions, -1, n_actions)
    pi_log_grad_a_2 = onp.repeat(pi_log_grad[2].reshape(1, n_actions, -1, n_actions), 4, axis=0).reshape(4, n_actions, -1, n_actions)

    pi_log_grad_a = np.concatenate((
        pi_log_grad[0].reshape(1, n_actions, -1, n_actions),
        pi_log_grad_a_1.reshape(2, n_actions, -1, n_actions),
        pi_log_grad_a_2.reshape(4, n_actions, -1, n_actions)
        ), axis=0)

    pi_log_grad_b_1 = onp.repeat(pi_log_grad_a[:,:,1].reshape(-1, n_actions, 1, n_actions), 2, axis=2).reshape(n_states, n_actions, 2, n_actions)

    pi_log_grad_b_2 = onp.repeat(pi_log_grad_a[:,:,2].reshape(-1, n_actions, 1, n_actions), 4, axis=2).reshape(n_states, n_actions, 4, n_actions)

    pi_log_grad = np.concatenate((
        pi_log_grad_a[:,:,0].reshape(n_states, n_actions, 1, n_actions),
        pi_log_grad_b_1.reshape(n_states, n_actions, 2, n_actions),
        pi_log_grad_b_2.reshape(n_states, n_actions, 4, n_actions)
        ), axis=2).reshape((n_states, n_actions, n_states * n_actions))
    return pi_log_grad


def mbrl_loop_exact(args, env):
    '''
    Loop:
        2. Train model using PAML
        3. Update policy using learned model

    '''
    # training = 'rand'
    if 'chain' in args.env_name:
        params = args.env_name.split(',')
        env = chain(int(params[1]))
    elif args.env_name == 'mdp3states':
        env = mdp_3states()
    elif 'BiasedGridworld' in args.env_name:
        params = args.env_name.split(',')
        env = BiasedGridworld(int(params[1]))
    # elif 'garnet' in args.env_name:
    #     params = args.env_name.split(',')
    #     env = Garnet(seed, StateSize=params[1], ActionSize=params[2], GarnetParam=(params[3],params[4]))

    def signal_handler(sig, frame):
        print('Terminating ...')
        if loss_type != 'true-env':
            # save model
            np.save(f'{args.save_sub_dir}model_params.npy', model_params)
            np.save(f'{args.save_sub_dir}agg_vector.npy', agg_vector)
        #save policy params
        np.save(f'{args.save_sub_dir}policy_params.npy', policy_params)
        #save Q params
        np.save(f'{args.save_sub_dir}Q_params.npy', Q_params)
        sys.exit(0) 

    signal.signal(signal.SIGINT, signal_handler)
    key = jax.random.PRNGKey(0)

    k = args.model_ksteps if args.model_ksteps != 'inf' else 0 #is this necessary?
    iters_PE = args.iters_PE if args.iters_PE != 'inf' else 0

    temperature = args.temperature
    n_actions, n_states = env[0].shape[:2]
    constraint = args.constraint

    model_lr, policy_lr, Q_lr = args.lr_model, args.lr_policy, args.lr_q#0.05, 0.01, 0.05

    model_iters = args.model_iters #30
    policy_iters = args.policy_iters#10
    total_iters = args.epochs

    loss_type = args.model_type
    limit_model = False
    if args.env_name =='amfexample':
        limit_model = True #this is separate from aggregation, this is for amfexample, where we only want to learn one aspect of the model
    agg_vector = args.agg_vector
    print(agg_vector)

    agg_q_vector = args.agg_vector
    if args.env_name == 'amfexample' and (args.model_aggregate is not None) and (args.model_aggregate <= n_states):
        # model_params = np.zeros((1*1*(n_states-2) + n_states*n_actions))
        # model_params = np.zeros((1*1*(args.model_aggregate) + n_states*n_actions)) + 0.5
        model_params = np.zeros((1*1*(args.model_aggregate))) + 0.5
        # agg_vector = get_random_aggs(n_states - 1, args.model_aggregate)
    elif args.env_name == 'raexample':
        model_params = np.zeros((2 * 6 * 7)) #+ 0.0005
    elif args.env_name == 'aggexample' and args.model_aggregate == True:
        model_params = np.zeros((1*1*2)) + 0.5
    else:
        model_params = np.zeros((n_states*n_actions*n_states))
        # model_params = np.zeros((n_states*n_actions*n_states))
        args.model_aggregate = n_states
    
    # model_params = np.zeros((1 + n_states*n_actions)) + 0.5
    # model_params[1:] = env[1].reshape(-1)
    # model_params = np.array(model_params)
    

    # samples_from_unif = jax.random.uniform(key, shape=(1, n_states * n_actions))
    # key,subkey = jax.random.split(key)
    # Q_params = np.ones((n_states * n_actions)) * jax.random.uniform(subkey, shape=samples_from_unif.shape)
    model_opt_init, model_opt_update, model_get_params = optimizers.adam(model_lr)
    model_opt_state = model_opt_init(model_params)

    # pi_opt_init, pi_opt_update, pi_get_params = optimizers.adam(policy_lr)
    # pi_opt_state = pi_opt_init(policy_params)

    # Q_opt_init, Q_opt_update, Q_get_params = optimizers.adam(Q_lr)
    # Q_opt_state = Q_opt_init(Q_params)

    initial_distribution = onp.zeros(n_states)
    initial_distribution[0] = 1.0
    # initial_distribution[2] = 0.5

    def policy_perf_model(p_params, m_params):
        return policy_performance(model(m_params, n_states, n_actions, temperature, limit_model=limit_model), get_policy(p_params, n_states, n_actions, temperature), initial_distribution)

    def policy_perf_mdp(p_params):
        return policy_performance(env, get_policy(p_params, n_states, n_actions, temperature), initial_distribution)

    def get_Q_loss(params, true_q, p_params):
        Q_pi_w = np.einsum('sap,kp->sa', jax.jacrev(log_softmax)(p_params.reshape(n_states, n_actions), temperature).reshape(n_states, n_actions, n_actions*n_states), params)
        norm_pi = np.einsum('sa,sa->s', get_policy(p_params, n_states, n_actions, temperature), (Q_pi_w - true_q)**2)  
        return np.sum(norm_pi)

    def expanded_pg(p_params, m_params, qf, use_true=False):
        m_params = get_m_params(m_params, env, args.env_name, n_states, n_actions, agg_vector=agg_vector)

        if use_true:
            del m_params
            model_p, model_r, discount = env
        else:
            model_p, model_r, discount = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)

        #if state agg
        origin_p_params = p_params[:].reshape(-1, n_actions)
        if (p_params.reshape(-1).shape[0] != n_states * n_actions):
            p_params = get_policy_params(origin_p_params)

        policy = get_policy(p_params, n_states, n_actions, temperature)
        model_ppi = np.einsum('ast,sa->st', model_p, policy)
        # print(policy)
        if (origin_p_params.reshape(-1).shape[0] != n_states * n_actions):
            # print('state-agg')
            pi_log_grad = jacrev(log_softmax)(origin_p_params.reshape(-1, n_actions), temperature)

            pi_log_grad = get_pi_log_grad(pi_log_grad, n_states, n_actions)

            # pi_log_grad = jacrev(log_softmax)(origin_p_params.reshape(1, n_actions), temperature)
            # pi_log_grad = np.tile(pi_log_grad, (n_states, 1, n_states, 1)).reshape((n_states,n_actions, n_states * n_actions))
            # print(pi_log_grad.shape)
        else:
            pi_log_grad = jacrev(log_softmax)(p_params.reshape(n_states, n_actions), temperature).reshape(n_states, n_actions, n_states * n_actions)

        H_theta = np.multiply(np.expand_dims(policy, axis=2), np.multiply(pi_log_grad, np.expand_dims(qf, axis=2))).sum(axis=1) #sum over actions
        # print(H_theta.shape)
        inv_H = np.linalg.inv(np.eye(model_ppi.shape[0]) - discount*model_ppi) @ H_theta
        # print(np.linalg.inv(np.eye(model_ppi.shape[0]) - discount*model_ppi))
        pg = initial_distribution @ inv_H
        # print(f'PG:{pg}')
        if (origin_p_params.reshape(-1).shape[0] != n_states * n_actions):
            #pg = pg[:n_actions]
            # pg = np.concatenate((
            #     pg[0:1], pg[3:]
            #     ))
            pg = get_pg(pg, n_states, n_actions)
        # print(f'AFTER PG:{pg}')
        return pg

    def pg_calculated(p_params, m_params):
        policy = get_policy(p_params, n_states, n_actions, temperature)
        pi_log_grad = jacrev(get_log_policy)(p_params, n_states, n_actions, temperature).squeeze() 

        model_p, model_r, discount = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)
        model_ppi = np.einsum('ast,sa->st', model_p, policy)
        minv = np.linalg.inv(np.eye(n_states) - discount * model_ppi)

        _, model_q = policy_evaluation((model_p, model_r, discount), policy)

        model_grad_term = np.einsum('sat,sa->sat', pi_log_grad, model_q)
        model_exp_grad_term = np.einsum('sa,sat->st', policy, model_grad_term)

        model_grad = initial_distribution @ np.einsum('sp,pt->st', minv, model_exp_grad_term) 
        return model_grad

    key,subkey = jax.random.split(key)
    samples_from_unif = jax.random.uniform(subkey, shape=(1, n_actions))
    key,subkey = jax.random.split(key)
    Q_params_random = jax.random.uniform(subkey, shape=(1, n_states*n_actions))
    key,subkey = jax.random.split(key)
    samples_from_unif_model = jax.random.normal(subkey, shape=(n_states*n_actions*n_states + n_states*n_actions,1))

    true_pi_grads = []
    compare_true_policy_params = []
    compare_Q_params = []

    # model_params = np.ones((n_states*n_actions*n_states + n_states*n_actions,1))*samples_from_unif_model

    # policy_params = np.ones((n_states * n_actions)) * samples_from_unif

    # true_policy_params = onp.ones((1, n_actions)) * samples_from_unif
    # policy_params = np.tile(np.array(true_policy_params), (n_states, 1)).reshape(-1)

    true_policy_params = onp.ones((3, n_actions)) * samples_from_unif
    policy_params = get_policy_params(true_policy_params)
    # policy_params = np.tile(np.array(true_policy_params), (1, 1)).reshape(-1)
    # if n_actions > 3: # this is mainly for amfexample to show what happens with a bad initial policy
    #     true_policy_params[:, 3] = 10.
    #     true_policy_params[:, 2] = 5.
    
    #PARTITION
    # if 'BiasedGridworld' in args.env_name:
    #     n_states_wall = int(onp.sqrt(n_states))
    #     policy_params = policy_params.reshape((n_states_wall, n_states_wall, n_actions))
    #     new_policy_params = onp.zeros((n_states_wall, n_states_wall, n_actions))
    #     for j in range(0, n_states_wall, 2):
    #         for i in range(0, n_states_wall, 2):
    #             new_policy_params[i, j] = policy_params[i, j]
    #             new_policy_params[i + 1, j] = policy_params[i, j]
    #             new_policy_params[i, j + 1] = policy_params[i, j]
    #             new_policy_params[i + 1, j + 1] = policy_params[i, j]

    #     policy_params = np.array(new_policy_params.reshape(-1))

    Q_params = np.ones(policy_params.shape) * Q_params_random

    model_opt_init, model_opt_update, model_get_params = optimizers.sgd(model_lr)
    model_opt_state = model_opt_init(model_params)

    pi_opt_init, pi_opt_update, pi_get_params = optimizers.adam(policy_lr)
    pi_opt_state = pi_opt_init(true_policy_params.reshape(-1))

    Q_opt_init, Q_opt_update, Q_get_params = optimizers.adam(Q_lr)
    Q_opt_state = Q_opt_init(Q_params)
    
    data_dict = {
                 'Iter': 0,
                 'mle_model_loss': [],
                 'paml_model_loss': [],
                 'value_loss': [],
                 'policy_perf_mdp': [],
                 'cpg': []
                 }
    fieldnames = [key for key, _ in data_dict.items()]
    csv_logger = load_logger(args, fieldnames)
    print(f'Loss: {loss_type}')

    def mle_model_objective(m_params, p_params, mdp, Q_params, k, value_losses):
        m_params = get_m_params(m_params, env, args.env_name, n_states, n_actions, agg_vector=agg_vector)
        Phat, rhat, _ = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)
        p, r, _ = mdp
        # print(rhat)
        # print(r)
        # print(np.sum((r - rhat)**2))
        return kl_divergence(p, Phat) #+ np.sum((r - rhat)**2))

    def paml_compatible_model_loss(m_params, p_params, mdp, Q_params, k, value_losses):
        policy = get_policy(p_params, n_states, n_actions, temperature)
        pi_log_grad = jacrev(get_log_policy)(p_params, n_states, n_actions, temperature).squeeze() 

        # pi_log_grad2 = jacrev(log_softmax)(p_params.reshape(n_states, n_actions), temperature).squeeze()
        fisher_states = [] 
        for i in range(n_states):
            pi_si = policy[i]

            pi_log_grad_si = pi_log_grad[i]
            outer_si_a0 = np.outer(pi_log_grad_si.squeeze()[0],pi_log_grad_si.squeeze()[0])
            outer_si_a1 = np.outer(pi_log_grad_si.squeeze()[1],pi_log_grad_si.squeeze()[1])
            outer_si_a2 = np.outer(pi_log_grad_si.squeeze()[2],pi_log_grad_si.squeeze()[2])

            fisher_si_0 = pi_si[0] * outer_si_a0
            fisher_si_1 = pi_si[1] * outer_si_a1
            fisher_si_2 = pi_si[2] * outer_si_a2

            fisher_si = fisher_si_0 + fisher_si_1 + fisher_si_2
            fisher_states.append(fisher_si)

        fisher = np.stack(fisher_states)
        model_p, model_r, discount = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)
        model_ppi = np.einsum('ast,sa->st', model_p, policy)
        minv = np.linalg.inv(np.eye(n_states) - discount * model_ppi)

        p, r, discount = env

        true_ppi = np.einsum('ast,sa->st', p, policy)
        tinv = np.linalg.inv(np.eye(n_states) - discount * true_ppi)

        loss = np.einsum('is,stl->itl', tinv - minv, fisher)
        return np.sum(np.linalg.norm(loss)) + np.sum((r - model_r)**2)

    def paml_vaml_model_loss(m_params, p_params, mdp, Q_params, k, value_losses):
        policy = get_policy(p_params, n_states, n_actions, temperature)
        pi_log_grad = jacrev(get_log_policy)(p_params, n_states, n_actions, temperature).squeeze() 

        model_p, model_r, discount = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)
        model_ppi = np.einsum('ast,sa->st', model_p, policy)
        minv = np.linalg.inv(np.eye(n_states) - discount * model_ppi)

        p, r, discount = env

        true_ppi = np.einsum('ast,sa->st', p, policy)
        tinv = np.linalg.inv(np.eye(n_states) - discount * true_ppi)
        
        _, true_q = policy_evaluation(env, policy)
        _, model_q = policy_evaluation((model_p, model_r, discount), policy)                 
        # model_q = np.einsum('sap,kp->sa', jax.jacrev(log_softmax)(p_params.reshape(n_states, n_actions), temperature).reshape(n_states, n_actions, n_actions*n_states), Q_params) 

        model_grad_term = np.einsum('sat,sa->sat', pi_log_grad, model_q)
        model_exp_grad_term = np.einsum('sa,sat->st', policy, model_grad_term)

        model_grad = initial_distribution @ np.einsum('sp,pt->st', minv, model_exp_grad_term) 

        true_grad_term = np.einsum('sat,sa->sat', pi_log_grad, true_q)
        true_exp_grad_term = np.einsum('sa,sat->st', policy, true_grad_term)
        
        true_grad = initial_distribution @ np.einsum('sp,pt->st', tinv, true_exp_grad_term) 

        paml_loss = np.linalg.norm(true_grad - model_grad)**2 
        value_loss = np.linalg.norm(true_q - model_q)**2
        
        # paml_losses.append(jax.lax.stop_gradient(paml_loss))
        value_losses.append(float(jax.lax.stop_gradient(value_loss)))
        return paml_loss + value_loss

    def vaml_loss(m_params, p_params, mdp, Q_params, k, value_losses):
        policy = get_policy(p_params, n_states, n_actions, temperature)
        model_p, model_r, discount = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)

        _, true_q = policy_evaluation(env, policy)
        _, model_q = policy_evaluation((model_p, model_r, discount), policy)

        value_loss = np.linalg.norm(true_q - model_q)**2
        return value_loss

    def paml_regular_model_loss(m_params, p_params, mdp, Q_params, k, value_losses):
        policy = get_policy(p_params, n_states, n_actions, temperature)
        pi_log_grad = jacrev(get_log_policy)(p_params, n_states, n_actions, temperature).squeeze() 

        model_p, model_r, discount = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)
        model_ppi = np.einsum('ast,sa->st', model_p, policy)
        minv = np.linalg.inv(np.eye(n_states) - discount * model_ppi)

        p, r, discount = env

        true_ppi = np.einsum('ast,sa->st', p, policy)
        tinv = np.linalg.inv(np.eye(n_states) - discount * true_ppi)
        
        _, true_q = policy_evaluation(env, policy)
        _, model_q = stop_gradient(policy_evaluation((model_p, model_r, discount), policy))       

        # print(f'kl: {float(stop_gradient(kl_divergence(p, model_p)))}, q diff: {float(stop_gradient(np.linalg.norm((true_q - model_q)**2)))}')

        model_grad_term = np.einsum('sat,sa->sat', pi_log_grad, model_q)
        model_exp_grad_term = np.einsum('sa,sat->st', policy, model_grad_term)

        model_grad = initial_distribution @ np.einsum('sp,pt->st', minv, model_exp_grad_term) 

        true_grad_term = np.einsum('sat,sa->sat', pi_log_grad, true_q)
        true_exp_grad_term = np.einsum('sa,sat->st', policy, true_grad_term)
        
        true_grad = initial_distribution @ np.einsum('sp,pt->st', tinv, true_exp_grad_term) 

        paml_loss = np.linalg.norm(true_grad - model_grad)**2 

        return paml_loss + np.sum((r - model_r)**2)

    def paml_infstep_dist_trueQ(m_params, p_params, mdp, Q_params, k, value_losses):
        if (p_params.reshape(-1).shape[0] != n_states * n_actions):
            print('state-agg')
            origin_p_params = p_params[:]

            p_params = get_policy_params(origin_p_params)
            policy = get_policy(p_params, n_states, n_actions, temperature)

            pi_log_grad = jacrev(log_softmax)(origin_p_params.reshape(-1, n_actions), temperature)
            pi_log_grad = get_pi_log_grad(pi_log_grad, n_states, n_actions)
        else:
            policy = get_policy(p_params, n_states, n_actions, temperature)
            pi_log_grad = jacrev(get_log_policy)(p_params, n_states, n_actions, temperature).squeeze().reshape((n_states,n_actions, n_states * n_actions))

        # print(m_params)
        m_params = get_m_params(m_params, env, args.env_name, n_states, n_actions, agg_vector=agg_vector)
        # print(m_params)
        model_p, model_r, discount = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)
        model_ppi = np.einsum('ast,sa->st', model_p, policy)
        minv = np.linalg.inv(np.eye(n_states) - discount * model_ppi)

        p, r, discount = env

        true_ppi = np.einsum('ast,sa->st', p, policy)
        tinv = np.linalg.inv(np.eye(n_states) - discount * true_ppi)

        # minv = model_p[0]
        # tinv = p[0]
        true_v, true_q = policy_evaluation(env, policy)
        true_q = get_q(true_q, limit_q=limit_q, agg_q_vector=agg_q_vector)
        #true_q = true_q - np.tile(true_v.reshape(-1,1), (1, n_actions))

        print(f'kl: {float(stop_gradient(kl_divergence(p, model_p)))}, q diff: {float(stop_gradient(np.linalg.norm((true_q - policy_evaluation((model_p, model_r, discount), policy)[1])**2)))}')

        model_grad_term = np.einsum('sat,sa->sat', pi_log_grad, true_q)
        model_exp_grad_term = np.einsum('sa,sat->st', policy, model_grad_term)

        # model_grad = initial_distribution @ np.linalg.solve(np.eye(n_states) - discount * model_ppi, model_exp_grad_term)
        model_grad = initial_distribution @ np.einsum('sp,pt->st', minv, model_exp_grad_term) 

        true_grad_term = np.einsum('sat,sa->sat', pi_log_grad, true_q)
        true_exp_grad_term = np.einsum('sa,sat->st', policy, true_grad_term)

        # true_grad = initial_distribution @ np.linalg.solve(np.eye(n_states) - discount * true_ppi, true_exp_grad_term)
        true_grad = initial_distribution @ np.einsum('sp,pt->st', tinv, true_exp_grad_term) 

        paml_loss = np.linalg.norm(true_grad - model_grad)**2 
        # paml_loss = - np.dot(true_grad, model_grad) / (np.linalg.norm(true_grad) * np.linalg.norm(model_grad)) #cos similarity
        # paml_loss = ((true_grad @ model_grad) / (np.linalg.norm(true_grad) * np.linalg.norm(model_grad)))
        # paml_losses.append(jax.lax.stop_gradient(paml_loss))
        value_losses.append(0.)
        # print('paml loss:', paml_loss)
        return paml_loss #+ np.sum((r - model_r)**2)

    def paml_kstep_dist_trueQ(m_params, p_params, mdp, Q_params, k, value_losses):
        policy = get_policy(p_params, n_states, n_actions, temperature)
        pi_log_grad = jacrev(get_log_policy)(p_params, n_states, n_actions, temperature).squeeze() 

        model_p, model_r, discount = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)
        model_ppi = np.einsum('ast,sa->st', model_p, policy)
        # minv = np.linalg.inv(np.eye(n_states) - discount * model_ppi)
        minv_estimate_kstep = np.sum(np.array([np.linalg.matrix_power(discount * model_ppi, ell) for ell in range(k)]),0)

        p, r, discount = env

        true_ppi = np.einsum('ast,sa->st', p, policy)
        tinv = np.linalg.inv(np.eye(n_states) - discount * true_ppi)
        tinv_estimate_kstep = np.sum(np.array([np.linalg.matrix_power(discount * true_ppi, ell) for ell in range(k)]), 0)

        _, true_q = policy_evaluation(env, policy)    

        model_grad_term = np.einsum('sat,sa->sat', pi_log_grad, true_q)
        model_exp_grad_term = np.einsum('sa,sat->st', policy, model_grad_term)

        model_grad = initial_distribution @ np.einsum('sp,pt->st', minv_estimate_kstep, model_exp_grad_term) 

        true_grad_term = np.einsum('sat,sa->sat', pi_log_grad, true_q)
        true_exp_grad_term = np.einsum('sa,sat->st', policy, true_grad_term)
        
        true_grad = initial_distribution @ np.einsum('sp,pt->st', tinv_estimate_kstep, true_exp_grad_term) 

        paml_loss = np.linalg.norm(true_grad - model_grad)**2 
        value_losses.append(0.)
        return paml_loss + np.sum((r - model_r)**2)
    
    def paml_kstep_dist_approxQ(m_params, p_params, mdp, Q_params, k, value_losses):
        policy = get_policy(p_params, n_states, n_actions, temperature)
        pi_log_grad = jacrev(get_log_policy)(p_params, n_states, n_actions, temperature).squeeze() 

        model_p, model_r, discount = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)
        model_ppi = np.einsum('ast,sa->st', model_p, policy)
        # minv = np.linalg.inv(np.eye(n_states) - discount * model_ppi)
        minv_estimate_kstep = np.sum(np.array([np.linalg.matrix_power(discount * model_ppi, ell) for ell in range(k)]),0)

        p, r, discount = env

        true_ppi = np.einsum('ast,sa->st', p, policy)
        tinv = np.linalg.inv(np.eye(n_states) - discount * true_ppi)
        tinv_estimate_kstep = np.sum(np.array([np.linalg.matrix_power(discount * true_ppi, ell) for ell in range(k)]), 0)

        _, true_q = policy_evaluation(env, policy)
        _, estimated_true_q = iterative_policy_evaluation(env, policy, iters_PE)

        model_grad_term = np.einsum('sat,sa->sat', pi_log_grad, estimated_true_q)
        model_exp_grad_term = np.einsum('sa,sat->st', policy, model_grad_term)

        model_grad = initial_distribution @ np.einsum('sp,pt->st', minv_estimate_kstep, model_exp_grad_term) 

        true_grad_term = np.einsum('sat,sa->sat', pi_log_grad, estimated_true_q)
        true_exp_grad_term = np.einsum('sa,sat->st', policy, true_grad_term)
        
        true_grad = initial_distribution @ np.einsum('sp,pt->st', tinv_estimate_kstep, true_exp_grad_term) 

        paml_loss = np.linalg.norm(true_grad - model_grad)**2 

        value_losses.append(float(jax.lax.stop_gradient(np.sum(np.einsum('sa,sa->s', policy, (estimated_true_q - true_q)**2)))))
        return paml_loss + np.sum((r - model_r)**2)

    def paml_infstep_dist_approxQ(m_params, p_params, mdp, Q_params, k, value_losses):
        policy = get_policy(p_params, n_states, n_actions, temperature)
        pi_log_grad = jacrev(get_log_policy)(p_params, n_states, n_actions, temperature).squeeze() 

        model_p, model_r, discount = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)
        model_ppi = np.einsum('ast,sa->st', model_p, policy)
        minv = np.linalg.inv(np.eye(n_states) - discount * model_ppi)

        p, r, discount = env

        true_ppi = np.einsum('ast,sa->st', p, policy)
        tinv = np.linalg.inv(np.eye(n_states) - discount * true_ppi)
        
        _, true_q = policy_evaluation(env, policy)
        _, estimated_true_q = iterative_policy_evaluation(env, policy, iters_PE)

        # model_q = np.einsum('sap,kp->sa', jax.jacrev(log_softmax)(p_params.reshape(n_states, n_actions), temperature).reshape(n_states, n_actions, n_actions*n_states), Q_params) 

        model_grad_term = np.einsum('sat,sa->sat', pi_log_grad, estimated_true_q)
        model_exp_grad_term = np.einsum('sa,sat->st', policy, model_grad_term)

        model_grad = initial_distribution @ np.einsum('sp,pt->st', minv, model_exp_grad_term) 

        true_grad_term = np.einsum('sat,sa->sat', pi_log_grad, estimated_true_q)
        true_exp_grad_term = np.einsum('sa,sat->st', policy, true_grad_term)
        
        true_grad = initial_distribution @ np.einsum('sp,pt->st', tinv, true_exp_grad_term) 

        paml_loss = np.linalg.norm(true_grad - model_grad)**2 
        value_loss = np.sum(np.einsum('sa,sa->s', policy, (estimated_true_q - true_q)**2))
        
        # paml_losses.append(jax.lax.stop_gradient(paml_loss))
        value_losses.append(float(jax.lax.stop_gradient(value_loss)))
        return paml_loss + np.sum((r - model_r)**2)

    def paml_infstep_dist_Qmodel(m_params, p_params, mdp, Q_params, k, value_losses):
        policy = get_policy(p_params, n_states, n_actions, temperature)
        pi_log_grad = jacrev(get_log_policy)(p_params, n_states, n_actions, temperature).squeeze() 

        model_p, model_r, discount = model(m_params, n_states, n_actions, temperature, limit_model=limit_model)
        model_ppi = np.einsum('ast,sa->st', model_p, policy)
        minv = np.linalg.inv(np.eye(n_states) - discount * model_ppi)

        p, r, discount = env

        true_ppi = np.einsum('ast,sa->st', p, policy)
        tinv = np.linalg.inv(np.eye(n_states) - discount * true_ppi)
        
        _, true_q = policy_evaluation(env, policy)
        # estimated_true_q = 
        # model_q = np.einsum('sap,kp->sa', jax.jacrev(log_softmax)(p_params.reshape(n_states, n_actions), temperature).reshape(n_states, n_actions, n_actions*n_states), Q_params)
        _, model_q = policy_evaluation((model_p, model_r, discount), policy)

        model_grad_term = np.einsum('sat,sa->sat', pi_log_grad, stop_gradient(model_q))
        model_exp_grad_term = np.einsum('sa,sat->st', policy, model_grad_term)

        model_grad_modelq = initial_distribution @ np.einsum('sp,pt->st', minv, model_exp_grad_term) 

        true_grad_term = np.einsum('sat,sa->sat', pi_log_grad, stop_gradient(model_q))
        true_exp_grad_term = np.einsum('sa,sat->st', policy, true_grad_term)
        
        true_grad_modelq = initial_distribution @ np.einsum('sp,pt->st', tinv, true_exp_grad_term) 

        paml_loss = np.linalg.norm(true_grad_modelq - model_grad_modelq)**2  
        value_loss = np.sum(np.einsum('sa,sa->s', policy, (model_q - true_q)**2))
        # value_loss = np.linalg.norm(model_q - true_q)**2
        
        # paml_losses.append(jax.lax.stop_gradient(paml_loss))
        value_losses.append(float(jax.lax.stop_gradient(value_loss)))
        return paml_loss + value_loss

    losses = {
              'true-env': None,
              'paml-modelkstep-trueQ': paml_kstep_dist_trueQ,
              'paml-modelinf-trueQ': paml_infstep_dist_trueQ,
              'paml-modelinf-approxQ': paml_infstep_dist_approxQ,
              'paml-modelkstep-approxQ': paml_kstep_dist_approxQ,
              'paml-modelinf-Qmodel': paml_infstep_dist_Qmodel,
              # 'paml-vaml-model': paml_vaml_model_loss,
              # 'vaml-model': vaml_loss,
              'paml-model': paml_regular_model_loss,
              'mle-model': mle_model_objective
             }

    for i in range(total_iters):
        paml_model_losses = []
        mle_model_losses = []
        # paml_losses = []
        value_losses = []
        policy_perfs = []
        cpgs = []
        # policy_params = np.tile(true_policy_params, (n_states, 1))
        # policy_params = np.tile(true_policy_params, (1, 1)).reshape(-1)
        policy_params = get_policy_params(true_policy_params)
        # policy_params = np.concatenate((true_policy_params[0].reshape(1,-1), true_policy_params[1].reshape(1,-1), true_policy_params[1:]), axis=0).reshape(-1)
        paml_m_loss = 0
        if loss_type != 'true-env' and loss_type != 'random-env':
            for qiter in range(model_iters):
                policy_params_model_train = policy_params if loss_type == 'mle-model' else true_policy_params
                m_loss, m_grad = jax.value_and_grad(losses[loss_type])(model_params, policy_params_model_train, env, Q_params, k, value_losses)

                print(m_grad)
                if 'mle' in loss_type:
                    paml_m_loss = jax.lax.stop_gradient(paml_infstep_dist_trueQ(model_params, true_policy_params, env, Q_params, k, value_losses))
                    mle_m_loss = m_loss
                elif 'paml' in loss_type:
                    paml_m_loss = m_loss
                    mle_m_loss = jax.lax.stop_gradient(mle_model_objective(model_params, true_policy_params, env, Q_params, k, value_losses))
                else: # in case, shouldn't get here
                    paml_m_loss = jax.lax.stop_gradient(paml_infstep_dist_trueQ(model_params, true_policy_params, env, Q_params, k, value_losses))
                    mle_m_loss = jax.lax.stop_gradient(mle_model_objective(model_params, true_policy_params, env, Q_params, k, value_losses))
                
                model_opt_state = model_opt_update(0, m_grad, model_opt_state)
                model_params = model_get_params(model_opt_state)
                # model_params = np.clip(model_params, a_min=1.e-6, a_max=1.-1e-6)
                if constraint is not None:
                    model_params = constraint * model_params / np.linalg.norm(model_params)
                # print(np.linalg.norm(model_params))

                paml_model_losses.append(float(paml_m_loss))
                mle_model_losses.append(float(mle_m_loss))
                print('model_loss:', m_loss)
        
        # pi_grad = expanded_pg(policy_params, model_params, true_qf) #confirmed that this works by taking grads of policy perf with jax
        for piter in range(policy_iters):
            # policy_params = np.tile(true_policy_params, (n_states, 1))
            # policy_params = np.tile(true_policy_params, (1, 1)).reshape(-1)
            true_policy_params = true_policy_params.reshape(-1, n_actions)

            policy_params = get_policy_params(true_policy_params)

            # Q_pi_w = np.einsum('sap,kp->sa', jax.jacrev(log_softmax)(policy_params.reshape(n_states, n_actions), temperature).reshape(n_states, n_actions, n_actions*n_states), Q_params)

            # if loss_type != 'true-env':
            #     m = model(model_params, n_states, n_actions, temperature)
            # else:
            m = env

            vf, qf = policy_evaluation(m, get_policy(policy_params, n_states, n_actions, temperature))
            
            # qf = qf - np.tile(vf.reshape(-1,1), (1, n_actions))
            print('Q function:', qf)
            # Q_loss, Q_grad = jax.value_and_grad(get_Q_loss)(Q_params, qf, policy_params)

            # Q_opt_state = Q_opt_update(0, Q_grad, Q_opt_state)
            # Q_params = Q_get_params(Q_opt_state)

            # pi_grad = expanded_pg(true_policy_params, model_params, Q_pi_w, use_true=(loss_type=='true-env'))
            # print(get_q(qf, limit_q=limit_q))
            # print(model(get_m_params(model_params, env, n_states, n_actions), n_states, n_actions, temperature, limit_model=limit_model)[0])
            pi_grad = expanded_pg(true_policy_params.reshape(-1), model_params, get_q(qf, limit_q=limit_q, agg_q_vector=agg_q_vector), use_true=(loss_type=='true-env'))
            # print("MODEL PI GRAD", pi_grad)

            # print("TRUE PI GRAD", expanded_pg(true_policy_params, model_params, get_q(qf, limit_q=limit_q, agg_q_vector=agg_q_vector), use_true=True))
            
            # print(f'RANDOM MODEL: {model(model_params, n_states, n_actions, temperature)[0]}')
            # print(f'TRUE: {env[0]}')
            # compare_true_policy_params.append(jax.lax.stop_gradient(policy_params))
            # true_pi_grads.append(jax.lax.stop_gradient(pi_grad))
            # compare_Q_params.append(jax.lax.stop_gradient(Q_params))
            # policy = get_policy(policy_params, n_states, n_actions, temperature)

            # if dist_type == 'true':
            #     pi_grad = grad(policy_perf_mdp)(policy_params)
            # else:
            #     pi_grad = grad(policy_perf_model)(policy_params, model_params)#pg_calculated(policy_params, model_params)

            pi_opt_state = pi_opt_update(0, -pi_grad, pi_opt_state)
            true_policy_params = pi_get_params(pi_opt_state)
            # policy_params = np.tile(true_policy_params, (n_states, 1))
            # policy_params = np.tile(true_policy_params, (1, 1)).reshape(-1)


            # print(policy_params)
            #aggregate some states together: PARTITION
            # if 'BiasedGridworld' in args.env_name:
            #     n_states_wall = int(onp.sqrt(n_states))
            #     policy_params = policy_params.reshape((n_states_wall, n_states_wall, n_actions))
            #     new_policy_params = onp.zeros((n_states_wall, n_states_wall, n_actions))
            #     for j in range(0, n_states_wall, 2):
            #         for i in range(0, n_states_wall, 2):
            #             new_policy_params[i, j] = policy_params[i, j]
            #             new_policy_params[i + 1, j] = policy_params[i, j]
            #             new_policy_params[i, j + 1] = policy_params[i, j]
            #             new_policy_params[i + 1, j + 1] = policy_params[i, j]

            #     policy_params = np.array(new_policy_params.reshape(-1))
                # print(policy_params)
            # else:
                # n_states_wall = n_states
                    
                # half_states = n_states // 2
                # rest_states = n_states - n_states // 2

                # half_actions = n_actions // 2
                # rest_actions = n_actions - n_actions//2

                # # print(np.linalg.norm(policy_params))
                # # policy_norm_constraint = 0.5
                # # policy_params = jax.lax.stop_gradient(policy_norm_constraint * policy_params/np.linalg.norm(policy_params))

                # policy_params = policy_params.reshape(n_states, n_actions)
                # # #aggregate actions
                # # # policy_params = jax.lax.stop_gradient(
                # # #     np.hstack((
                # # #         np.repeat(policy_params[:,0].reshape(n_states, 1), half_actions, axis=1), 
                # # #         np.repeat(policy_params[:,-1].reshape(n_states, 1), rest_actions, axis=1))).reshape(-1))

                # # # print(policy_params.shape)
                # #aggregate states
                # policy_params = jax.lax.stop_gradient(
                #     np.vstack((
                #         np.repeat(policy_params[0,:].reshape(1, n_actions), half_states, axis=0), 
                #         np.repeat(policy_params[-1,:].reshape(1, n_actions), rest_states, axis=0))).reshape(-1))

            # print(policy_params)
            # if q_type != 'true':
            #     model_p, model_r, discount = model(model_params, n_states, n_actions, temperature)
            #     _, true_q = policy_evaluation(env, policy)
            #     _, model_q = policy_evaluation((model_p, model_r, discount), policy)    
            #     Q_loss = np.linalg.norm(true_q - model_q)
            # else:
            #     Q_loss = 0.

            # print(f'Iteration: {i}, Q_loss: {Q_loss:.4f}, policy perf: {policy_perf_mdp(policy_params):.4f}')
            # policy_perfs.append(float(policy_perf_mdp(policy_params)))
            print(f'Iteration: {i}, policy perf: {policy_perf_mdp(policy_params):.4f}')
            policy_perfs.append(float(policy_perf_mdp(policy_params)))
            # if loss_type != 'true-env':
            #     cpg_policy = get_policy(policy_params, n_states, n_actions, temperature)
            #     true_p, true_r, discount = env
            #     true_ppi = np.einsum('ast,sa->st', true_p, cpg_policy)
            #     tinv = np.linalg.inv(np.eye(n_states) - discount * true_ppi) * (1. - discount)
            #     # if (dist_type != 'random'):
            #     print(model_params.shape)
            #     model_ = model(model_params, n_states, n_actions, temperature, limit_model=limit_model)
            #     _ppi = np.einsum('ast,sa->st', model_[0], cpg_policy)
            #     _inv = np.linalg.inv(np.eye(n_states) - discount * _ppi) * (1. - discount)
            #     cpg = np.mean(tinv / _inv)
            # else:
            #     cpg = 1.0
            # cpgs.append(float(cpg))

        data_dict = {
                 'Iter': i,
                 'mle_model_loss': mle_model_losses,
                 'paml_model_loss': paml_model_losses,
                 'value_loss': value_losses,
                 'policy_perf_mdp': policy_perfs,
                 'cpg': cpgs
            }
        csv_logger.writerow(data_dict)

    if loss_type != 'true-env':
        # models[loss_type] = model(model_params, n_states, n_actions, temperature)
        # save model
        np.save(f'{args.save_sub_dir}model_params.npy', model_params)
        np.save(f'{args.save_sub_dir}agg_vector.npy', agg_vector)
    #save policy params
    np.save(f'{args.save_sub_dir}policy_params.npy', policy_params)

    #save Q params
    np.save(f'{args.save_sub_dir}Q_params.npy', Q_params)


    # axs_perfs[0].plot(range(len(policy_perfs)), policy_perfs, label=f'{loss_type}')#label=f'{q_type} q, {loss_type} distribution')
    # axs_perfs[1].plot(range(len(cpgs)), cpgs, label=f'{loss_type}')

    # axs_perfs[0].legend()
    # axs_perfs[0].set_xlabel('Policy Updates')
    # axs_perfs[0].set_ylabel('True policy performance')
    # axs_perfs[1].legend()
    # axs_perfs[1].set_xlabel('Policy Updates')
    # axs_perfs[1].set_ylabel('Concentrability coeff')
    # fig_perfs.savefig(f'compatible_q_dist_perfs_{n_states}states_{regularize_factor}model_param_norm.pdf', bbox_inches='tight')


def graph_stationary_state(seed, argss, env, limit_model=False):
    inverses = {}
    pi_log_grads = {}
    H_thetas = {}
    policies = {}

    n_states = env[0].shape[-1]
    n_actions = env[0].shape[0]

    fid = f'limit_q_{limit_q}'
    
    for args in argss:
        policy_params = np.load(f'{args.save_sub_dir}policy_params.npy')
        if args.model_type != 'true-env':
            model_params = np.load(f'{args.save_sub_dir}model_params.npy')
            # print(model_params)
            try:
                agg_vector = np.load(f'{args.save_sub_dir}agg_vector.npy')
            except:
                agg_vector = None
            print(agg_vector)
            
            model_params = get_m_params(model_params, env, args.env_name, n_states, n_actions, agg_vector=agg_vector)
            _p, _r, _discount = model(model_params, n_states, n_actions, args.temperature, limit_model=limit_model)
            print(_p)
            # print(args.model_type, _p[0,0])
        else:
            _p, _r, _discount = env

        Q_params = np.load(f'{args.save_sub_dir}Q_params.npy')
        qf = np.einsum('sap,kp->sa', jax.jacrev(log_softmax)(policy_params.reshape(n_states, n_actions), args.temperature).reshape(n_states, n_actions, n_actions*n_states), Q_params)

        # print(policy_params)
        policy = get_policy(policy_params, n_states, n_actions, args.temperature)
        policies[args.model_type] = policy.reshape(n_states,n_actions)
        
        pi_log_grad = jacrev(log_softmax)(policy_params.reshape(n_states, n_actions), args.temperature).reshape(n_states, n_actions, n_states * n_actions)
        pi_log_grads[args.model_type] = pi_log_grad

        H_theta = np.multiply(np.expand_dims(policy, axis=2), np.multiply(pi_log_grad, np.expand_dims(qf, axis=2))).sum(axis=1)
        H_thetas[args.model_type] = H_theta

        k = args.model_ksteps
        ppi = np.einsum('ast,sa->st', _p, policy)
        if 'kstep' in args.model_type:
            inv = np.sum(np.array([np.linalg.matrix_power(_discount * ppi, ell) for ell in range(k)]),0)
        else:
            # inv = np.linalg.inv(np.eye(n_states) - _discount * ppi) * (1. - _discount)
            inv = _p[1, :, :]
        inverses[args.model_type] = inv

        constraint = args.constraint

    xlist = np.linspace(0.0, n_states - 1, n_states)
    ylist = np.linspace(0.0, n_states - 1, n_states)
    X,Y = np.meshgrid(xlist,ylist)


    ########################################################################################

    fig_dist, axs = plt.subplots(nrows=1, ncols=len(argss), figsize=(len(argss)*4, 4))
    # fig_dist.suptitle('Stationary state distribution with final policy')
    fig_dist.suptitle('Transitions')

    i = 0
    for key, value in inverses.items():
        # cp = axs[i].contourf(X, Y, value.T)
        cp = axs[i].imshow(value.T)
        axs[i].set_xlabel('X')
        axs[i].set_ylabel('X prime')
        fig_dist.colorbar(cp, ax=axs[i])
        axs[i].set_title(key)
        axs[i].xaxis.set_major_locator(MultipleLocator(1))
        axs[i].yaxis.set_major_locator(MultipleLocator(1))
        i += 1

    fig_dist.tight_layout()
    fig_dist.savefig(f'images/{args.env_name}_compatible_q_dist_stationarydist_finalpolicy_{n_states}states_{constraint}model_param_norm_seed_{seed}_{fid}.pdf', bbox_inches='tight')


    ########################################################################################

    fig_dist, axs = plt.subplots(nrows=1, ncols=len(argss), figsize=(len(argss)*4, 4))
    fig_dist.suptitle('final policy')

    i = 0
    for key, value in policies.items():
        cp = axs[i].imshow(value.T)
        axs[i].set_xlabel('State')
        axs[i].set_ylabel('Action')
        fig_dist.colorbar(cp, ax=axs[i])
        axs[i].set_title(key)
        axs[i].xaxis.set_major_locator(MultipleLocator(1))
        axs[i].yaxis.set_major_locator(MultipleLocator(1))
        i += 1

    fig_dist.tight_layout()
    fig_dist.savefig(f'images/{args.env_name}_compatible_q_finalpolicy_{n_states}states_{constraint}model_param_norm_seed_{seed}_{fid}.pdf', bbox_inches='tight')


    ########################################################################################

    fig_dist, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    # true_P = env[0]  
    true_R = env[1].reshape((n_states, n_actions)) #SxA
    # true_R = np.einsum('sa,sa->s', true_R, policies['true-env'])

    cp = axs.imshow(true_R.T)
    axs.set_xlabel('State')
    axs.set_ylabel('Action')
    fig_dist.colorbar(cp, ax=axs)
    axs.set_title('True Reward')
    axs.xaxis.set_major_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(1))

    fig_dist.tight_layout()
    fig_dist.savefig(f'images/{args.env_name}_true_rewards_seed_{seed}.pdf', bbox_inches='tight')

    
    ########################################################################################

    fig_dist, axs = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))

    # true_P = env[0]
    vf, qf = policy_evaluation((_p, _r, _discount), get_policy(policy_params, n_states, n_actions, 1.0))  
    # true_q = qf - np.tile(vf.reshape(-1,1), (1, n_actions))#env[1].reshape((n_states, n_actions)) 
    #SxA
    
    cp = axs.imshow(qf.T)
    axs.set_xlabel('State')
    axs.set_ylabel('Action')
    fig_dist.colorbar(cp, ax=axs)
    axs.set_title('True Q function of final policy')
    axs.xaxis.set_major_locator(MultipleLocator(1))
    axs.yaxis.set_major_locator(MultipleLocator(1))

    fig_dist.tight_layout()
    fig_dist.savefig(f'images/{args.env_name}_true_q_seed_{seed}.pdf', bbox_inches='tight')

    # fig_dist, axs = plt.subplots(nrows=3, ncols=len(argss), figsize=(len(argss)*4, 4))
    # fig_dist.suptitle('final policy log grad')

    # i = 0
    # for key, value in pi_log_grads.items():
    #     for j in range(value.shape[1]):    # cp = axs[i].contourf(X, Y, value.T)
    #         cp = axs[j, i].imshow(value[:,j,:])
    #         axs[j, i].set_xlabel('Policy Param')
    #         axs[j, i].set_ylabel('State')
    #         fig_dist.colorbar(cp, ax=axs[j, i])
    #         axs[j, i].set_title(key)
    #     i += 1

    # fig_dist.tight_layout()
    # fig_dist.savefig(f'images/{args.env_name}_compatible_q_loggrad_finalpolicy_{n_states}states_{regularize_factor}model_param_norm_{fid}.pdf', bbox_inches='tight')

    # fig_dist, axs = plt.subplots(nrows=1, ncols=len(argss), figsize=(len(argss)*4, 4))
    # fig_dist.suptitle('final policy H-theta')

    # i = 0
    # for key, value in H_thetas.items():
    #     # cp = axs[i].contourf(X, Y, value.T)
    #     cp = axs[i].imshow(value)
    #     axs[i].set_xlabel('Policy Param')
    #     axs[i].set_ylabel('State')
    #     fig_dist.colorbar(cp, ax=axs[i])
    #     axs[i].set_title(key)
    #     i += 1

    # fig_dist.tight_layout()
    # fig_dist.savefig(f'images/{args.env_name}_compatible_q_Htheta_finalpolicy_{n_states}states_{regularize_factor}model_param_norm_{fid}.pdf', bbox_inches='tight')



if __name__ == "__main__":
    # seed = 0
    # seeds = [0, 1234567, 1, 2, 41]    

    seeds = [0]
    for seed in seeds:
        onp.random.seed(seed)

        limit_model = False
        deploy_argss = deploy_loss_types_Nota(seed)
        # env = amfexample(seed, deploy_argss[0].n_states)
        env = raexample()

        # env = Garnet(seed, 10, 3, (4,5)) #this needs to be entered by hand for now for this env
        # print(env[0][0,0])
        # print(deploy_argss[0].agg_vector)
        # print(deploy_argss[1].agg_vector)
        for deploy_args in deploy_argss:
            mbrl_loop_exact(deploy_args, env)
        graph_loss_types(seed, deploy_argss)
        # try:
        graph_stationary_state(seed, deploy_argss, env, limit_model=limit_model)
        # except:
        #     print("Stationary didn't plot")
        #     continue

    # # For testing iterative policy evaluation
    # # n_states = env[0].shape[-1]
    # # n_actions = env[0].shape[0]
    # # key = jax.random.PRNGKey(0)
    # # key,subkey = jax.random.split(key)
    # # samples_from_unif = jax.random.uniform(subkey, shape=(1, n_states * n_actions))
    # # policy_params = np.ones((n_states * n_actions)) * samples_from_unif
    # # policy = get_policy(policy_params, n_states, n_actions, 1)
    # # v_i, q_i = iterative_policy_evaluation(env, policy, 100)
    # # v, q = policy_evaluation(env, policy)

    # # print(v_i, q_i)
    # # print(v, q)