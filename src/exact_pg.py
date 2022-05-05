"""Run experiments on training a model to make an policy generalize."""
import jax  # TODO: Do I want this import?
import jax.numpy as np
from jax import grad, jit, jacfwd, jacrev
from jax.scipy.special import logsumexp
import numpy as onp  # For numpy functions not yet implemented in jax
import argparse  # For args
import copy  # For copying new args
import matplotlib.pyplot as plt  # For graphing
import matplotlib as mpl  # For changing matplotlib arguments
import csv  # For logging the data
import os  # Manipulating folders for storing/loading data
import time  # For recording runtime

from src.utils import *
# from utils import *
# from plotting import *
# from deploy import *

models=[]

def lqr():
    """ Figure 2 d) of
    ''The Value Function Polytope in Reinforcement Learning''
    by Dadashi et al. (2019) https://arxiv.org/abs/1901.11524
    Returns:
        tuple (P, R, gamma) where the first element is a tensor of shape
        (A x S x S), the second element 'R' has shape (S x A) and the
        last element is the scalar (float) discount factor.
    """
    # real model
    # 4.811091
    # [-0.45       -0.1         0.5         0.5         0.42364877 -0.42364895
    #  -0.69314706  0.693147    2.2975073  -2.2975154   2.2975073  -2.2975154 ]
    # TODO: Randomize over P and R, and perhaps different dimension
    p = np.array([[[0.7, 0.3], [0.2, 0.8]],
                  [[0.99, 0.01], [0.99, 0.01]]])
    r = np.array(([[-0.45, -0.1],
                   [0.5, 0.5]]))
    discount_factor = 0.9
    # TODO: Can I get a closed form on the optimal policy?
    return p, r, discount_factor


def dadashi_deterministic():
    """ Figure 2 d) of
    ''The Value Function Polytope in Reinforcement Learning''
    by Dadashi et al. (2019) https://arxiv.org/abs/1901.11524
    Returns:
        tuple (P, R, gamma) where the first element is a tensor of shape
        (A x S x S), the second element 'R' has shape (S x A) and the
        last element is the scalar (float) discount factor.
    """
    # real model
    # 4.811091
    # [-0.45       -0.1         0.5         0.5         0.42364877 -0.42364895
    #  -0.69314706  0.693147    2.2975073  -2.2975154   2.2975073  -2.2975154 ]
    # TODO: Randomize over P and R, and perhaps different dimension
    p = np.array([[[1.-1e-6, 1e-6], [1e-6, 1.-1e-6]],
                  [[1.-1e-6, 1e-6], [1.-1e-6, 1e-6]]])
    r = np.array(([[-0.45, -0.1],
                   [0.5, 0.5]]))
    discount_factor = 0.9
    # TODO: Can I get a closed form on the optimal policy?
    return p, r, discount_factor


def mdp_3states():
    """ Figure 2 d) of
    ''The Value Function Polytope in Reinforcement Learning''
    by Dadashi et al. (2019) https://arxiv.org/abs/1901.11524
    Returns:
        tuple (P, R, gamma) where the first element is a tensor of shape
        (A x S x S), the second element 'R' has shape (S x A) and the
        last element is the scalar (float) discount factor.
    """
    #real model params norm:
    # ~10.5
    # TODO: Randomize over P and R, and perhaps different dimension
    p = np.array([[[0.6, 0.4 - 1e-6, 1e-6], [0.1, 0.8, 0.1], [0.9 - 1e-6, 1e-6, 0.1]],
                  [[0.98, 0.01, 0.01], [0.2, 1e-6, 0.8 - 1e-6], [1e-6, 0.3, 0.7 - 1e-6]]
                  ])
    r = np.array(([[0.1, -0.15],
                   [0.1, 0.8],
                   [-0.2, -0.1]
                   ]))
    discount_factor = 0.9
    return p, r, discount_factor


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
                 
    R_0 = np.array([[-1., -1., -1.]])
    R_last = np.array([[1., 1., 1.]])
    # R_zeros = np.zeros((n_states - 2, 3))
    R_zeros = np.ones((n_states - 2, 3)) * -1.
    R = np.vstack((R_0, R_zeros, R_last))
    # R = np.vstack((R_0, R_zeros, R_last))
    return P, R, 0.9


def get_policy(p_params, n_states, n_actions, temp):
    """

    :param p_params:
    :return:
    """
    return softmax(p_params.reshape(n_states, n_actions), temp)

def model(m_params, n_states, n_actions, temp, limit_model=False):
    """

    :param m_params:
    limit_model: used for amfexample (may be extended in the future) to limit which parts of the model are learnable. This is unrelated to state aggregation
    :return:
    """

    if temp is None:
        temp = 0.0001
    else:
        temp = 0.1
    temp = 1.0
    # print(m_params[-n_actions*n_states:])
    # print(m_params.shape)
    # print(m_params[:-n_actions*n_states].shape)
    if limit_model:
        p = limit_model_softmax(m_params[:-n_actions*n_states].reshape(n_actions, n_states, n_states), temp)
    else:
        p = model_softmax(m_params[:-n_actions*n_states].reshape(n_actions, n_states, n_states), temp)

    # print(m_params[-n_actions*n_states:])
    r = m_params[-n_actions*n_states:].reshape(n_states, n_actions)
    # first 4 dimensions are for reward function
    discount_factor = 0.9
    return p, r, discount_factor

# def model(m_params, n_states, n_actions, temp):
#     """
#
#     :param m_params:
#     :return:
#     """
#     # first n_actions * n_states dimensions are for reward function
#     r = m_params[:n_actions * n_states].reshape(n_states, n_actions)
#
#     #last parameter is for model
#     pstar_params_ = m_params[n_actions*n_states:]
#     pstar_params = np.array([pstar_params_[0],
#                              (np.argmax(pstar_params_[1:]) - 1)*(-1)*np.amax(pstar_params_[1:])+ 1e-1*np.argmax(pstar_params_[1:]),
#                              np.argmax(pstar_params_[1:])*np.amax(pstar_params_[1:])
#                                                                          - 1e-1*(np.argmax(pstar_params_[1:]) - 1)
#                              #if zero, put -100 because it will make softmax much smaller, if one put actual value (0,1 refer to maximizing
#                              # indices in the last two indices of pstar_params)
#                              ])
#     first_row = pstar_params/np.sum(pstar_params)
#     p = np.array([[first_row,
#                   [0, 1, 0],
#                   [0, 0, 1]
#                   ]])
#     # p = model_softmax(m_params[n_actions*n_states:].reshape(n_actions, n_states, n_states), temp)
#     discount_factor = 0.9
#     return p, r, discount_factor

def iterative_policy_evaluation(mdp, policy, num_iters):
    p, r, discount = mdp
    n_states = p.shape[-1]
    n_actions = p.shape[0]

    v_i = np.zeros(n_states)
    # q_i = np.zeros((n_states, n_actions))
    for i in range(num_iters):
        next_state_term = r + np.einsum('ast,t->as', p, discount * v_i).transpose()
        v_i = np.einsum('sa,sa->s', policy, next_state_term)
        q_i = next_state_term
    
    return v_i, q_i

def policy_evaluation(mdp, policy):
    """Policy Evaluation Solver
    We denote by 'A' the number of actions, 'S' for the number of
    states.

    :param mdp:
    :param policy: tensor of shape (S x A)
    :return: tuple (vf, qf) where the first element is vector of length S and the second element contains
      the Q functions as matrix of shape (S x A).
    """
    p, r, discount = mdp
    n_states = p.shape[-1]
    ppi = np.einsum('ast,sa->st', p, policy)
    # question, why not put ass -> ss, why differentiate between the two ss's?
    rpi = np.einsum('sa,sa->s', r, policy)
    vf = np.linalg.solve(np.eye(n_states) - discount * ppi, rpi)
    qf = r + discount * np.einsum('ast,t->sa', p, vf)
    return vf, qf

def policy_performance(mdp, policy, initial_distribution):
    """Expected discounted return from an initial distribution over states.

    :param mdp:
    :param policy: (S,) array
    :return: Scalar performance
    """
    vf, _ = policy_evaluation(mdp, policy)
    return initial_distribution @ vf


# #######################################______________PROBLEMS______________########################################
# TODO: Make some "toy" problem class that returns objectives, etc.

# #######################################______________EXPERIMENT______________########################################
def experiment(args):
    """Run an experiment according to the provided arguments.

    :param args: Arguments for the experiment.
    :return: None (or perhaps the final policy performance?)
    """
    # __________________________________________ SETUP THE PROBLEM ___________________________________________________
    random_key = jax.random.PRNGKey(args.seed)

    # fig, axs = plt.subplots(2,2)
    paml_model_objectives = []
    labels = []

    if args.toy_size == 2:
        real_mdp = lqr()#dadashi_deterministic()
    elif args.toy_size == 3:
        real_mdp = mdp_3states()#lqr()#mdp_2modes() #
    else:
        real_mdp = chain(args.toy_size)

    n_actions, n_states = real_mdp[0].shape[:2]
    e_0 = np.array([1,0,0])
    initial_distribution = np.ones(n_states) / n_states


    @jit
    def model_policy_perf(p_params, m_params):
        """

        :param p_params:
        :param m_params:
        :return:
        """
        return policy_performance(model(m_params, n_states, n_actions, args.temperature),
                                  get_policy(p_params, n_states, n_actions, args.temperature), initial_distribution)

    @jit
    def real_policy_perf(p_params, m_params):
        """The models objective function.

        :param p_params: The policy parameters.
        :param m_params: The model parameters. Unused here, but need this arg for grad to work properly.
        :return: The models objective value.
        """
        return policy_performance(real_mdp, get_policy(p_params, n_states, n_actions, args.temperature), initial_distribution)

    @jit
    def mle_model_objective(mdp, model_params):
        Phat, rhat, _ = model(model_params, n_states, n_actions, args.temperature)
        p, r, _ = mdp
        return -(kl_divergence(p, Phat) + np.sum((r - rhat)**2)) #+ args.regularize_factor*grad_norm(model_params)) #negative because we'll be maximizing it

    def paml_model_objective(policy_params, model_params):
        real_pg = grad(real_policy_perf)(policy_params, model_params)
        model_pg = grad(model_policy_perf)(policy_params, model_params)
        return -grad_norm(real_pg - model_pg) #+ args.regularize_factor*grad_norm(model_params)) #negative because we'll be maximizing it

    # @jit
    def optimize_policy(p_params, m_params, num_iters):
        """

        :param p_params:
        :param m_params:
        :param num_iters:
        :return:
        """
        if m_params is None: 
            mdp = real_mdp
        else:
            mdp = model(m_params, n_states, n_actions, None)
            models.append(mdp[0])
        losses = [-float(jax.lax.stop_gradient(paml_model_objective(p_params,m_params)))]
        p_objective, p_grad_norm = None, None
        for i in range(num_iters):
            # TODO: Generalize to other optimizers?
            if m_params is None:
                p_grad = grad(real_policy_perf)(p_params, m_params)    
            else:
                p_grad = grad(model_policy_perf)(p_params, m_params)

            p_params += args.lr_policy * p_grad
            p_objective = policy_performance(mdp, get_policy(p_params, n_states, n_actions, args.temperature), initial_distribution)
            # TODO: Doing another forward pass here wastes a decent amount of compute: we don't need it?
            # print(i, p_objective)
            p_grad_norm = grad_norm(p_grad)
            # losses.append(-float(jax.lax.stop_gradient(paml_model_objective(p_params,m_params))))
            # print(losses[-1])
            # print(f'(Policy) iter: {i:2d}, L(policy, learn_model) = {p_objective:.5f}, grad_norm: {p_grad_norm:.5f}')
        losses.append(-float(jax.lax.stop_gradient(paml_model_objective(p_params,m_params))))
        return p_params, p_objective, p_grad_norm, losses


    # __________________________________________ INITIALIZE OBJECTS FOR TRAINING _____________________________________
    # Setup the logger for storing data
    data_dict = {'epoch': 0, 'miter': 0, 'run_time': 0,
                 'policy_iteration': 0, 'model_iteration': 0,
                 'policy_objective_val': 0, 'model_objective_val': 0,
                 'policy_grad_norm': 0, 'model_grad_norm': 0,
                 'exact_grad_cos_diff': 0, 'exact_grad_l2_diff': 0,
                 'hessian_eigs': [],
                 'grad_projection': [],
                 # 'full_hessian': [],
                 'true_policy_objective_val': 0,
                 'true_policy_grad_norm': 0,
                 'Lpaml': 0,
                 'Lpaml_policy_iters': [],
                 'Lmle': 0,
                 'regularizer_norm': 0
                 }
    fieldnames = [key for key, _ in data_dict.items()]
    csv_logger = load_logger(args, fieldnames)


    init_key = jax.random.PRNGKey(args.seed)
    # model_params = jax.random.normal(init_key, shape=(n_states * n_states * n_actions + n_actions * n_states,))
    model_params = np.zeros((n_states * n_states * n_actions + n_actions * n_states))

    # model_params = jax.random.normal(init_key, shape=(n_states + n_actions * n_states,))
    # model_params = np.zeros((n_states + n_actions * n_states))

    #policy_params = jax.random.normal(init_key, shape=(n_states * n_actions,))
    policy_params = np.zeros((n_states * n_actions))

    # Train an agent to optimality on the true environment to see how far off the agent trained on the model is.
    compute_true_optimal_policy = False
    if compute_true_optimal_policy:
        num_true_iters = (args.epochs * args.iter_policy + args.iter_policy_warmup) * 100
        true_optimal_policy_params, true_policy_objective_val, true_policy_grad_norm = optimize_policy(
            policy_params, None, num_true_iters)
    else:
        true_optimal_policy_params, true_policy_objective_val, true_policy_grad_norm = None, 0, 0

    identity = np.eye(policy_params.shape[0])  # Store a copy out here, since its used in multiple places.
    model_grad_norm = None
    init_time = time.time()
    # policy_params, _, _ = optimize_policy(policy_params, model_params, num_iters=args.iter_policy_warmup)
    # policy_iteration, model_iteration = args.iter_policy_warmup, 0
    policy_iteration, model_iteration = 0, 0
    hessian_eigs = [0]
    paml_losses = []
    # __________________________________________ TRAINING  __________________________________________________________
    for epoch in range(args.epochs):
        if args.verbose: print(f"policy: {policy_params}")
        if args.verbose: print(f"model: {model_params}")

        # ______________________________ UPDATE POLICY ON THE MODEL _________________________________________________
        paml_losses = []
        policy_params, policy_objective_val, policy_grad_norm, paml_losses = optimize_policy(policy_params, model_params, args.iter_policy)
        policy_iteration += args.iter_policy

        for miter in range(args.iter_model):
            # _____________________________ UPDATE MODEL ON REAL DATA & POLICY ____________________________________________
            def get_model_grad():
                if args.model_loss == 'MLE':
                    return grad(mle_model_objective, 1)(real_mdp, model_params)
                elif args.model_loss == 'PAML':
                    return grad(paml_model_objective, 1)(policy_params, model_params)

                #Ignore below for now
                dLmodel_dpolicy = grad(real_policy_perf, 0)(policy_params, model_params)  # TODO: 0 magic number?
                dLagent_dpolicy = 0
                if args.model_loss == 'PAML':
                    dLagent_dpolicy = grad(model_policy_perf, 0)(policy_params, model_params)
                if args.verbose: print(f"    dLmodel_dpolicy: {dLmodel_dpolicy}")

                # _____________________________ APPROXIMATE VECTOR-INVERSE-HESSIAN PRODUCT____________________________________
                vector_inverse_hessian = None
                if args.inverse_approx in ['hessian', 'fisher']:
                    if args.inverse_approx == 'hessian':
                        # TODO: Probably wont work without dampening if init all zeros.
                        hessian_train_policy = hessian(model_policy_perf)(policy_params, model_params)
                        hessian_eigs[0] = np.linalg.eig(hessian_train_policy)[0]
                        if args.model_loss == 'PAML':
                            vector_inverse_hessian = np.linalg.solve(
                                hessian_train_policy + args.mix_factor * identity,
                                dLmodel_dpolicy - dLagent_dpolicy)
                        elif args.model_loss == 'IFT':
                            vector_inverse_hessian = np.linalg.solve(
                                hessian_train_policy + args.mix_factor * identity,
                                dLmodel_dpolicy)
                        if args.verbose:
                            print(f"    hessian_train_policy: {hessian_train_policy}")
                            print(f"    hessian_train_policy eigs: {np.linalg.eig(hessian_train_policy)[0]}")
                    elif args.inverse_approx == 'fisher':
                        # TODO: Doesn't work if init all zeros!
                        policy_grad = grad(model_policy_perf, 0)(policy_params, model_params)
                        v1 = policy_grad.reshape(1, -1) @ dLmodel_dpolicy.reshape(-1, 1)
                        vector_inverse_hessian = (policy_grad.reshape(-1, 1) @ v1.reshape(1, -1)).reshape(-1)
                elif args.inverse_approx in ['identity']:
                    if args.model_loss == 'PAML':
                        vector_inverse_hessian = dLmodel_dpolicy - dLagent_dpolicy
                    elif args.model_loss == 'IFT':
                        vector_inverse_hessian = dLmodel_dpolicy
                else:
                    print(f"Error: An inverse approximation strategy of {args.inverse_approx} is not supported.")
                    exit(0)
                if args.verbose: print(f"    vector_inverse_hessian: {vector_inverse_hessian}")

                # __________________________________ APPROXIMATE MIXED 2ND DERIVATIVES_________________________________________
                mixed_train_derivatives = mixed_hessian(model_policy_perf)(policy_params, model_params)
                if args.verbose: print(f"    mixed_train_derivatives: {mixed_train_derivatives}")
                return vector_inverse_hessian @ mixed_train_derivatives

            # __________________________________ COMBINE TERMS FOR FINAL MODEL UPDATE _____________________________________
            direct_term = 0
            model_grad = direct_term + get_model_grad()
            if args.verbose: print(f"    model_grad: {model_grad}")
            # model_grad_noise = jax.random.normal(random_key, shape=(
            #     n_states * n_states * n_actions + n_actions * n_states,)) * args.model_grad_noise_scale
            model_grad_noise = 0
            model_params += args.lr_model * (model_grad + model_grad_noise)  # TODO: Generalize to other optimizers?
            if args.pgd:
                model_params = args.regularize_factor * model_params / np.sqrt(grad_norm(model_params))
            model_iteration += 1
        
            # if args.model_loss == "MLE" or args.model_loss == "PAML":
            #     model_objective_val = -mle_model_objective(real_mdp, model_params)
            # else:
            model_objective_val = real_policy_perf(policy_params, model_params)
            model_grad_norm = np.sqrt(grad_norm(model_grad))
            # print(model(model_params, n_states, n_actions, args.temperature)[0])
            # print(np.sqrt(grad_norm(model_params)))
            # print(model_params)
            

            # print(f'(Model)epoch:{epoch:2d},L(policy, true_model):{model_objective_val:.5f},grad_norm:{model_grad_norm}')
            
            if args.model_loss == 'PAML':
                if (epoch >= 0):
                    hessian_Lmodel_modelmodel = jacfwd(jacrev(paml_model_objective, 1), 1)(policy_params, model_params)
                    hessian_Lmodel_modelpolicy = jacfwd(jacrev(paml_model_objective, 1), 0)(policy_params, model_params)
                    hessian_Lpolicy_policypolicy = hessian(model_policy_perf)(policy_params, model_params)
                    hessian_Lpolicy_policymodel = jacfwd(jacrev(model_policy_perf, 0), 1)(policy_params, model_params)
                    full_hessian = np.block([[hessian_Lmodel_modelmodel,   hessian_Lmodel_modelpolicy  ],
                                             [hessian_Lpolicy_policymodel, hessian_Lpolicy_policypolicy]])

                    dLpolicy_dpolicy = jacrev(model_policy_perf)(policy_params,model_params)
                    grad_update = np.block([model_grad, dLpolicy_dpolicy])
                    #project (model_grad,policy_grad) onto full_hessian eigenspace 
                    #I think this projection is wrong, check computations
                    hess_eigvals, hess_eigvectors = np.linalg.eig(full_hessian)
                    grad_projection = np.linalg.inv(hess_eigvectors) @ grad_update

                    # if (epoch % 20 == 0):
                        
                    # axs[0].scatter(grad_projection.real, grad_projection.imag, s=10, alpha=.5)
                    # if epoch == (args.epochs - 1):
                    #     # labels.append(f'iter {epoch}')
                    #     axs[0,0].bar(range(hess_eigvectors.shape[0]), np.abs(grad_projection))
                        
                    #     # Compute areas and colors
                    #     N = hess_eigvals.shape[0]
                    #     ranking = list(np.abs(grad_projection)/np.sum(np.abs(grad_projection)))
                    #     scatter = axs[0,1].scatter(np.abs(hess_eigvals), np.angle(hess_eigvals), s=[10*(10.*i)**2 for i in ranking], alpha=0.5)
                    #     scatter = axs[1,1].scatter(np.abs(hess_eigvals), np.angle(hess_eigvals), alpha=0.5)
                    #     # axs[1].scatter(range(hess_eigvectors.shape[0]), np.abs(hess_eigvals), s=10, alpha=.5, label='magnitude')
                    #     # axs2 = axs[1].twinx()
                    #     # axs2.scatter(range(hess_eigvectors.shape[0]), np.angle(hess_eigvals), s=10, alpha=.5, color='r', label='phase')
                    #     # axs2.set_ylabel('Phase')

                paml_model_objectives.append(-paml_model_objective(policy_params,model_params))
            
            elif args.model_loss == 'IFT':
                pass
                #this section doesn't work
                # hessian_Lmodel_modelmodel = jacfwd(jacrev(real_policy_perf, 1), 1)(policy_params, model_params)
                # hessian_Lmodel_modelpolicy = jacfwd(jacrev(real_policy_perf, 1), 0)(policy_params, model_params)
                # hessian_Lpolicy_policypolicy = hessian(model_policy_perf)(policy_params, model_params)
                # hessian_Lpolicy_modelpolicy = jacfwd(jacrev(model_policy_perf, 0), 1)(policy_params, model_params)
                # print(f'hessian_Lmodel_modelmodel {hessian_Lmodel_modelmodel}')
                # print(f'hessian_Lmodel_modelpolicy {hessian_Lmodel_modelpolicy}')
                # print(f'hessian_Lpolicy_policypolicy {hessian_Lpolicy_policypolicy}')
                # print(f'hessian_Lpolicy_modelpolicy {hessian_Lpolicy_modelpolicy}')
                # full_hessian = np.block([[hessian_Lmodel_modelmodel,   hessian_Lmodel_modelpolicy  ],
                #                          [hessian_Lpolicy_modelpolicy, hessian_Lpolicy_policypolicy]])

            elif args.model_loss == 'MLE':
                if (epoch >= 0):
                    hessian_Lmodel_modelmodel = jacfwd(jacrev(mle_model_objective, 1), 1)(real_mdp, model_params)
                    hessian_Lmodel_modelpolicy = np.tile(np.zeros_like(policy_params), (model_params.shape[0],1))
                    hessian_Lpolicy_policypolicy = hessian(model_policy_perf)(policy_params, model_params)
                    hessian_Lpolicy_policymodel = jacfwd(jacrev(model_policy_perf, 0), 1)(policy_params, model_params)
                    full_hessian = np.block([[hessian_Lmodel_modelmodel,   hessian_Lmodel_modelpolicy  ],
                                             [hessian_Lpolicy_policymodel, hessian_Lpolicy_policypolicy]])

                    dLpolicy_dpolicy = jacrev(model_policy_perf)(policy_params,model_params)
                    grad_update = np.block([model_grad, dLpolicy_dpolicy])
                    #project (model_grad,policy_grad) onto full_hessian eigenspace
                    #I think this projection is wrong, check computations
                    hess_eigvals, hess_eigvectors = np.linalg.eig(full_hessian)
                    grad_projection = np.linalg.inv(hess_eigvectors) @ grad_update

                    # if (epoch % 20 == 0):
                    #     labels.append(f'iter {epoch}')
                    #     axs[0].bar(range(hess_eigvectors.shape[0]), np.abs(grad_projection))

                    # if epoch == (args.epochs - 1):
                    #     axs[0,0].bar(range(hess_eigvectors.shape[0]), np.abs(grad_projection))
                        
                    #     N = hess_eigvals.shape[0]
                    #     ranking = list(np.abs(grad_projection)/np.sum(np.abs(grad_projection)))
                    #     scatter = axs[0,1].scatter(np.abs(hess_eigvals), np.angle(hess_eigvals), s=ranking, alpha=0.5) #[10*(10.*i)**2 for i in ranking]
                    #     scatter = axs[1,1].scatter(np.abs(hess_eigvals), np.angle(hess_eigvals), alpha=0.5)
                    #     # axs[1].scatter(range(hess_eigvectors.shape[0]), np.abs(hess_eigvals), s=10, alpha=.5, label='magnitude')
                    #     # axs2 = axs[1].twinx()
                    #     # axs2.scatter(range(hess_eigvectors.shape[0]), np.angle(hess_eigvals), s=10, alpha=.5, color='r', label='phase')
                    #     # axs2.set_ylabel('Phase')
                    #     # axs2.legend()
                paml_model_objectives.append(-mle_model_objective(real_mdp, model_params))

            # print(f'(Model)epoch:{epoch:2d},L(policy, true_model):{model_objective_val:.5f},'
            #       f'grad_norm:{-paml_model_objective(policy_params, model_params)}')
            # __________________________________ STORE DATA FROM THIS EPOCH  ______________________________________________
            # TODO: compute "exact gradient" and these comparisons if we want
            if miter > 0:
                paml_losses = [] 

            data_dict = {'epoch': epoch,
                         'miter': miter,
                         'run_time': time.time() - init_time,
                         'policy_iteration': policy_iteration, 'model_iteration': model_iteration,
                         'policy_objective_val': policy_objective_val, 
                         'model_objective_val': model_objective_val,
                         'policy_grad_norm': policy_grad_norm, 'model_grad_norm': model_grad_norm,
                         'exact_grad_cos_diff': 0, 'exact_grad_l2_diff': 0,
                         'hessian_eigs': hessian_eigs,
                         # 'full_hessian': full_hessian.reshape(-1,).tolist(),
                         'true_policy_objective_val': true_policy_objective_val,
                         'true_policy_grad_norm': true_policy_grad_norm,
                         'Lpaml' : -paml_model_objective(policy_params,model_params),
                         'Lpaml_policy_iters': paml_losses,
                         'Lmle'  : -mle_model_objective(real_mdp, model_params),
                         'regularizer_norm': np.sqrt(grad_norm(model_params))
                         }
            # print(data_dict['full_hessian'])
            csv_logger.writerow(data_dict)

            # if epoch % args.save_mod == 0 and epoch >= 200: 
            # # onp.save(args.save_sub_dir + '/model_params', model_params)  # Put expensive storage operations here. 
            #     onp.save(args.save_sub_dir + f'/full_hessian_{epoch}', full_hessian)

    # ______________________________ FINAL BOOKKEEPING BEFORE EXPERIMENT FINISHED  ____________________________________
    paml_losses = [] 
    policy_params, policy_objective_val, policy_grad_norm, paml_losses = optimize_policy(policy_params, model_params, args.iter_policy_final)
    
    policy_iteration += args.iter_policy_final

    # if args.model_loss == "MLE":
    #     model_objective_val = mle_model_objective(real_mdp, model_params)
    # else:
    if args.model_loss == 'PAML':
        hessian_Lmodel_modelmodel = jacfwd(jacrev(paml_model_objective, 1), 1)(policy_params, model_params)
        hessian_Lmodel_modelpolicy = jacfwd(jacrev(paml_model_objective, 1), 0)(policy_params, model_params)
        hessian_Lpolicy_policypolicy = hessian(model_policy_perf)(policy_params, model_params)
        hessian_Lpolicy_modelpolicy = jacfwd(jacrev(model_policy_perf, 0), 1)(policy_params, model_params)
        full_hessian = np.block([[hessian_Lmodel_modelmodel,   hessian_Lmodel_modelpolicy  ],
                                 [hessian_Lpolicy_modelpolicy, hessian_Lpolicy_policypolicy]])
    elif args.model_loss == 'MLE':
        hessian_Lmodel_modelmodel = jacfwd(jacrev(mle_model_objective, 1), 1)(real_mdp, model_params)
        hessian_Lmodel_modelpolicy = np.tile(np.zeros_like(policy_params), (model_params.shape[0],1))
        hessian_Lpolicy_policypolicy = hessian(model_policy_perf)(policy_params, model_params)
        hessian_Lpolicy_modelpolicy = jacfwd(jacrev(model_policy_perf, 0), 1)(policy_params, model_params)
        full_hessian = np.block([[hessian_Lmodel_modelmodel,   hessian_Lmodel_modelpolicy  ],
                                 [hessian_Lpolicy_modelpolicy, hessian_Lpolicy_policypolicy]])
    
    model_objective_val = real_policy_perf(policy_params, model_params)

    data_dict = {'epoch': args.epochs,
                 'miter': miter,
                 'run_time': time.time() - init_time,
                 'policy_iteration': policy_iteration, 'model_iteration': model_iteration,
                 'policy_objective_val': policy_objective_val, 'model_objective_val': model_objective_val,
                 'policy_grad_norm': policy_grad_norm, 'model_grad_norm': model_grad_norm,
                 'exact_grad_cos_diff': 0, 'exact_grad_l2_diff': 0,
                 'hessian_eigs': hessian_eigs,
                 # 'full_hessian': full_hessian.reshape(-1,).tolist(),
                 'true_policy_objective_val': true_policy_objective_val,
                 'true_policy_grad_norm': true_policy_grad_norm,
                 'Lpaml': -paml_model_objective(policy_params, model_params),
                 'Lpaml_policy_iters': paml_losses,
                 'Lmle': -mle_model_objective(real_mdp, model_params),
                 'regularizer_norm': np.sqrt(grad_norm(model_params))
                 }
    csv_logger.writerow(data_dict)
    



    # onp.save(args.save_sub_dir + f'/full_hessian_{epoch}', full_hessian)





    if args.verbose:
        v_model_final, q_model_final = policy_evaluation(model(model_params, n_states, n_actions, args.temperature),
                                                         get_policy(policy_params, n_states, n_actions, args.temperature))
        v_true_final, q_true_final = policy_evaluation(real_mdp, get_policy(policy_params, n_states, n_actions,
                                                                            args.temperature))
        print(f'v_model_final: {v_model_final}, v_true_final: {v_true_final}')  # ordering of v also holds
        print(f'q_model_final: {q_model_final}, q_true_final: {q_true_final}')

    # ordering of q is the same for model and mdp even though the actual parameters are very different, holds for
    # both states marginalized over actions, and for actions and states together
    # orderings are all the same, even though proportions are different

    real_final_performance = policy_performance(real_mdp,
                                                get_policy(policy_params, n_states, n_actions, args.temperature),
                                                initial_distribution)
    if args.verbose: print(f"Final L(policy*(learn_model), true_model): {real_final_performance}")
    # Conclusion: can get really good policy, with a model that is very different than true mdp, and value/q
    # functions that are also very different but have correct ordering over states and actions
    # Also, just doing one step in the inner optimization works pretty well
    # Doing 5 works better but takes much longer
    

    # axs[1,0].plot(range(len(paml_model_objectives)), paml_model_objectives, label=f'{args.model_loss} Model loss')
    
    # # axs[1].set_ylabel('Magnitude')
    # # axs[1].set_xlabel('Eigenvector index')
    # # axs[1].legend()

    # axs[0,0].legend(labels=labels)
    # axs[0,0].set_ylabel('Grad dot Eigenvector')
    # axs[0,0].set_xlabel('Eigenvector index')
    # axs[0,0].axvline(x=24, c='black')

    # axs[0,1].set_ylabel('Phase')
    # axs[0,1].set_xlabel('Magnitude')
    # axs[0,1].set_xscale('log')
    # axs[0,1].legend()

    # axs[1,1].set_ylabel('Phase')
    # axs[1,1].set_xlabel('Magnitude')
    # axs[1,1].set_xscale('log')
    # axs[1,1].legend()

    # axs[1,0].legend()

    # fig.savefig(f'images/grad_projections_{args.model_loss}.pdf')
    return real_final_performance



# TODO: Graph real/model loss vs iteration with error bars OVER DIFFERENT SEEDS.  Do we want opt loss graphed?
# TODO: loss vs iter graphs for each inversion strategies.
# TODO: loss vs iter graphs for PAML/MLE.
# TODO: loss vs iter graphs for learning rates.
# TODO: loss vs iter graphs for different sized problems.
# TODO: loss vs iter graphs for different capacity learned model
# plt.savefig('images/temp')


# #######################################______________EXAMPLE______________########################################
if __name__ == "__main__":
    do_all_experiments = False
    do_all_graphing = False
    # deploy_argss = deploy_seeds()
    # TODO: Make more deployments

    '''if do_all_experiments:
        print("Beginning experiments...")
        for deploy_argss in deployments:
            for deploy_args in deploy_argss:
                print(f"Running {deploy_args}")
                final_objective_val = experiment(deploy_args)
                print(f"Final L(policy*(learn_model), true_model): {final_objective_val}")
                print("______________________________________________________________________________________________")
        print("Finished experiments!")

    print("Beginning graphing...")
    if do_all_graphing:
        for deploy_argss in deployments:
            for x_type_local in ['epoch']:
                for y_type_local in ['policy_objective_val', 'model_objective_val',
                                     'policy_grad_norm', 'model_grad_norm',
                                     'exact_grad_cos_diff', 'exact_grad_l2_diff']:
                    for deploy_args in deploy_argss:
                        graph_single(deploy_args, x_type_local, y_type_local)
                        graph_grad_norms(deploy_args, x_type_local)'''

    # graph_iters(deploy_iters())
    # print(model(
    # np.array([-0.45, -0.1, 0.5, 0.5, 0.42364877, -0.42364895, -0.69314706, 0.693147,2.2975073, -2.2975154, 2.2975073, -2.2975154]), 2, 2, 1))

    deploy_argss = deploy_regularizer()#deploy_losses()#
    # for deploy_args in deploy_argss:
    #     # deploy_args = deploy_argss[0]
    #     experiment(deploy_args)
        # print(models)
        # np.save('mleprog.npy', np.asarray(models))
    
    # graph_iters(deploy_argss)
    # graph_lrs(deploy_argss)
    # graph_losses_with_policy_iters(deploy_argss)

    graph_regularizers_v_rf(deploy_argss)
    # graph_seeds(deploy_seeds())
    print("Finished graphing!")

    # TODO: Don't materialize the matrices we don't need!  So we can scale.
    # TODO: Implement grid search over parameters to tune: ex., policy/model lr/iter
    # TODO: Implement baselines: ex., paml vs mle.  different inverse strategies
    # TODO: Implement deployment for comparing baselines
    # TODO: Implement graphs for deployments


