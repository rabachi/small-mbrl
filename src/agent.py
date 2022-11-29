import numpy as np
# from regex import F
import torch
import jax.numpy as jnp
from jax.scipy.linalg import lu_factor, lu_solve
import jax
from itertools import product
from typing import List, Tuple, Dict
from scipy.special import comb
import pickle
# from memory_profiler import profile
from src.utils import softmax, get_log_policy, get_policy
import gc

# CVAR_CONSTRAINT = -1.5 #RFL
# CVAR_CONSTRAINT = -4.0 #cliffwalking
# CVAR_CONSTRAINT = -10.0 #cliffwalking
# CVAR_CONSTRAINT = 30.0 #Chain

def policy_evaluation(mdp: Tuple[np.ndarray], policy: jnp.ndarray, nState, discount): 
    # Vals[state, timestep]
    r_matrix, p_matrix = mdp
    ppi = jnp.einsum('sat,sa->st', p_matrix, policy)
    rpi = jnp.einsum('sa,sa->s', r_matrix, policy)
    v_pi = jnp.linalg.solve(np.eye(nState) -
                            discount*ppi, rpi)
    return v_pi

def policy_performance(r, p, policy_params, initial_distribution, nState, nAction, discount):
    policy = get_policy(policy_params, nState, nAction)
    vf = policy_evaluation((r, p), policy, nState, discount)
    return initial_distribution @ vf

def calculate_value_and_grad(r_matrix, p_matrix, grad_pi, policy, nState, discount, initial_distribution):
    # r_matrix, p_matrix = mdp
    ppi = jnp.einsum('sat,sa->st', p_matrix, policy)
    rpi = jnp.einsum('sa,sa->s', r_matrix, policy)
    # lu = np.zeros((rpi.shape[0], nState, nState))
    # pivot = np.zeros((rpi.shape[0], nState, nState))
    P_factor = lu_factor(jnp.eye(nState) - discount*ppi)
    # for i in range(rpi.shape[0]):
    #     lu[i], pivot[i] = lu_factor(np.eye(nState) - discount*ppi[i])
    vpi = lu_solve(P_factor, rpi)
    # vpi = np.array([lu_solve((lu[i], pivot[i]), rpi[i]) for i in range(rpi.shape[0])])
    # vpi = np.array([np.linalg.solve(np.eye(nState) - discount*ppi[i], rpi[i]) for i in range(rpi.shape[0])])
    qpi = r_matrix + discount * jnp.einsum('sat,t->sa', p_matrix, vpi)
    gradpi_q = jnp.einsum('sapq,sa->spq', grad_pi, qpi)
    # j_grad = initial_distribution @ np.array([np.linalg.solve(np.eye(nState) - discount * ppi[i], gradpi_q[i]) for i in range(rpi.shape[0])])
    # y = np.array([lu_solve((lu[i], pivot[i]), gradpi_q[i]) for i in range(rpi.shape[0])])
    y = lu_solve(P_factor, gradpi_q)
    j_grad = initial_distribution @ y
    j = initial_distribution @ vpi
    return j, j_grad

# def calculate_value_and_grad(mdp, grad_pi, policy, nState, discount, initial_distribution):
    # r_matrix, p_matrix = mdp
    # ppi = jnp.einsum('nsat,sa->nst', p_matrix, policy)
    # rpi = jnp.einsum('nsa,sa->ns', r_matrix, policy)
    # # lu = np.zeros((rpi.shape[0], nState, nState))
    # # pivot = np.zeros((rpi.shape[0], nState, nState))
    # P_factor = torch.linalg.lu_factor(torch.eye(nState) - discount*ppi)
    # # for i in range(rpi.shape[0]):
    # #     lu[i], pivot[i] = lu_factor(np.eye(nState) - discount*ppi[i])
    # vpi = torch.linalg.lu_solve(P_factor, rpi)
    # # vpi = np.array([lu_solve((lu[i], pivot[i]), rpi[i]) for i in range(rpi.shape[0])])
    # # vpi = np.array([np.linalg.solve(np.eye(nState) - discount*ppi[i], rpi[i]) for i in range(rpi.shape[0])])
    # qpi = r_matrix + discount * np.einsum('nsat,nt->nsa', p_matrix, vpi)
    # gradpi_q = np.einsum('sapq,nsa->nspq', grad_pi, qpi)
    # # j_grad = initial_distribution @ np.array([np.linalg.solve(np.eye(nState) - discount * ppi[i], gradpi_q[i]) for i in range(rpi.shape[0])])
    # # y = np.array([lu_solve((lu[i], pivot[i]), gradpi_q[i]) for i in range(rpi.shape[0])])
    # y = torch.linalg.lu_solve(P_factor, gradpi_q)
    # j_grad = initial_distribution @ y
    # j = np.einsum('ns,ns->n', np.tile(initial_distribution, rpi.shape[0]).reshape(vpi.shape[0], vpi.shape[1]), vpi)
    # return np.asarray(j), np.asarray(j_grad)

class Agent:
    def __init__(self, nState, discount, initial_distribution, policy, init_lambda, lambda_lr, policy_lr) -> None:
        self.nState = nState
        self.discount = discount
        self.initial_distribution = initial_distribution
        self.policy = policy
        self.counter = 0
        self.lambda_param = init_lambda
        self.lambda_lr = lambda_lr
        self.policy_lr = policy_lr

    def value_iteration(self, r_matrix: np.ndarray, p_matrix: np.ndarray):
        epsilon = 1e-5
        V = np.zeros(self.nState)
        #V_progress = [np.zeros(self.nState)] * 3
        delta = np.infty

        i = 0
        while delta > epsilon:
            for s in range(self.nState):
                prev_v = V[s]
                V[s] = np.max(r_matrix[s] + np.einsum('at,t->a', p_matrix[s], self.discount*V))
                delta = np.abs(prev_v - V[s])
            i += 1
        pi = np.zeros((self.nState, self.nAction))
        for s in range(self.nState):
            pi[s][np.argmax(r_matrix[s] + np.einsum('at,t->a', p_matrix[s], self.discount * V))] = 1
        # self.policy.update_params(pi)
        return V, pi

    def sample_policy_evaluation(self, batch):
        alpha = 0.5 #how to set this?
        obses, _, rewards, next_obses, not_dones, _ = batch #should check terminal states when updating value
        
        V = np.zeros((self.nState, 1))
        for s, r, s_prime in zip(obses, rewards, next_obses):
            V[int(s)] = V[int(s)].copy() + alpha * (r + self.discount * V[int(s_prime)].copy() - V[int(s)].copy()) 
        return V #should I do a fitted method here?
        
    #all methods below should go into a super 'model' or agent class. 
    # @profile
    def grad_step(self, train_type, **kwargs):
        if train_type == 'VaR-sigmoid':
            return self.VaR_sigmoid_grad_step(**kwargs)
        elif train_type == 'VaR-delta':
            return self.VaR_delta_grad_step(**kwargs)
        #optimistic only
        elif train_type == 'max-opt':
            return self.optimistic_only('max', **kwargs)
        elif train_type == 'upper-cvar':
            return self.optimistic_only('upper-cvar', **kwargs)
        #optimistic + cvar constrained
        elif train_type == 'psrl-opt-cvar':
            return self.psrl_CVaR_constrained('psrl', **kwargs) 
        elif train_type == 'pg-cvar':
            return self.optimistic_CVaR_constrained('pg', **kwargs)
        elif train_type == 'max-opt-cvar':
            return self.optimistic_CVaR_constrained('max', **kwargs) 
        elif train_type == 'optimistic-psrl-opt-cvar':
            return self.optimistic_CVaR_constrained('optimistic-psrl', **kwargs) 
        elif train_type == 'upper-cvar-opt-cvar':
            return self.optimistic_CVaR_constrained('upper-cvar', **kwargs) 
        #both-max
        elif train_type == 'both-max-CVaR':
            return self.both_max_CVaR(**kwargs)
        #risk-aware
        elif train_type == 'CVaR':
            return self.CVaR_grad_step(**kwargs)
        elif train_type == 'grad-risk-eval':
            return self.grad_risk_evaluation(**kwargs)
        #regret version
        elif train_type == 'regret-CVaR':
            return self.regret_CVaR_grad_step(**kwargs)
        elif train_type == 'robust-DP':
            return self.robust_DP(**kwargs)
        elif train_type == 'k-of-N':
            return self.k_of_N(**kwargs)
        elif train_type == 'pg':
            return self.pg(**kwargs)
        elif train_type == 'pg-CE':
            return self.pg_CE(**kwargs)
        elif train_type == 'psrl':
            return self.psrl(**kwargs)
        else:
            raise NotImplementedError(f'Not implemented this type: {train_type}')

    def backtrackingls(self, alpha0, c, rho, grad_f, f, x_k):
        alpha = alpha0
        while f(x_k + alpha * grad_f) < f(x_k) + c * alpha * 0.5* np.linalg.norm(grad_f)**2:
            alpha = rho * alpha
        return alpha

    def pg(self,
            num_samples_plan,
            k_value,
            risk_threshold,
            R_j,
            P_j
        ):
        p_params = self.policy.get_params()
        p_grad = np.zeros_like(p_params)
        V_p_pi, V_p_pi_grad, var_alpha, cvar_alpha, samples_taken = self.posterior_sampling(p_params, num_samples_plan, risk_threshold, R_j=R_j, P_j=P_j)
        p_grad = np.mean(V_p_pi_grad, axis=0)
        av_vpi = np.mean(V_p_pi, axis=0)

        use_fim = False
        if not use_fim:
        #     lr = 2.
        #     lr = self.backtrackingls(lr, 0.1, 0.8, p_grad, lambda p: np.mean(self.posterior_sampling(p, num_samples_plan, risk_threshold)[0]), p_params)
        #     print(lr)
            p_params = np.clip(p_params + self.policy_lr * p_grad, 1.e-6, 1.-1.e-6)
        else:
            fim = self.calc_FIM(p_params)
            step = np.linalg.solve(fim, p_grad)
            p_params = np.clip(p_params + self.policy_lr * step, 0, None)
        
        grad_norm = np.linalg.norm(p_grad)
        self.policy.update_params(p_params)
        return av_vpi, var_alpha, cvar_alpha, grad_norm, samples_taken

    def calc_FIM(self, p_params):
        policy = get_policy(p_params, self.nState, self.nAction)
        grad_log_pi = jax.jacrev(get_log_policy)(p_params, self.nState, self.nAction, 1.0)
        gradsouter = np.einsum('sat,sap->satp', grad_log_pi, grad_log_pi)
        F_s_theta = np.einsum('sa,satp->stp', policy, gradsouter)
        F_theta = np.einsum('s,stp->tp', self.initial_distribution, F_s_theta)
        return F_theta + 0.01 * np.eye(p_params.shape[0])

    def psrl(self,
            num_samples_plan,
            k_value,
            risk_threshold,
            R_j,
            P_j
        ):
        p_params = self.policy.get_params()
        R_samp, P_samp = self.sample_mdp()
        # R_samp = np.expand_dims(R_samp, axis=0)
        # P_samp = np.expand_dims(P_samp, axis=0)
        V_p_pi_grad = 10. #placeholder
        i = 0
        while i < 200:
        #not np.isclose(np.linalg.norm(V_p_pi_grad), 0.0, atol=1e-2) and i <= 200:
            if (i + 1) % 1000 == 0:
                self.policy_lr /= 10.
            # policy = jnp.array(softmax(p_params))
            # grad_pi = jax.jacrev(softmax)(p_params)
            # _, V_p_pi_grad = calculate_value_and_grad((R_samp, P_samp), grad_pi, policy, self.nState, self.discount, self.initial_distribution)
            V_p_pi, V_p_pi_grad = jax.lax.stop_gradient(jax.value_and_grad(policy_performance, 2)(R_samp, P_samp, p_params, self.initial_distribution, self.nState, self.nAction, self.discount))
            print(f'iter: {i}, Vppi: {V_p_pi}, gradnorm: {np.linalg.norm(V_p_pi_grad)}')
            p_params = np.clip(p_params + self.policy_lr * np.mean(V_p_pi_grad, axis=0), 1e-6, 1.-1e-6)
            i += 1
        
        self.policy.update_params(p_params)
        V_p_pi, _, var_alpha, cvar_alpha, samples_taken = self.posterior_sampling(p_params, num_samples_plan, risk_threshold, R_j, P_j)
        return np.mean(V_p_pi, axis=0), var_alpha, cvar_alpha, np.linalg.norm(V_p_pi_grad), samples_taken

    def pg_CE(self,
            num_samples_plan,
            k_value,
            risk_threshold,
            R_j,
            P_j
        ):
        p_params = self.policy.get_params()
        p_grad = np.zeros_like(p_params)
        R_CE, P_CE = self.get_CE_model() 
        V_p_pi, V_p_pi_grad = jax.value_and_grad(policy_performance, 2)(R_CE, P_CE, p_params, self.initial_distribution, self.nState, self.nAction, self.discount)

        p_grad = V_p_pi_grad
        av_vpi = V_p_pi
        p_params += self.policy_lr * p_grad
        self.policy.update_params(p_params)
        grad_norm = np.linalg.norm(p_grad)

        _, _, v_alpha_quantile, cvar_alpha, samples_taken = self.posterior_sampling(p_params, num_samples_plan, risk_threshold, R_j=R_j, P_j=P_j)
        
        return av_vpi.item(), v_alpha_quantile.item(), cvar_alpha, grad_norm, samples_taken

    def VaR_delta_grad_step(self,
            num_samples_plan,
            k_value,
            risk_threshold
        ):
        p_params = self.policy.get_params()
        num_nonzeros = 0
        p_grad = jnp.zeros_like(p_params)
        to_log_v = 0.
        total_tries = 0
        while num_nonzeros <= num_samples_plan:
            total_tries += 1
            R_samp, P_samp = self.sample_mdp()
            V_p_pi, V_p_pi_grad = jax.value_and_grad(
                    self.policy_performance, 1)(
                    (R_samp, P_samp),
                    p_params
                )
            if np.abs(V_p_pi - risk_threshold) <= 1.e-1: #this way the value function should never get better?!
                num_nonzeros += 1
                p_grad += V_p_pi_grad
                to_log_v += V_p_pi

        print(f'total tries: {total_tries}')
        if num_nonzeros > 0:
            av_vpi = to_log_v/num_nonzeros
            p_params += self.policy_lr * p_grad/num_nonzeros
            self.policy.update_params(p_params)
            
            return av_vpi.item()
        else:
            return 0

    def CVaR_grad_step(self,
            num_samples_plan,
            k_value,
            risk_threshold,
            R_j,
            P_j
        ):
        p_params = self.policy.get_params()
        U_pi, U_pi_grads, var_alpha, cvar_alpha, samples_taken = self.posterior_sampling(p_params, num_samples_plan, risk_threshold, R_j=R_j, P_j=P_j)
        
        one_indices = np.nonzero(U_pi<var_alpha)
        grad_terms = U_pi_grads[one_indices]
        avg_term = 1./(risk_threshold*num_samples_plan)
        p_grad = avg_term * np.sum(grad_terms, axis=0) 
        av_vpi = np.mean(U_pi)

        p_params = np.clip(p_params + self.policy_lr * p_grad, 1.e-6, 1.-1e-6)
        grad_norm = np.linalg.norm(p_grad)

        # self.counter += 1
        # with open(f'grad-cvar/iter_{self.counter}.npy', 'wb') as fh:
            # np.save(fh, p_grad)
        #extra bookkeeping for subgradient step
        # if av_vpi > self.f_best: #not totally correct, have to make sure the next av_Vpi is less than the current one to update
        #     self.f_best = av_vpi
        #     self.x_best = p_params
        self.policy.update_params(p_params)
        return av_vpi, var_alpha, cvar_alpha, grad_norm, samples_taken

    def psrl_CVaR_constrained(self,
            optimism_type,
            num_samples_plan,
            k_value,
            risk_threshold,
            R_j,
            P_j
            ):

        cvar_constraint = self.constraint
        cvar_alpha = cvar_constraint - 1. #just initialization
        V_p_pi = 0
        suggested_p_params = self.policy.get_params()
        trials = 0
        best_suggested = self.policy.get_params()
        V_p_pi, V_p_pi_grad, _, cvar_alpha = self.posterior_sampling(best_suggested, num_samples_plan, risk_threshold, R_j=R_j, P_j=P_j)
        best_cvar = cvar_alpha
        best_V_p_pi = np.mean(V_p_pi, axis=0)

        R_samp, P_samp = self.sample_mdp()
        # for i in range(100):
        i = 0
        while not np.isclose(np.linalg.norm(V_p_pi_grad),0.0, atol=1e-2) and i <= 10000:
            if (i + 1) % 2500 == 0:
                self.policy_lr /= 10.
            V_p_pi, V_p_pi_grad = jax.lax.stop_gradient(jax.value_and_grad(policy_performance, 2)(R_samp, P_samp, suggested_p_params, self.initial_distribution, self.nState, self.nAction, self.discount))
            print(f'iter: {i}, Vppi: {V_p_pi}, cvar: {cvar_alpha}')
            suggested_p_params += self.policy_lr * np.mean(V_p_pi_grad, axis=0)
            i+=1
        V_p_pi, _, var_alpha, cvar_alpha = self.posterior_sampling(suggested_p_params, num_samples_plan, risk_threshold)
        if cvar_alpha > best_cvar:
            best_suggested = suggested_p_params
            best_cvar = cvar_alpha
            best_V_p_pi = np.mean(V_p_pi, axis=0)

        while (cvar_alpha < cvar_constraint) and trials < 50:
            R_samp, P_samp = self.sample_mdp()
            i = 0
            # for i in range(200):
            while not np.isclose(np.linalg.norm(V_p_pi_grad),0.0, atol=1e-2) and i <= 1000: 
                # suggested_policy = np.array(softmax(suggested_p_params))
                # grad_pi = jax.jacrev(softmax)(suggested_p_params)
                V_p_pi, V_p_pi_grad = jax.lax.stop_gradient(jax.value_and_grad(policy_performance, 2)(R_samp, P_samp, suggested_p_params, self.initial_distribution, self.nState, self.nAction, self.discount))
                print(f'iter: {i}, Vppi: {V_p_pi}, cvar: {cvar_alpha}')
                suggested_p_params += self.policy_lr * np.mean(V_p_pi_grad, axis=0)
                i+=1
            V_p_pi, _, var_alpha, cvar_alpha = self.posterior_sampling(suggested_p_params, num_samples_plan, risk_threshold)
            if cvar_alpha > best_cvar:
                best_suggested = suggested_p_params
                best_cvar = cvar_alpha
                best_V_p_pi = np.mean(V_p_pi, axis=0)
            trials += 1

        self.policy.update_params(best_suggested)
        return best_V_p_pi, 0, best_cvar, 0

    # @profile
    def posterior_sampling(self,
        p_params,
        num_samples_plan,
        risk_threshold,
        R_j=None,
        P_j=None
        ):
        significance_level = 0.1
        eps_rel = 0.1
        U_pi = np.zeros(num_samples_plan)
        U_pi_grads = np.zeros((num_samples_plan, self.nState, self.nAction))#[]
        continue_sampling = True
        total_samples = 0
        vmap_calc_value_and_grad = jax.vmap(calculate_value_and_grad, in_axes=(0,0,None,None,None,None,None))
        grad_perf = jax.value_and_grad(policy_performance, 2)
        vmap_grad = jax.vmap(grad_perf, in_axes=(0,0,None,None,None,None,None))
        sample_until = False
        if sample_until:
            while continue_sampling:
                R_j, P_j = self.multiple_sample_mdp(num_samples_plan)
                # policy = softmax(p_params)
                # grad_pi = jax.jacrev(softmax)(p_params)

                # U_pi_j, U_pi_j_grad = jax.lax.stop_gradient(vmap_calc_value_and_grad(R_j, P_j, grad_pi, policy, self.nState, self.discount, self.initial_distribution))

                U_pi_j, U_pi_j_grad = jax.lax.stop_gradient(vmap_grad(R_j, P_j, p_params, self.initial_distribution, self.nState, self.nAction, self.discount))

                U_pi_j = np.asarray(U_pi_j)
                U_pi_j_grad = np.asarray(U_pi_j_grad)
                # print(jax_U, U_pi_j)
                # print('values', np.isclose(U_pi_j, jax_U, rtol=1e-4))
                # print(np.isclose(U_pi_j_grad, jax_U_grad, rtol=1e-4))
                # U_pi[j] = U_pi_j
                # U_pi_grads[j] = U_pi_j_grad
                if total_samples == 0:
                    U_pi = U_pi_j
                    U_pi_grads = U_pi_j_grad
                else:
                    U_pi = np.concatenate((U_pi, U_pi_j), axis=0)
                    U_pi_grads = np.concatenate((U_pi_grads, U_pi_j_grad), axis=0)
                del R_j, P_j, U_pi_j, U_pi_j_grad
                total_samples += num_samples_plan
                L_pi = U_pi.shape[0]
                sorted_U_pi = np.sort(U_pi)

                #find s,r indices
                mode = int(np.floor((L_pi + 1)*risk_threshold))
                r = mode
                s = mode
                prob_r_s_x = lambda r_x, s_x: np.sum(np.array([comb(L_pi, i)* risk_threshold**i * (1-risk_threshold)**(L_pi - i) for i in range(r_x, s_x)]))
                prob_r_s = prob_r_s_x(r, s)
                i = 0
                while (prob_r_s <= 1 - significance_level) and ((s != L_pi - 1) and (r != 0)): #if both have reached the ends of the array, then we have the largest interval and shouldn't keep looping
                    if (r > 0) and (i % 2 == 0):
                        r = r - 1
                    if (s < L_pi - 1) and (i % 2 == 1):
                        s = s + 1
                    prob_r_s = prob_r_s_x(r, s)
                    i += 1
                continue_sampling = (sorted_U_pi[s] - sorted_U_pi[r] >= eps_rel * (sorted_U_pi[-1] - sorted_U_pi[0]))
        else:
            U_pi_j, U_pi_j_grad = jax.lax.stop_gradient(vmap_grad(R_j, P_j, p_params, self.initial_distribution, self.nState, self.nAction, self.discount))

            U_pi = np.asarray(U_pi_j)
            U_pi_grads = np.asarray(U_pi_j_grad)
            
            total_samples += num_samples_plan
            L_pi = U_pi.shape[0]
            sorted_U_pi = np.sort(U_pi)

        U_pi = np.asarray(U_pi)
        U_pi_grads = np.asarray(U_pi_grads)
        floor_index = int(np.floor(risk_threshold * L_pi))
        var_alpha = sorted_U_pi[floor_index]
        cvar_alpha = np.mean(sorted_U_pi[:floor_index])
        return U_pi, U_pi_grads, var_alpha, cvar_alpha, total_samples

    def optimistic_only(self,
            optimism_type,
            num_samples_plan,
            k_value,
            risk_threshold,
            R_j,
            P_j
        ):
        '''
        optimism_type: 'max', 'upper-cvar'
        '''
        p_params = self.policy.get_params()
        U_pi, U_pi_grads, var_alpha, cvar_alpha, samples_taken = self.posterior_sampling(p_params, num_samples_plan, risk_threshold, R_j=R_j, P_j=P_j)
        av_vpi = np.mean(U_pi)
        upper_risk_threshold = 0.4

        if optimism_type == 'max':
            optimistic_grad = U_pi_grads[np.argmax(U_pi)]
            objective = np.max(U_pi)
        elif optimism_type == 'upper-cvar':
            # upper_cvar_alpha = self.CVaR_estimate(U_pi, 1 - upper_risk_threshold)
            # objective = upper_cvar_alpha
            upper_v_alpha_quantile, _ = self.quantile_estimate(U_pi, 1 - upper_risk_threshold)
            upper_one_indices = np.nonzero(U_pi>upper_v_alpha_quantile)[0]
            upper_cvar_grad_terms = U_pi_grads[upper_one_indices]
            upper_avg_term = 1./((1-upper_risk_threshold)*num_samples_plan)
            objective = upper_avg_term * np.sum(U_pi[upper_one_indices], axis=0)
            optimistic_grad = upper_avg_term * np.sum(upper_cvar_grad_terms, axis=0)
        else:
            raise NotImplementedError(f'{optimism_type} not implemented')

        p_grad = optimistic_grad
        grad_norm = np.linalg.norm(p_grad)
        # print(grad_norm)

        p_params += self.policy_lr * p_grad
        self.policy.update_params(p_params)
        del U_pi, U_pi_grads
        gc.collect()

        # print(f'Train_step: {train_step}, Model Value fn: {av_vpi:.3f}, lambda: {self.lambda_param:.3f}, cvar: {cvar_alpha.item():.3f}')
        return av_vpi, objective, cvar_alpha, grad_norm, samples_taken
        # return av_vpi, 0, 0, grad_norm

    def both_max_CVaR(self,
            num_samples_plan: int,
            k_value: int,
            risk_threshold: float,
            R_j,
            P_j
        ):
        p_params = self.policy.get_params()
        U_pi, U_pi_grads, var_alpha, cvar_alpha, samples_taken = self.posterior_sampling(p_params, num_samples_plan, risk_threshold, R_j=R_j, P_j=P_j)

        objective = 0
        one_indices = np.nonzero(U_pi<var_alpha)
        cvar_grad_terms = U_pi_grads[one_indices]
        avg_term = 1./(risk_threshold*num_samples_plan)
        constraint_grad = avg_term * np.sum(cvar_grad_terms, axis=0) 
        av_vpi = np.mean(U_pi)

        objective += self.lambda_param * cvar_alpha
        
        upper_risk_threshold = risk_threshold
        # upper_cvar_alpha = self.CVaR_estimate(U_pi, 1 - upper_risk_threshold)
        upper_v_alpha_quantile, _ = self.quantile_estimate(U_pi, 1 - upper_risk_threshold)
        upper_one_indices = np.nonzero(U_pi>upper_v_alpha_quantile)
        upper_cvar_grad_terms = U_pi_grads[upper_one_indices]
        upper_avg_term = 1./((1-upper_risk_threshold)*num_samples_plan)

        objective += upper_avg_term * np.sum(U_pi[upper_one_indices], axis=0)

        optimistic_grad = upper_avg_term * np.sum(upper_cvar_grad_terms, axis=0)
       
        p_grad = optimistic_grad + self.lambda_param * constraint_grad
        p_params = p_params + self.policy_lr * p_grad
        grad_norm = np.linalg.norm(p_grad)
        self.policy.update_params(p_params)

        return av_vpi, objective, cvar_alpha, grad_norm

    def optimistic_CVaR_constrained(self,
            optimism_type,
            num_samples_plan,
            k_value,
            risk_threshold,
            R_j,
            P_j
        ):
        '''
        optimism_type: 'pg' (not optimistic), 'max', 'optimistic-psrl', 'upper-cvar', 'random'
        '''
        damping = 10.
        cvar_constraint = self.constraint #should be based on the risk_threshold
        p_params = self.policy.get_params()
        U_pi, U_pi_grads, var_alpha, cvar_alpha, samples_taken = self.posterior_sampling(p_params, num_samples_plan, risk_threshold, R_j=R_j, P_j=P_j)

        one_indices = np.nonzero(U_pi<var_alpha)
        cvar_grad_terms = U_pi_grads[one_indices]
        avg_term = 1./(risk_threshold*num_samples_plan)
        constraint_grad = avg_term * np.sum(cvar_grad_terms, axis=0) 
        av_vpi = np.mean(U_pi)
        objective = 0
        # optimistic_grad = pi_grads[np.argmax(v_samples)] #change
        upper_risk_threshold = risk_threshold
        if optimism_type == 'random':
            optimistic_grad = U_pi_grads[jnp.random.choice(num_samples_plan)] #this is closer to what we do in psrl
        elif optimism_type == 'pg':
            optimistic_grad = np.mean(U_pi_grads, axis=0)
        elif optimism_type == 'optimistic-psrl':
            random_idcs = jnp.random.choice(num_samples_plan, size=10)
            optimistic_grad = U_pi_grads[random_idcs][jnp.argmax(U_pi[random_idcs])]
        elif optimism_type == 'max':
            optimistic_grad = U_pi_grads[np.argmax(U_pi)] 
            objective = np.max(U_pi)
        elif optimism_type == 'upper-cvar':
            # upper_cvar_alpha = self.CVaR_estimate(U_pi, 1 - upper_risk_threshold)
            upper_v_alpha_quantile, _ = self.quantile_estimate(U_pi, 1 - upper_risk_threshold)
            upper_one_indices = np.nonzero(U_pi>upper_v_alpha_quantile)
            upper_cvar_grad_terms = U_pi_grads[upper_one_indices]
            upper_avg_term = 1./((1-upper_risk_threshold)*num_samples_plan)
            objective = upper_avg_term * np.sum(U_pi[upper_one_indices], axis=0)
            optimistic_grad = upper_avg_term * np.sum(upper_cvar_grad_terms, axis=0)
        else:
            raise NotImplementedError(f'{optimism_type} not implemented')

        # damp = damping * (cvar_alpha - cvar_constraint)
        damp = 0
        p_grad = optimistic_grad + (self.lambda_param - damp) * constraint_grad
        # print(np.linalg.norm((self.lambda_param - damp) * constraint_grad))
        p_params = p_params + self.policy_lr * p_grad
        grad_norm = np.linalg.norm(p_grad)
        self.policy.update_params(p_params)

        #update lambda here
        # lambda_grad = cvar_alpha - cvar_constraint
        # print(lambda_grad)
        # self.lambda_param = np.clip(self.lambda_param - self.lambda_lr * lambda_grad, 0, None) #lambda >= 0
        # if cvar_alpha > cvar_constraint:
        #     # print('set to zero')
        #     self.lambda_param = 0. 
        # print(av_vpi, var_alpha, cvar_alpha, grad_norm)
        return av_vpi, objective, cvar_alpha, grad_norm

    def regret_CVaR_grad_step(self,
            num_samples_plan,
            k_value,
            risk_threshold
        ):
        p_params = self.policy.get_params()
        # v_samples = np.zeros(num_samples_plan)
        # v_opt_samples = np.zeros(num_samples_plan)
        regret_samples = np.zeros(num_samples_plan) #stores -regret
        pi_grads = np.zeros((num_samples_plan, self.nState, self.nAction))
        for sample in range(num_samples_plan):
            R_samp, P_samp = self.sample_mdp()
        
            V_p_pi, V_p_pi_grad = jax.value_and_grad(
                self.policy_performance, 1)(
                (R_samp, P_samp),
                p_params
            )
            V_p_opt, _ = self.value_iteration(R_samp, P_samp)
            # v_opt_samples[sample] = V_p_opt
            regret_samples[sample] = V_p_pi - (self.initial_distribution @ V_p_opt)
            # v_samples[sample] = V_p_pi
            pi_grads[sample] = V_p_pi_grad

        v_alpha_quantile, _ = self.quantile_estimate(regret_samples, risk_threshold)
        cvar_alpha = self.CVaR_estimate(regret_samples, risk_threshold)
        # grads = np.einsum('i,ijk->ijk',(v_samples - v_alpha_quantile), pi_grads)
        # one_indices = np.nonzero(v_samples>=v_alpha_quantile)
        one_indices = np.nonzero(regret_samples<v_alpha_quantile)
        grad_terms = pi_grads[one_indices]
        avg_term = 1./(risk_threshold*num_samples_plan)
        p_grad = avg_term * np.sum(grad_terms, axis=0) 
        av_vpi = np.mean(regret_samples)

        p_params += self.policy_lr * p_grad
        
        #extra bookkeeping for subgradient step
        # if av_vpi > self.f_best: #not totally correct, have to make sure the next av_Vpi is less than the current one to update
        #     self.f_best = av_vpi
        #     self.x_best = p_params
        self.policy.update_params(p_params)
        return av_vpi.item(), v_alpha_quantile.item(), cvar_alpha.item(), samples_taken

    def quantile_estimate(self, v_samples, alpha):
        sorted_indices = np.argsort(v_samples)
        quantile_index = int(np.ceil(alpha * v_samples.shape[0]))
        return v_samples[sorted_indices[quantile_index]], quantile_index

    def CVaR_estimate(self, v_samples, alpha):
        VaR_alpha, VaR_index = self.quantile_estimate(v_samples, alpha)
        sorted_indices = np.argsort(v_samples)
        return np.mean(v_samples[sorted_indices[:VaR_index]])

    def grad_risk_evaluation(self,
            num_samples_plan,
            k_value,
            risk_threshold,
        ):
        eps_rel = 0.1
        significance_level = 0.1
        p_params = self.policy.get_params()
        cvar, grad_cvar = jax.value_and_grad(self.risk_evaluation_for_grad, 0)(p_params, eps_rel, significance_level, risk_threshold, num_samples_plan)
        
        p_params += self.policy_lr * grad_cvar
        self.counter += 1
        with open(f'grad-risk-eval/iter_{self.counter}.npy', 'wb') as fh:
            np.save(fh, np.array(grad_cvar))

        self.policy.update_params(p_params)
        return 0, 0, cvar.item()

    def risk_evaluation_for_grad(self, p_params, eps_rel, alpha, q, num_samples):
        #loop in parallel sample from posterior
        U_pi = []
        continue_sampling = True
        total_samples = 0
        while continue_sampling:
            for j in range(num_samples): #should vectorize this!
                total_samples += 1
                R_j, P_j = self.sample_mdp()
                U_pi_j = self.policy_performance((R_j, P_j), p_params)
                U_pi.append(U_pi_j)
            L_pi = len(U_pi)
            sorted_U_pi = jnp.sort(jnp.array(U_pi))
            
            #find s,r indices
            mode = int(jnp.floor((L_pi + 1)*q))
            r = mode
            s = mode
            prob_r_s_x = lambda r_x, s_x: jnp.sum(jnp.array([comb(L_pi, i)* q**i * (1-q)**(L_pi - i) for i in range(r_x, s_x)]))
            prob_r_s = prob_r_s_x(r, s)
            i = 0
            while (prob_r_s <= 1 - alpha) and ((s != L_pi - 1) and (r != 0)): #if both have reached the ends of the array, then we have the largest interval and shouldn't keep looping
                if (r > 0) and (i % 2 == 0):
                    r = r - 1
                if (s < L_pi - 1) and (i % 2 == 1):
                    s = s + 1
                prob_r_s = prob_r_s_x(r, s)
                i += 1
            continue_sampling = (sorted_U_pi[s] - sorted_U_pi[r] >= eps_rel * (sorted_U_pi[-1] - sorted_U_pi[0])) and (total_samples < 200)

        floor_index = int(jnp.floor(q * L_pi))
        # var = sorted_U_pi[floor_index]
        cvar = jnp.mean(sorted_U_pi[:floor_index])
        self.total_risk_samples = total_samples
        # print(total_samples)
        return cvar

    ############### BELOW FOR MC2PS #################################
    def MC2PS(self,
            batch_size,
            num_models,
            num_discounts,
            sigma,
            eps_rel,
            significance_level,
            risk_threshold):
        '''
        sigma: str (for now): risk measure 'CVaR' or 'VaR'
        eps_rel: float in [0,1]: relative error tolerance
        significance level: float in [0, 1]: siginificance level (controls number of samples needed)
        risk_threshold: float in [0, 1]: quantile order

        Return: 
        '''
        discount_factors = np.linspace(0.1, self.discount, num_discounts)
        all_policies, all_discounts = self.generate_policies(discount_factors, num_models) #check shape of this
        # U_all = np.zeros(len(all_p_params))
        var_all = np.zeros(len(all_policies))
        cvar_all = np.zeros(len(all_policies))
        for pi_idx in range(len(all_policies)):
            var_all[pi_idx], cvar_all[pi_idx] = self.risk_evaluation(all_policies[pi_idx], all_discounts[pi_idx], eps_rel, significance_level, risk_threshold, batch_size)
        #find best policy in a set
        U_all = cvar_all if (sigma == 'CVaR') else var_all
        pi_star_idx = np.argmax(U_all)
        pi_star_params = all_policies[pi_star_idx]
        self.policy.update_params(pi_star_params)
        print(f'var: {var_all[pi_star_idx]}, cvar: {cvar_all[pi_star_idx]}')
        return var_all[pi_star_idx], cvar_all[pi_star_idx]
    
    def generate_policies(self, discount_factors, num_models):
        evaluation_discount = self.discount
        models_R = np.zeros((num_models + 1, self.nState, self.nAction))
        models_P = np.zeros((num_models + 1, self.nState, self.nAction, self.nState))
        models_R[0], models_P[0] = self.get_CE_model()
        for m in range(1,  num_models):
            models_R[m], models_P[m] = self.sample_mdp()
        
        policies = []
        discounts = []
        for gamma, model_idx in product(discount_factors, range(num_models)):
            self.discount = gamma
            _, pi_opt = self.value_iteration(models_R[model_idx], models_P[model_idx])
            #check if policy in list of policies
            if (policies == []) or (not np.any(np.all(pi_opt == np.array(policies), axis=1))):
                policies.append(pi_opt)
                discounts.append(gamma)
        
        self.discount = evaluation_discount
        return np.array(policies), np.array(discounts)

    #try taking gradient of this and compare to CVaR one! (have to use p_params instead of policy, and policy_performance instead of policy_evaluation)
    def risk_evaluation(self, policy, gamma_pi, eps_rel, alpha, q, num_samples):
        evaluation_discount = self.discount
        self.discount = gamma_pi
        #loop in parallel sample from posterior
        U_pi = []
        continue_sampling = True
        while continue_sampling:
            for j in range(num_samples): #should vectorize this!
                R_j, P_j = self.sample_mdp()
                U_pi_j = self.initial_distribution @ self.policy_evaluation((R_j, P_j), policy)
                U_pi.append(U_pi_j)
            L_pi = len(U_pi)
            sorted_U_pi = np.sort(np.array(U_pi))
            
            #find s,r indices
            mode = int(np.floor((L_pi + 1)*q))
            r = mode
            s = mode
            prob_r_s_x = lambda r_x, s_x: np.sum(np.array([comb(L_pi, i)* q**i * (1-q)**(L_pi - i) for i in range(r_x, s_x)]))
            prob_r_s = prob_r_s_x(r, s)
            i = 0
            while (prob_r_s <= 1 - alpha) and ((s != L_pi - 1) and (r != 0)): #if both have reached the ends of the array, then we have the largest interval and shouldn't keep looping
                if (r > 0) and (i % 2 == 0):
                    r = r - 1
                if (s < L_pi - 1) and (i % 2 == 1):
                    s = s + 1
                prob_r_s = prob_r_s_x(r, s)
                i += 1
            continue_sampling = (sorted_U_pi[s] - sorted_U_pi[r] >= eps_rel * (sorted_U_pi[-1] - sorted_U_pi[0]))

        floor_index = int(np.floor(q * L_pi))
        var = sorted_U_pi[floor_index]
        cvar = np.mean(sorted_U_pi[:floor_index])
        self.discount = evaluation_discount
        return var, cvar
    ############### ABOVE FOR MC2PS #################################

    def robust_DP(self, *args, **kwargs):
        # this is not a sampling based algorithm, 
        # how to build confidence set?
        # robust_vi implementation how? Does training algorithm need to change as well?
        pass

    def k_of_N(self,
            num_samples_plan,
            k_value,
            risk_threshold
        ):
        assert (k_value <= num_samples_plan)
        p_params = self.policy.get_params()
        N_values = np.zeros(num_samples_plan)
        N_value_grads = np.zeros((num_samples_plan, self.nState, self.nAction))

        for sample in range(num_samples_plan):
            R_samp, P_samp = self.sample_mdp()
        
            V_p_pi, V_p_pi_grad = jax.value_and_grad(
                self.policy_performance, 1)(
                (R_samp, P_samp),
                p_params
            )
            N_values[sample] = V_p_pi
            N_value_grads[sample] = V_p_pi_grad
        
        # pick randomly from bottom k models
        sorted_bottom_k_index = np.argsort(N_values)[:k_value]
        rnd_sample = self.rng.choice(sorted_bottom_k_index)
        p_grad = N_value_grads[rnd_sample]
        av_vpi = N_values[rnd_sample]
        
        p_params += self.policy_lr * p_grad
        self.policy.update_params(p_params)
        
        v_alpha_quantile, _ = self.quantile_estimate(N_values, risk_threshold)
        cvar_alpha = self.CVaR_estimate(N_values, risk_threshold)
        return av_vpi, v_alpha_quantile, cvar_alpha

    def VaR_sigmoid_grad_step(self,
            num_samples_plan,
            k_value,
            risk_threshold
        ):
        to_log_v = 0
        p_params = self.policy.get_params()
        v_samples = np.zeros(num_samples_plan)
        pi_grads = np.zeros((num_samples_plan, self.nState, self.nAction))
        # p_grad = jnp.zeros_like(p_params)
        for sample in range(num_samples_plan):
            R_samp, P_samp = self.sample_mdp()
        
            V_p_pi, V_p_pi_grad = jax.value_and_grad(
                self.policy_performance, 1)(
                (R_samp, P_samp),
                p_params
            )

            v_samples[sample] = V_p_pi
            pi_grads[sample] = V_p_pi_grad

        v_alpha_quantile, _ = self.quantile_estimate(v_samples, risk_threshold)
        
        cvar_alpha = self.CVaR_estimate(v_samples, risk_threshold)

        sig_grad = jax.vmap(jax.grad(jax.nn.sigmoid))(v_samples - v_alpha_quantile)
        
        p_grad = np.mean(np.einsum('i,ijk->ijk', np.array(sig_grad).reshape(-1,), pi_grads), axis=0)
        
        av_vpi = np.mean(v_samples)
        p_params += self.policy_lr * p_grad
        self.policy.update_params(p_params)
        
        return av_vpi.item(), v_alpha_quantile.item(), cvar_alpha.item()
