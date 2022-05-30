import numpy as np
import jax.numpy as jnp
import jax
from itertools import product
from typing import List, Tuple, Dict
from scipy.special import comb
import pickle

CVAR_CONSTRAINT = -1.0 #RFL
# CVAR_CONSTRAINT = -4.0 #cliffwalking

class Agent:
    def __init__(self, nState, discount, initial_distribution, policy) -> None:
        self.nState = nState
        self.discount = discount
        self.initial_distribution = initial_distribution
        self.policy = policy
        self.counter = 0
        self.lambda_param = 10.
        
    def policy_evaluation(self, mdp: Tuple[np.ndarray], policy: jnp.ndarray): 
        # Vals[state, timestep]
        r_matrix, p_matrix = mdp
        ppi = jnp.einsum('sat,sa->st', p_matrix, policy)
        rpi = jnp.einsum('sa,sa->s', r_matrix, policy)
        v_pi = jnp.linalg.solve(np.eye(self.nState) -
                               self.discount*ppi, rpi)
        return v_pi

    def policy_performance(self, mdp, p_params):
        vf = self.policy_evaluation(mdp, self.policy.get_policy(p_params))
        return self.initial_distribution @ vf

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
    def grad_step(self, train_type, **kwargs):
        if train_type == 'VaR-sigmoid':
            return self.VaR_sigmoid_grad_step(**kwargs)
        elif train_type == 'VaR-delta':
            return self.VaR_delta_grad_step(**kwargs)
        #optimistic only
        elif train_type == 'max-opt':
            return self.optimistic_only(optimism_type='max', **kwargs)
        elif train_type == 'upper-cvar':
            return self.optimistic_only(optimism_type='upper-cvar', **kwargs)
        #optimistic + cvar constrained
        elif train_type == 'psrl-opt-cvar':
            return self.psrl_CVaR_constrained(optimism_type='psrl', **kwargs) 
        elif train_type == 'max-opt-cvar':
            return self.optimistic_CVaR_constrained(optimism_type='max', **kwargs) 
        elif train_type == 'optimistic-psrl-opt-cvar':
            return self.optimistic_CVaR_constrained(optimism_type='optimistic-psrl', **kwargs) 
        elif train_type == 'upper-cvar-opt-cvar':
            return self.optimistic_CVaR_constrained(optimism_type='upper-cvar', **kwargs) 
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
            raise NotImplementedError('Not implemented this type')

    def pg(self,
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None
        ):
        p_params = self.policy.get_params()
        to_log_v = 0
        p_grad = jnp.zeros_like(p_params)
        v_samples = np.zeros(num_samples_plan)
        for sample in range(num_samples_plan):
            R_samp, P_samp = self.sample_mdp()
        
            V_p_pi, V_p_pi_grad = jax.value_and_grad(
                self.policy_performance, 1)(
                (R_samp, P_samp),
                p_params
            )
            p_grad += V_p_pi_grad
            to_log_v += V_p_pi
            v_samples[sample] = V_p_pi

        av_vpi = to_log_v/num_samples_plan
        p_params += p_lr * p_grad/num_samples_plan
        self.policy.update_params(p_params)
        v_alpha_quantile, _ = self.quantile_estimate(v_samples, risk_threshold)
        cvar_alpha = self.CVaR_estimate(v_samples, risk_threshold)
        return av_vpi, v_alpha_quantile, cvar_alpha

    def psrl(self,
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None
        ):
        p_params = self.policy.get_params()
        R_samp, P_samp = self.sample_mdp()
        
        for i in range(100):
            V_p_pi, V_p_pi_grad = jax.value_and_grad(
                self.policy_performance, 1)(
                (R_samp, P_samp),
                p_params
            )
            # print(f'iter: {i}, Vppi: {V_p_pi}')
            p_params += p_lr * V_p_pi_grad
        
        self.policy.update_params(jax.lax.stop_gradient(p_params))
        eps_rel = 0.1
        significance_level = 0.1
        cvar_alpha = jax.lax.stop_gradient(self.risk_evaluation_for_grad(self.policy.p_params, eps_rel, significance_level, risk_threshold, num_samples_plan))
        
        return jax.lax.stop_gradient(V_p_pi), 0, cvar_alpha

    def pg_CE(self,
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None
        ):
        p_params = self.policy.get_params()
        p_grad = jnp.zeros_like(p_params)
        R_CE, P_CE = self.get_CE_model()
        V_p_pi, V_p_pi_grad = jax.value_and_grad(
            self.policy_performance, 1)(
            (R_CE, P_CE),
            p_params
        )
        p_grad = V_p_pi_grad
        av_vpi = V_p_pi
        p_params += p_lr * p_grad
        self.policy.update_params(p_params)
        
        v_alpha_quantile = 0 #placeholder, not sure how to do this one
        cvar_alpha = 0
        return av_vpi, v_alpha_quantile, cvar_alpha

    def VaR_delta_grad_step(self,
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None
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
            p_params += p_lr * p_grad/num_nonzeros
            self.policy.update_params(p_params)
            
            return av_vpi
        else:
            return 0

    def old_CVaR_grad_step(self,
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None
        ):
        U_pi = []
        continue_sampling = True
        total_samples = 0
        p_params = self.policy.get_params()
        v_samples = np.zeros(num_samples_plan)
        pi_grads = np.zeros((num_samples_plan, self.nState, self.nAction))
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
        # grads = np.einsum('i,ijk->ijk',(v_samples - v_alpha_quantile), pi_grads)
        # one_indices = np.nonzero(v_samples>=v_alpha_quantile)
        one_indices = np.nonzero(v_samples<v_alpha_quantile)
        grad_terms = pi_grads[one_indices]
        avg_term = 1./(risk_threshold*num_samples_plan)
        p_grad = avg_term * np.sum(grad_terms, axis=0) 
        av_vpi = np.mean(v_samples)

        p_params += p_lr * p_grad

        self.counter += 1
        with open(f'grad-cvar/iter_{self.counter}.npy', 'wb') as fh:
            np.save(fh, p_grad)
        #extra bookkeeping for subgradient step
        # if av_vpi > self.f_best: #not totally correct, have to make sure the next av_Vpi is less than the current one to update
        #     self.f_best = av_vpi
        #     self.x_best = p_params
        print(num_samples_plan)
        self.policy.update_params(p_params)

        return av_vpi, v_alpha_quantile, cvar_alpha

    def CVaR_grad_step(self,
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None
        ):
        significance_level = 0.1
        eps_rel = 0.1
        U_pi = []
        U_pi_grads = []
        continue_sampling = True
        total_samples = 0
        p_params = self.policy.get_params()
        while continue_sampling:
            for j in range(num_samples_plan):
                total_samples += 1
                R_j, P_j = self.sample_mdp()
                U_pi_j, U_pi_j_grad = jax.value_and_grad(
                    self.policy_performance, 1)(
                    (R_j, P_j),
                    p_params
                )
                U_pi.append(U_pi_j)
                U_pi_grads.append(U_pi_j_grad)
            L_pi = len(U_pi)
            sorted_U_pi = jnp.sort(jnp.array(U_pi))

            #find s,r indices
            mode = int(jnp.floor((L_pi + 1)*risk_threshold))
            r = mode
            s = mode
            prob_r_s_x = lambda r_x, s_x: jnp.sum(jnp.array([comb(L_pi, i)* risk_threshold**i * (1-risk_threshold)**(L_pi - i) for i in range(r_x, s_x)]))
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

        U_pi = np.asarray(U_pi)
        U_pi_grads = np.asarray(U_pi_grads)
        floor_index = int(jnp.floor(risk_threshold * L_pi))
        var_alpha = sorted_U_pi[floor_index]
        cvar_alpha = jnp.mean(sorted_U_pi[:floor_index])

        one_indices = np.nonzero(U_pi<var_alpha)
        grad_terms = U_pi_grads[one_indices]
        avg_term = 1./(risk_threshold*num_samples_plan)
        p_grad = avg_term * np.sum(grad_terms, axis=0) 
        av_vpi = np.mean(U_pi)

        p_params += p_lr * p_grad

        self.counter += 1
        with open(f'grad-cvar/iter_{self.counter}.npy', 'wb') as fh:
            np.save(fh, p_grad)
        #extra bookkeeping for subgradient step
        # if av_vpi > self.f_best: #not totally correct, have to make sure the next av_Vpi is less than the current one to update
        #     self.f_best = av_vpi
        #     self.x_best = p_params
        print(total_samples)
        self.policy.update_params(p_params)
        return av_vpi, var_alpha, cvar_alpha

    def psrl_CVaR_constrained(self,
            optimism_type='psrl',
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None
            ):

        eps_rel = 0.1
        significance_level = 0.1
        cvar_constraint = CVAR_CONSTRAINT
        cvar_alpha = cvar_constraint + 1. #just initialization

        suggested_p_params = self.policy.get_params()
        
        while cvar_alpha > cvar_constraint:
            R_samp, P_samp = self.sample_mdp()
            
            for i in range(100):
                V_p_pi, V_p_pi_grad = jax.value_and_grad(
                    self.policy_performance, 1)(
                    (R_samp, P_samp),
                    suggested_p_params
                )
                print(f'iter: {i}, Vppi: {V_p_pi}')
                suggested_p_params += p_lr * V_p_pi_grad
                # self.policy.update_params(p_params)
            
            cvar_alpha = jax.lax.stop_gradient(self.risk_evaluation_for_grad(suggested_p_params, eps_rel, significance_level, risk_threshold, num_samples_plan))
        
        self.policy.update_params(suggested_p_params)
        return V_p_pi, 0, cvar_alpha
        

    def optimistic_only(self,
            optimism_type='upper-cvar',
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None,
        ):
        '''
        optimism_type: 'max', 'upper-cvar'
        '''
        significance_level = 0.1
        eps_rel = 0.1
        U_pi = []
        U_pi_grads = []
        continue_sampling = True
        total_samples = 0
        p_params = self.policy.get_params()
        while continue_sampling:
            for j in range(num_samples_plan):
                total_samples += 1
                R_j, P_j = self.sample_mdp()
                U_pi_j, U_pi_j_grad = jax.value_and_grad(
                    self.policy_performance, 1)(
                    (R_j, P_j),
                    p_params
                )
                U_pi.append(U_pi_j)
                U_pi_grads.append(U_pi_j_grad)
            L_pi = len(U_pi)
            sorted_U_pi = jnp.sort(jnp.array(U_pi))

            #find s,r indices
            mode = int(jnp.floor((L_pi + 1)*risk_threshold))
            r = mode
            s = mode
            prob_r_s_x = lambda r_x, s_x: jnp.sum(jnp.array([comb(L_pi, i)* risk_threshold**i * (1-risk_threshold)**(L_pi - i) for i in range(r_x, s_x)]))
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

        U_pi = jnp.asarray(U_pi)
        U_pi_grads = jnp.asarray(U_pi_grads)
        floor_index = int(jnp.floor(risk_threshold * L_pi))
        var_alpha = sorted_U_pi[floor_index]
        cvar_alpha = jnp.mean(sorted_U_pi[:floor_index])

        av_vpi = jnp.mean(U_pi)
        upper_risk_threshold = 0.4
        if optimism_type == 'max':
            optimistic_grad = U_pi_grads[jnp.argmax(U_pi)] 
        elif optimism_type == 'upper-cvar':
            upper_cvar_alpha = self.CVaR_estimate(U_pi, 1 - upper_risk_threshold)
            upper_v_alpha_quantile, _ = self.quantile_estimate(U_pi, 1 - upper_risk_threshold)
            upper_one_indices = jnp.nonzero(U_pi>upper_v_alpha_quantile)
            print(upper_one_indices)
            upper_cvar_grad_terms = U_pi_grads[upper_one_indices]
            upper_avg_term = 1./((1-upper_risk_threshold)*num_samples_plan)
            optimistic_grad = upper_avg_term * jnp.sum(upper_cvar_grad_terms, axis=0)
        else:
            raise NotImplementedError(f'{optimism_type} not implemented')

        p_grad = optimistic_grad
        print(p_grad)

        p_params += p_lr * p_grad
        self.policy.update_params(p_params)

        return av_vpi, var_alpha, cvar_alpha

    def optimistic_CVaR_constrained(self,
            optimism_type='upper-cvar',
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None,
        ):
        '''
        optimism_type: 'max', 'optimistic-psrl', 'upper-cvar', 'random'
        '''

        lambda_lr = 0.05
        cvar_constraint = CVAR_CONSTRAINT #should be based on the risk_threshold

        significance_level = 0.1
        eps_rel = 0.1
        U_pi = []
        U_pi_grads = []
        continue_sampling = True
        total_samples = 0
        p_params = self.policy.get_params()
        while continue_sampling:
            for j in range(num_samples_plan):
                total_samples += 1
                R_j, P_j = self.sample_mdp()
                U_pi_j, U_pi_j_grad = jax.value_and_grad(
                    self.policy_performance, 1)(
                    (R_j, P_j),
                    p_params
                )
                U_pi.append(U_pi_j)
                U_pi_grads.append(U_pi_j_grad)
            L_pi = len(U_pi)
            sorted_U_pi = jnp.sort(jnp.array(U_pi))

            #find s,r indices
            mode = int(jnp.floor((L_pi + 1)*risk_threshold))
            r = mode
            s = mode
            prob_r_s_x = lambda r_x, s_x: jnp.sum(jnp.array([comb(L_pi, i)* risk_threshold**i * (1-risk_threshold)**(L_pi - i) for i in range(r_x, s_x)]))
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

        U_pi = jnp.asarray(U_pi)
        U_pi_grads = jnp.asarray(U_pi_grads)
        floor_index = int(jnp.floor(risk_threshold * L_pi))
        var_alpha = sorted_U_pi[floor_index]
        cvar_alpha = jnp.mean(sorted_U_pi[:floor_index])

        one_indices = jnp.nonzero(U_pi<var_alpha)
        cvar_grad_terms = U_pi_grads[one_indices]
        avg_term = 1./(risk_threshold*num_samples_plan)
        constraint_grad = avg_term * jnp.sum(cvar_grad_terms, axis=0) 
        av_vpi = jnp.mean(U_pi)

        # optimistic_grad = pi_grads[np.argmax(v_samples)] #change
        upper_risk_threshold = 0.4
        if optimism_type == 'random':
            optimistic_grad = U_pi_grads[jnp.random.choice(num_samples_plan)] #this is closer to what we do in psrl
        elif optimism_type == 'optimistic-psrl':
            random_idcs = jnp.random.choice(num_samples_plan, size=10)
            optimistic_grad = U_pi_grads[random_idcs][jnp.argmax(U_pi[random_idcs])]
        elif optimism_type == 'max':
            optimistic_grad = U_pi_grads[jnp.argmax(U_pi)] #this is closer to what we do in psrl
        elif optimism_type == 'upper-cvar':
            upper_cvar_alpha = self.CVaR_estimate(U_pi, 1 - upper_risk_threshold)
            upper_v_alpha_quantile, _ = self.quantile_estimate(U_pi, 1 - upper_risk_threshold)
            upper_one_indices = jnp.nonzero(U_pi>upper_v_alpha_quantile)
            upper_cvar_grad_terms = U_pi_grads[upper_one_indices]
            upper_avg_term = 1./((1-upper_risk_threshold)*num_samples_plan)
            optimistic_grad = upper_avg_term * jnp.sum(upper_cvar_grad_terms, axis=0)
        else:
            raise NotImplementedError(f'{optimism_type} not implemented')

        p_grad = optimistic_grad + self.lambda_param * constraint_grad

        p_params += p_lr * p_grad
        print(p_grad)
        self.policy.update_params(p_params)

        #update lambda here
        lambda_grad = cvar_alpha - cvar_constraint
        lambda_param_update = self.lambda_param - lambda_lr * lambda_grad
        self.lambda_param = max(0, lambda_param_update) #lambda >= 0
        print(self.lambda_param)
        return av_vpi, var_alpha, cvar_alpha

    def regret_CVaR_grad_step(self,
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None
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

        p_params += p_lr * p_grad
        
        #extra bookkeeping for subgradient step
        # if av_vpi > self.f_best: #not totally correct, have to make sure the next av_Vpi is less than the current one to update
        #     self.f_best = av_vpi
        #     self.x_best = p_params
        self.policy.update_params(p_params)
        return av_vpi, v_alpha_quantile, cvar_alpha

    def quantile_estimate(self, v_samples, alpha):
        sorted_indices = np.argsort(v_samples)
        quantile_index = int(np.ceil(alpha * v_samples.shape[0]))
        return v_samples[sorted_indices[quantile_index]], quantile_index

    def CVaR_estimate(self, v_samples, alpha):
        VaR_alpha, VaR_index = self.quantile_estimate(v_samples, alpha)
        sorted_indices = np.argsort(v_samples)
        return np.mean(v_samples[sorted_indices[:VaR_index]])

    def grad_risk_evaluation(self,
            num_samples_plan=None,
            k_value=None,
            p_lr=0.1,
            risk_threshold=None,
        ):
        eps_rel = 0.1
        significance_level = 0.1
        p_params = self.policy.get_params()
        cvar, grad_cvar = jax.value_and_grad(self.risk_evaluation_for_grad, 0)(p_params, eps_rel, significance_level, risk_threshold, num_samples_plan)
        
        p_params += p_lr * grad_cvar
        self.counter += 1
        with open(f'grad-risk-eval/iter_{self.counter}.npy', 'wb') as fh:
            np.save(fh, np.array(grad_cvar))

        self.policy.update_params(p_params)
        return 0, 0, cvar

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
            continue_sampling = (sorted_U_pi[s] - sorted_U_pi[r] >= eps_rel * (sorted_U_pi[-1] - sorted_U_pi[0]))

        floor_index = int(jnp.floor(q * L_pi))
        # var = sorted_U_pi[floor_index]
        cvar = jnp.mean(sorted_U_pi[:floor_index])
        self.total_risk_samples = total_samples
        print(total_samples)
        return cvar

    ############### BELOW FOR MC2PS #################################
    def MC2PS(self,
            batch_size=None,
            num_models=None,
            num_discounts=None,
            sigma=None,
            eps_rel=None,
            significance_level=None,
            risk_threshold=None):
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
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None
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
        
        p_params += p_lr * p_grad
        self.policy.update_params(p_params)
        
        v_alpha_quantile, _ = self.quantile_estimate(N_values, risk_threshold)
        cvar_alpha = self.CVaR_estimate(N_values, risk_threshold)
        return av_vpi, v_alpha_quantile, cvar_alpha

    def VaR_sigmoid_grad_step(self,
            num_samples_plan=None,
            k_value=0,
            p_lr=0.1,
            risk_threshold=None
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
        p_params += p_lr * p_grad
        self.policy.update_params(p_params)
        
        return av_vpi, v_alpha_quantile, cvar_alpha
