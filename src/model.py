import numpy as np
import jax.numpy as jnp
import jax
from itertools import product
from typing import List, Tuple, Dict
from scipy.stats import dirichlet
from scipy.special import comb
from src.policy import Policy #is this gonna give rise to circular dependency?
from src.agent import Agent
# SEED = 0

# Dirichlet model
class DirichletModel(Agent):
    def __init__(self, nState, nAction, seed, discount, initial_distribution, init_lambda, lambda_lr, policy_lr, use_incorrect_priors, alpha0=1., mu0=0., tau0=1., tau=1.):
        self.nState = nState
        self.nAction = nAction
        # self.epLen = epLen
        self.use_incorrect_priors = use_incorrect_priors
        self.rng = np.random.RandomState(seed)
        if use_incorrect_priors:
            self.alpha0 = self.rng.beta(2, 5, size=(self.nState))
            self.mu0 = -1.
            self.tau0 = 0.01
            self.R_prior = {}
            self.P_prior = {}
            for state in range(nState):
                for action in range(nAction):
                    self.R_prior[state, action] = (self.mu0, self.tau0)
                    self.P_prior[state, action] = self.alpha0
        else:
            self.alpha0 = alpha0
            self.mu0 = mu0
            self.tau0 = tau0
            self.R_prior = {}
            self.P_prior = {}
            for state in range(nState):
                for action in range(nAction):
                    self.R_prior[state, action] = (self.mu0, self.tau0)
                    self.P_prior[state, action] = (self.alpha0 * np.ones(self.nState, dtype=np.float32))

        self.tau = tau
        self.discount = discount
        self.initial_distribution = initial_distribution
        self.constraint = -1. #placeholder value, set dynamically later
        temp = 1.0
        self.policy = Policy(nState, nAction, temp, seed, p_params=None)
        super().__init__(self.nState, self.discount, self.initial_distribution, self.policy, init_lambda, lambda_lr, policy_lr)
        self.CE_model = (np.zeros((self.nState, self.nAction)), np.zeros((self.nState, self.nAction, self.nState)))
        self.f_best = - np.infty

    def update_obs(self, oldState, action, reward, newState, done):
        mu0, tau0 = self.R_prior[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior[oldState, action] = (mu1, tau1)
        # print(pContinue)
        if not done:
            self.P_prior[oldState, action][newState] += 1

    def get_matrix_PR(self, R: Dict, P: Dict) -> Tuple[np.ndarray]:
        p_matrix = np.zeros((self.nState, self.nAction, self.nState))
        r_matrix = np.zeros((self.nState, self.nAction))
        for s, a in P.keys():
            p_matrix[s, a, :] = P[s, a]
            r_matrix[s, a] = R[s, a]
        return r_matrix, p_matrix

    def get_matrix_priors(self, R: Dict, P: Dict) -> Tuple[np.ndarray]:
        alphas = np.zeros((self.nState, self.nAction, self.nState))
        mus = np.zeros((self.nState, self.nAction))
        taus = np.zeros((self.nState, self.nAction))

        for s, a in P.keys():
            alphas[s, a, :] = self.P_prior[s, a]
            mus[s, a] = self.R_prior[s, a][0]
            taus[s, a] = self.R_prior[s, a][1]
        return mus, taus, alphas

    def sample_mdp(self):
        R_samp = {}
        P_samp = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                mu, tau = self.R_prior[s, a]
                R_samp[s, a] = mu + self.rng.normal() * \
                               1./np.sqrt(tau)
                P_samp[s, a] = self.rng.dirichlet(self.P_prior[s, a])
        
        R_samp_, P_samp_ = self.get_matrix_PR(R_samp, P_samp)
        return R_samp_, P_samp_

    def multiple_sample_mdp(self, num_samples):
        R_samp = np.zeros((num_samples, self.nState, self.nAction))
        P_samp = np.zeros((num_samples, self.nState, self.nAction, self.nState))
        mus, taus, alphas = self.get_matrix_priors(self.R_prior, self.P_prior)
        for s in range(self.nState):
            for a in range(self.nAction):
                R_samp[:, s, a] = mus[s,a] + self.rng.normal(size=num_samples) * 1./np.sqrt(taus[s,a])
                P_samp[:, s,a] = self.rng.dirichlet(alphas[s,a,:], size=num_samples)
        return R_samp, P_samp
    # def update_CE_model(self, R_samp: np.ndarray, P_samp: np.ndarray, num_samp: int):
    #     curr_R, curr_P = self.CE_model
    #     next_R = curr_R + (R_samp - curr_R)/num_samp
    #     next_P = curr_P + (P_samp - curr_P)/num_samp
    #     self.CE_model = (next_R, next_P)

    def get_CE_model(self):
        p_matrix = np.zeros((self.nState, self.nAction, self.nState))
        mean_p_matrix = np.zeros((self.nState, self.nAction, self.nState))
        r_matrix = np.zeros((self.nState, self.nAction))
        for s, a in self.P_prior.keys():
            p_matrix[s, a, :] = self.P_prior[s, a]
            r_matrix[s, a] = self.R_prior[s, a][0]
            mean_p_matrix[s,a] = dirichlet.mean(p_matrix[s,a])
        p_matrix /= np.sum(p_matrix, axis=2, keepdims=True)
        # print(np.isclose(p_matrix, mean_p_matrix))
        return r_matrix, mean_p_matrix
