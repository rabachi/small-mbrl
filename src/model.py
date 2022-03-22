import numpy as np

# Dirichlet model
class DirichletModel(object):
    def __init__(self, nState, nAction, alpha0=1., mu0=0.,
                 tau0=1., tau=1., **kwargs):
        self.nState = nState
        self.nAction = nAction
        # self.epLen = epLen
        self.alpha0 = alpha0
        self.mu0 = mu0
        self.tau0 = tau0
        self.tau = tau
        self.discount = 0.9

        self.R_prior = {}
        self.P_prior = {}
        for state in range(nState):
            for action in range(nAction):
                self.R_prior[state, action] = (self.mu0, self.tau0)
                self.P_prior[state, action] = (self.alpha0 * np.ones(self.nState, dtype=np.float32))

    def update_obs(self, oldState, action, reward, newState,
                   pContinue):
        mu0, tau0 = self.R_prior[oldState, action]
        tau1 = tau0 + self.tau
        mu1 = (mu0 * tau0 + reward * self.tau) / tau1
        self.R_prior[oldState, action] = (mu1, tau1)

        if pContinue == 1:
            self.P_prior[oldState, action][newState] += 1

    def sample_mdp(self):
        R_samp = {}
        P_samp = {}
        for s in range(self.nState):
            for a in range(self.nAction):
                mu, tau = self.R_prior[s, a]
                R_samp[s, a] = mu + np.random.normal() * \
                               1./np.sqrt(tau)
                P_samp[s, a] = np.random.dirichlet(self.P_prior[s, a])
        return R_samp, P_samp

    def policy_evaluation(self, R, P, pi):
        # Vals[state, timestep]
        p_matrix = np.zeros((self.nState, self.nAction, self.nState))
        r_matrix = np.zeros((self.nState, self.nAction))
        for s, a in P.keys():
            p_matrix[s, a, :] = P[s, a]
            r_matrix[s, a] = R[s, a]
        ppi = np.einsum('sat,sa->st', p_matrix, pi)
        rpi = np.einsum('sa,sa->s', r_matrix, pi)
        v_pi = np.linalg.solve(np.eye(self.nState) -
                               self.discount*ppi, rpi)

        return v_pi

    def value_iteration(self, R, P):
        epsilon = 1e-5
        V = np.zeros(self.nState)
        V_progress = [np.zeros(self.nState)] * 3
        delta = np.infty

        p_matrix = np.zeros((self.nState, self.nAction, self.nState))
        r_matrix = np.zeros((self.nState, self.nAction))
        for s, a in P.keys():
            p_matrix[s, a, :] = P[s, a]
            r_matrix[s, a] = R[s, a]

        i = 0
        j = 0
        while delta > epsilon:
            for s in range(self.nState):
                prev_v = V[s]
                V[s] = np.max(r_matrix[s] + np.einsum('at,t->a',
                                                p_matrix[s],
                                    self.discount*V))
                delta = np.abs(prev_v - V[s])

            if (i % 5 == 0) and (j < 3):
                V_progress[j] = V
                j += 1
            i += 1


        pi = np.zeros(self.nState)
        for s in range(self.nState):
            pi[s] = np.argmax(r_matrix[s] + np.einsum('at,t->a',
                                                  p_matrix[s],
                                                  self.discount * V))
        return V, pi #, V_progress

