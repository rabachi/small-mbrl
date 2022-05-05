import numpy as np
import jax.numpy as jnp
from src.utils import *
from src.logger import CSVLogger

class MBRLLoop():
    def __init__(self, 
                env,
                agent,
                nState,
                nAction,
                initial_distribution,
                model_lr,
                policy_lr,
                num_samples_plan,
                num_eps,
                train_type,
                save_sub_dir,
                seed,
                traj_len=10,
                risk_threshold=0.1,
                k_value=1,
                log_freq=5,
                num_eps_eval=5,
                batch_size=100, 
                num_models=5, 
                num_discounts=9, 
                sigma='CVaR',
                eps_rel=0.1, 
                significance_level=0.1 
                ):
        self.env = env
        self.nState = nState
        self.nAction = nAction
        self.initial_distribution = initial_distribution
        self.agent = agent
        self.num_samples_plan = num_samples_plan
        self.traj_len = traj_len #either an int or infty
        self.num_eps = num_eps
        self.model_lr = model_lr
        self.policy_lr = policy_lr
        self.risk_threshold = risk_threshold
        self.logger = CSVLogger(
            fieldnames={
                'av-V-model-pi' : 0,
                'av-V-env-pi' : 0,
                'v-alpha-quantile': 0,
                'cvar-alpha' : 0
            },
            filename=save_sub_dir
        )
        self.train_type = train_type
        self.k_value = k_value
        self.num_eps_eval = num_eps_eval
        self.log_freq = log_freq
        self.seed = seed
        self.args = {
            'num_samples_plan': self.num_samples_plan,
            # 'num_train_iters': self.num_train_iters,
            'risk_threshold' : self.risk_threshold,
            'p_lr' : self.policy_lr,
            'k_value' : self.k_value
        }
        self.mc2ps_args = {
            'batch_size' : batch_size, 
            'num_models' : num_models, 
            'num_discounts' : num_discounts, 
            'sigma' : sigma, 
            'eps_rel' : eps_rel, 
            'significance_level' : significance_level, 
            'risk_threshold' : self.risk_threshold 
        }

    def training_loop(self):
        for ep in range(self.num_eps):
            #generate data
            done = False
            step = 0
            state = self.env.reset()
            while not done and (step < self.traj_len): 
                action = self.agent.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.update_obs(state, action, reward, next_state, done)
                state = next_state
                step += 1

            if ep >= 1:
                # R, P = self.agent.get_CE_model()
                #FIX THIS
                vf_model = self.agent.grad_step(self.train_type, **self.args)

                # print(self.agent.policy.get_params())
                if ep % self.log_freq == 0:
                    env_returns = self.evaluate_agent_exact()

                #FIX THIS
                self.logger.writerow(
                    {
                        'av-V-model-pi' : vf_model,
                        'av-V-env-pi' : env_returns,
                        }
                    )
                print(f'Iter: {ep}, Model Value fn: {vf_model}, Env rets: {env_returns}')
        
        self.logger.close()

    def evaluate_agent(self):
        eval_rewards = np.zeros((self.num_eps_eval, 1))
        for ep in range(self.num_eps_eval):
            ep_rewards = []
            done = False
            step = 0
            state = self.env.reset()
            while not done and (step < self.traj_len): 
                action = self.agent.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                ep_rewards.append(reward)
                state = next_state
                step += 1
            eval_rewards[ep] = discount(ep_rewards, self.agent.discount)[0]

        return_avg = np.mean(eval_rewards)
        return return_avg

    def evaluate_agent_exact(self):
        mdp = (self.env.R, self.env.P)
        v_pi = self.agent.policy_performance(mdp, self.agent.policy.get_params())
        # print(self.agent.value_iteration(self.env.R, self.env.P) @ self.agent.initial_distribution, v_pi)
        return v_pi

    def training_then_sample(self):
        for ep in range(self.num_eps):
            #generate data
            done = False
            step = 0
            state = self.env.reset()
            while not done and (step < self.traj_len): 
                action = self.agent.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.update_obs(state, action, reward, next_state, done)
                state = next_state
                step += 1
        
        if self.train_type == 'MC2PS':
            v_alpha_quantile, cvar_alpha = self.agent.MC2PS(**self.mc2ps_args)
            vf_model = 0
            env_returns = self.evaluate_agent()#_exact()
            self.logger.writerow(
                {
                    'av-V-model-pi' : vf_model,
                    'av-V-env-pi' : env_returns,
                    'v-alpha-quantile' : v_alpha_quantile,
                    'cvar-alpha' : cvar_alpha
                    }
                )
            print(f'MC2PS: {ep}, Model Value fn: {vf_model}, Env rets: {env_returns}')
            self.logger.close()
            return

        for p_step in range(100):
            # R, P = self.agent.get_CE_model()
            vf_model, v_alpha_quantile, cvar_alpha = self.agent.grad_step(self.train_type, **self.args)

            # print(self.agent.policy.get_params())
            if p_step % self.log_freq == 0:
                env_returns = self.evaluate_agent_exact()

            self.logger.writerow(
                {
                    'av-V-model-pi' : vf_model,
                    'av-V-env-pi' : env_returns,
                    'v-alpha-quantile' : v_alpha_quantile,
                    'cvar-alpha' : cvar_alpha
                    }
                )
            print(f'Iter: {p_step}, Model Value fn: {vf_model}, Env rets: {env_returns}, cvar: {cvar_alpha}')
            
        self.logger.close()
