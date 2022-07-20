import jax.numpy as jnp
from src.utils import *
import numpy as np
from src.logger import CSVLogger
# from memory_profiler import profile

class MBRLLoop():
    def __init__(self, 
                env,
                agent,
                nState,
                nAction,
                initial_distribution,
                data_dir
                ):
        self.env = env
        self.nState = nState
        self.nAction = nAction
        self.initial_distribution = initial_distribution
        self.agent = agent
        self.logger = CSVLogger(
            fieldnames={
                'ep' : 0,
                'av-V-model-pi' : 0,
                'av-V-env-pi' : 0,
                'v-alpha-quantile': 0,
                'cvar-alpha' : 0,
                'cvar-constraint-lambda': 0,
                'grad-norm': 0
            },
            filename=data_dir
        )

    # @profile
    def training_loop(self, args):
        env_returns = 0
        vf_model = 0 
        grad_norm, cvar_alpha = 0, 0
        regret, bayesian_regret = 0, 0
        print(args.num_eps)
        for ep in range(args.num_eps):
            #generate data
            done = False
            step = 0
            state = self.env.reset()
            while not done and (step < args.env.traj_len):
                action = self.agent.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.update_obs(state, action, reward, next_state, done)
                state = next_state
                step += 1

            if (ep > 0): #and (ep % 5 == 0): #update after every 5 collected episodes 
                if args.train_type.type == 'MC2PS':
                    v_alpha_quantile, cvar_alpha = self.agent.MC2PS(
                        args.num_samples_plan, 
                        args.num_models, 
                        args.num_discounts, 
                        args.sigma, 
                        args.eps_rel, 
                        args.significance_level, 
                        args.risk_threshold 
                    )
                    vf_model = 0
                    env_returns = self.evaluate_agent(args.num_eps_eval, args.env.traj_len)#_exact()
                    # regret, bayesian_regret = self.regret_evaluate_agent()
                    self.logger.writerow(
                        {   
                            'av-V-model-pi' : vf_model,
                            'av-V-env-pi' : env_returns,
                            'v-alpha-quantile' : v_alpha_quantile,
                            'cvar-alpha' : cvar_alpha,
                            'cvar-constraint-lambda': self.agent.lambda_param
                        }
                    )
                    print(f'MC2PS: {ep}, Model Value fn: {vf_model:.3f}, Env rets: {env_returns:.3f}')
                    self.logger.close()
                    return

                #reset policy params here
                if args.reset_params:
                    self.agent.policy.reset_params()

                train_steps = 0

                if (args.train_type.type in ['upper-cvar', 'max-opt', 'pg', 'pg-CE', 'CVaR']):
                    grad_norm = 5.0 #just a random number to start with
                    while not np.isclose(grad_norm, 0.0, atol=5e-3):
                    # for train_step in range(args.mid_train_steps):
                        (
                            vf_model,
                            v_alpha_quantile,
                            cvar_alpha, 
                            grad_norm
                        ) = self.agent.grad_step(
                            args.train_type.type, 
                            **{
                                'num_samples_plan': args.num_samples_plan,
                                'risk_threshold': args.risk_threshold,
                                'k_value': args.k_value
                            }
                        )
                        # print(f'lambda: {self.agent.lambda_param}, cvar: {cvar_alpha}')
                        train_steps += 1
                        print(f'Train_step: {train_steps}, Model Value fn: {vf_model:.3f}, grad_norm: {grad_norm:.3f}, cvar: {cvar_alpha:.3f}')
                
                elif args.train_type.type in ['psrl']:
                    for train_step in range(args.train_type.mid_train_steps):
                        (
                            vf_model,
                            v_alpha_quantile,
                            cvar_alpha, 
                            grad_norm
                        ) = self.agent.grad_step(
                            args.train_type.type, 
                            **{
                                'num_samples_plan': args.num_samples_plan,
                                'risk_threshold': args.risk_threshold,
                                'k_value': args.k_value
                            }
                        )
                        # print(f'lambda: {self.agent.lambda_param}, cvar: {cvar_alpha}')
                        print(f'Train_step: {train_step}, Model Value fn: {vf_model:.3f}, grad_norm: {grad_norm:.3f}, cvar: {cvar_alpha:.3f}')
                elif args.train_type.type == 'psrl-opt-cvar':
                    unconstrained_train_type = 'psrl'
                    (
                        vf_model,
                        v_alpha_quantile,
                        cvar_alpha, 
                        grad_norm
                    ) = self.agent.grad_step(
                        unconstrained_train_type, 
                        **{
                            'num_samples_plan': args.num_samples_plan,
                            'risk_threshold': args.risk_threshold,
                            'k_value': args.k_value
                        }
                    )
                    self.agent.constraint = cvar_alpha
                    (
                        vf_model,
                        v_alpha_quantile,
                        cvar_alpha, 
                        grad_norm
                    ) = self.agent.grad_step(
                        args.train_type.type, 
                        **{
                            'num_samples_plan': args.num_samples_plan,
                            'risk_threshold': args.risk_threshold,
                            'k_value': args.k_value
                        }
                    )
                    # print(f'lambda: {self.agent.lambda_param}, cvar: {cvar_alpha}')
                    print(f'Ep: {ep}, Model Value fn: {vf_model:.3f}, grad_norm: {grad_norm:.3f}, cvar: {cvar_alpha:.3f}')
                else:
                    unconstrained_train_type = 'CVaR' #'upper-cvar' if args.train_type.type=='upper-cvar-opt-cvar' else 'max-opt'
                    unconstrained_cvars = []
                    self.agent.policy_lr = 100.
                    u = 0
                    grad_norm = 5.
                    while not np.isclose(grad_norm, 0.0, atol=1e-2):
                        (
                            vf_model, 
                            v_alpha_quantile, 
                            cvar_alpha, 
                            grad_norm
                        ) = self.agent.grad_step(
                                unconstrained_train_type, 
                                **{
                                    'num_samples_plan': args.num_samples_plan,
                                    'risk_threshold': args.risk_threshold,
                                    'k_value': args.k_value
                                }
                        )
                        print(f'u: {u}, lambda: {self.agent.lambda_param:.3f}, grad_norm: {grad_norm:.3f}, cvar: {cvar_alpha:.3f}' )
                        unconstrained_cvars.append(cvar_alpha)
                        u += 1
                    max_cvar = np.max(np.asarray(unconstrained_cvars)) #sum(unconstrained_cvars)/len(unconstrained_cvars)
                    self.agent.constraint = max_cvar
                    print('New Constraint:', max_cvar)
                    cvar_alphas = []
                    grad_norm = 5.
                    self.agent.policy_lr = args.train_type.policy_lr
                    halved = False
                    while not np.isclose(grad_norm, 0.0, atol=1e-2):
                    # while not np.isclose(self.agent.lambda_param, 0.0):
                        # print(f'avg: {np.mean(np.array(cvar_alphas))}')

                        if grad_norm < 1.0 and self.agent.lambda_param <= 0.5 and not halved: #len(cvar_alphas) == args.mid_train_steps:
                            #  and non_decreasing(cvar_alphas):
                            self.agent.policy_lr /= 5.
                            self.agent.lambda_lr /= 2.
                            halved = True
                            print('halving learning rates')

                        if len(cvar_alphas) > 2*args.train_type.mid_train_steps:
                            break
                        (
                            vf_model, 
                            v_alpha_quantile, 
                            cvar_alpha, 
                            grad_norm
                            ) = self.agent.grad_step(
                            args.train_type.type, 
                            **{
                                'num_samples_plan': args.num_samples_plan,
                                'risk_threshold': args.risk_threshold,
                                'k_value': args.k_value
                            }
                        )
                        cvar_alphas.append(cvar_alpha)
                        print(f'train_step: {train_steps}, lambda: {self.agent.lambda_param:.3f}, grad_norm: {grad_norm:.3f}, cvar: {cvar_alpha:.3f}' )
                        train_steps += 1

                
                if ep % args.log_freq == 0:
                    env_returns = self.evaluate_agent(args.num_eps_eval, args.env.traj_len)
                    # regret, bayesian_regret = self.regret_evaluate_agent()
                    self.logger.writerow(
                        {   
                            'ep' : ep,
                            'av-V-model-pi' : vf_model,
                            'av-V-env-pi' : env_returns,
                            'v-alpha-quantile' : 0,
                            'cvar-alpha' : cvar_alpha,
                            'cvar-constraint-lambda' : self.agent.lambda_param,
                            'grad-norm'  : grad_norm
                            }
                        )
                    print(f'Iter: {ep}, Env rets: {env_returns:.3f}')
                
                if args.reset_params:
                    self.agent.lambda_param = args.init_lambda
                    self.agent.policy_lr = args.train_type.policy_lr
                    self.agent.lambda_lr = args.train_type.lambda_lr
        self.logger.close()

    def regret_evaluate_agent(self):
        regret = 0
        bayesian_regret = 0
        return regret, bayesian_regret

    def evaluate_agent(self, num_eps_eval, traj_len):
        eval_rewards = np.zeros((num_eps_eval, 1))
        for ep in range(num_eps_eval):
            ep_rewards = []
            done = False
            step = 0
            state = self.env.reset()
            while not done and (step < traj_len): 
                action = self.agent.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                ep_rewards.append(reward)
                state = next_state
                step += 1
            eval_rewards[ep] = discount(ep_rewards, self.agent.discount)[0]

        return_avg = np.mean(eval_rewards)
        return float(return_avg)

    def evaluate_agent_exact(self):
        mdp = (self.env.R, self.env.P)
        v_pi = self.agent.policy_performance(mdp, self.agent.policy.get_params())
        # print(self.agent.value_iteration(self.env.R, self.env.P) @ self.agent.initial_distribution, v_pi)
        return v_pi

    def training_then_sample(self, args):
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
            env_returns = self.evaluate_agent(args.num_eps_eval, args.env.traj_len)#_exact()
            regret, bayesian_regret = self.regret_evaluate_agent()
            self.logger.writerow(
                {
                    'av-V-model-pi' : vf_model,
                    'av-V-env-pi' : env_returns,
                    'v-alpha-quantile' : v_alpha_quantile,
                    'cvar-constraint-lambda' : self.agent.lambda_param,
                    'cvar-alpha' : cvar_alpha,
                    'regret'     : regret,
                    'bayesian-regret' : bayesian_regret
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
                env_returns = self.evaluate_agent(args.num_eps_eval, args.env.traj_len)
                regret, bayesian_regret = self.regret_evaluate_agent()

            self.logger.writerow(
                {
                    'av-V-model-pi' : vf_model,
                    'av-V-env-pi' : env_returns,
                    'v-alpha-quantile' : v_alpha_quantile,
                    'cvar-constraint-lambda' : self.agent.lambda_param,
                    'cvar-alpha' : cvar_alpha,
                    'regret'     : regret,
                    'bayesian-regret' : bayesian_regret
                    }
                )
            print(f'Iter: {p_step}, Model Value fn: {vf_model}, Env rets: {env_returns}, cvar: {cvar_alpha}')
            
        self.logger.close()
