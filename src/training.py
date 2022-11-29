import jax.numpy as jnp
from src.utils import *
import numpy as np
from src.logger import CSVLogger
# from memory_profiler import profile
from torch.utils.tensorboard import SummaryWriter  # type: ignore
from torchinfo import summary

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
        self.use_csv = False
        if self.use_csv:
            self.logger = CSVLogger(
                fieldnames={
                    'ep' : 0,
                    'av-V-model-pi' : 0,
                    'av-V-env-pi' : 0,
                    'v-alpha-quantile': 0,
                    'cvar-alpha' : 0,
                    'cvar-constraint-lambda': 0,
                    'grad-norm': 0,
                    'best_iter': 0,
                    'samples_taken': 0
                },
                filename=data_dir+'_csvlog'
            )
        else:
            # Tensorboard
            self.writer = SummaryWriter(data_dir)

    # @profile
    def training_loop(self, args):
        env_returns = 0
        vf_model = 0 
        grad_norm, cvar_alpha = 0, 0
        regret, bayesian_regret = 0, 0
        print(args.num_eps)

        if not args.env.terminates:
            num_eps = args.num_eps * args.env.traj_len
            step = 0
            state = self.env.reset()
            sum_rewards = 0.
        else:
            num_eps = args.num_eps

        for ep in range(num_eps):
            #generate data
            if args.env.terminates:
                done = False
                step = 0
                state = self.env.reset()
                while not done:
                    action = self.agent.policy(state)
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.update_obs(state, action, reward, next_state, False)
                    state = next_state
                    step += 1
                    if (step >= args.env.traj_len):
                        break
                if done:
                    while (step < args.env.traj_len):
                        action = self.agent.rng.choice(self.nAction)
                        reward = 0
                        self.agent.update_obs(state, action, reward, state, False)
                        step += 1
            else:
                action = self.agent.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.update_obs(state, action, reward, next_state, done)
                state = next_state
                sum_rewards += reward
                step += 1

            if (
                args.env.terminates and 
                (ep > args.env.start_train_ep) and 
                (ep % args.env.train_freq_ep == 0)
                ) or (
                    not args.env.terminates and
                    (step > args.env.start_train_step)
                    and
                    (step % args.env.train_freq_step == 0)
            ):
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
                    # self.logger.writerow(
                    #     {   
                    #         'av-V-model-pi' : vf_model,
                    #         'av-V-env-pi' : env_returns,
                    #         'v-alpha-quantile' : v_alpha_quantile,
                    #         'cvar-alpha' : cvar_alpha,
                    #         'cvar-constraint-lambda': self.agent.lambda_param
                    #     }
                    # )
                    print(f'MC2PS: {ep}, Model Value fn: {vf_model:.3f}, Env rets: {env_returns:.3f}')
                    # self.logger.close()
                    return

                #reset policy params here
                if args.reset_params:
                    self.agent.policy.reset_params()

                train_steps = 0
                if (args.train_type.type in ['upper-cvar', 'max-opt', 'pg', 'pg-CE', 'CVaR']):
                    grad_norm = 5.0 #just a random number to start with
                    # while not np.isclose(grad_norm, 0.0, atol=5e-3):
                    highest_objective = np.infty * -1.
                    best_iter = 0
                    best_params = self.agent.policy.get_params()
                    R_j, P_j = self.agent.multiple_sample_mdp(args.num_samples_plan)

                    for train_step in range(args.train_type.mid_train_steps):
                        (
                            vf_model,
                            v_alpha_quantile,
                            cvar_alpha, 
                            grad_norm, 
                            samples_taken
                        ) = self.agent.grad_step(
                            args.train_type.type, 
                            **{
                                'num_samples_plan': args.num_samples_plan,
                                'risk_threshold': args.risk_threshold,
                                'k_value': args.k_value,
                                'R_j' : R_j,
                                'P_j' : P_j
                            }
                        )
                        if args.train_type.type in ['pg', 'pg-CE']:
                            objective = vf_model
                        elif args.train_type.type in ['upper-cvar', 'max-opt']:
                            objective = v_alpha_quantile #not really var, should clean this up, this quantity takes on a different value for each of these objectives so that I can track them on the objective they are optimizing
                        elif args.train_type.type in ['CVaR']:
                            objective = cvar_alpha
                        if objective > highest_objective:
                            highest_objective = objective
                            print(highest_objective)
                            best_iter = train_step
                            best_params = self.agent.policy.get_params()
                        # print(f'lambda: {self.agent.lambda_param}, cvar: {cvar_alpha}')
                        train_steps += 1
                        print(f'Train_step: {train_steps}, Model Value fn: {vf_model:.3f}, grad_norm: {grad_norm:.3f}, cvar: {cvar_alpha:.3f}')
                    self.agent.policy.update_params(best_params)
                elif args.train_type.type in ['psrl']:
                    R_j, P_j = self.agent.multiple_sample_mdp(args.num_samples_plan)
                    for train_step in range(args.train_type.mid_train_steps):
                        (
                            vf_model, 
                            v_alpha_quantile, 
                            cvar_alpha, 
                            grad_norm,
                            samples_taken
                        ) = self.agent.grad_step(
                                args.train_type.type, 
                                **{
                                    'num_samples_plan': args.num_samples_plan,
                                    'risk_threshold': args.risk_threshold,
                                    'k_value': args.k_value,
                                    'R_j': R_j,
                                    'P_j': P_j
                                }
                        )
                        # print(f'lambda: {self.agent.lambda_param}, cvar: {cvar_alpha}')
                        best_iter = train_step
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
                elif args.train_type.type == 'both-max-CVaR':
                    u = 0
                    grad_norm = 5.
                    highest_objective = np.infty * -1.
                    best_iter = 0
                    best_params = self.agent.policy.get_params()
                    # self.agent.policy_lr = 1.0
                    R_j, P_j = self.agent.multiple_sample_mdp(args.num_samples_plan)
                    for train_step in range(args.train_type.mid_train_steps):
                        (
                            vf_model, 
                            upper_objective, 
                            cvar_alpha, 
                            grad_norm
                        ) = self.agent.grad_step(
                                args.train_type.type, 
                                **{
                                    'num_samples_plan': args.num_samples_plan,
                                    'risk_threshold': args.risk_threshold,
                                    'k_value': args.k_value,
                                    'R_j': R_j,
                                    'P_j': P_j
                                }
                        )
                        objective = cvar_alpha + args.init_lambda * upper_objective
                        if objective > highest_objective:
                            highest_objective = objective
                            print(highest_objective)
                            best_iter = train_step
                            best_params = self.agent.policy.get_params()
                        print(f'Train_step: {train_step}, Model Value fn: {vf_model:.3f}, grad_norm: {grad_norm:.3f}, cvar: {cvar_alpha:.3f}, objective: {objective:.3f}')
                    self.agent.policy.update_params(best_params)
                    samples_taken = args.num_samples_plan
                else:
                    unconstrained_train_type = 'CVaR' #'upper-cvar' if args.train_type.type=='upper-cvar-opt-cvar' else 'max-opt'
                    u = 0
                    grad_norm = 5.
                    if ep % 2 == 0:
                        constraint_target = np.infty * -1.
                    # highest_objective = np.infty * -1.
                    # best_iter = 0
                    # best_params = self.agent.policy.get_params()
                    # self.agent.policy_lr = 1.0
                    # R_j, P_j = self.agent.multiple_sample_mdp(args.num_samples_plan)
                    # for train_step in range(args.train_type.pretrain_steps):
                    #     (
                    #         vf_model, 
                    #         v_alpha_quantile, 
                    #         cvar_alpha, 
                    #         grad_norm,
                    #         samples_taken
                    #     ) = self.agent.grad_step(
                    #             unconstrained_train_type, 
                    #             **{
                    #                 'num_samples_plan': args.num_samples_plan,
                    #                 'risk_threshold': args.risk_threshold,
                    #                 'k_value': args.k_value,
                    #                 'R_j': R_j,
                    #                 'P_j': P_j
                    #             }
                    #     )
                    #     objective = cvar_alpha
                    #     if objective > highest_objective:
                    #         highest_objective = objective
                    #         print(highest_objective)
                    #         best_iter = train_step
                    #         best_params = self.agent.policy.get_params()

                    #     print(f'u: {u}, lambda: {self.agent.lambda_param:.3f}, grad_norm: {grad_norm:.3f}, cvar: {cvar_alpha:.3f}' )
                    #     u += 1
                    # pre_train_best_iter = best_iter
                    # self.agent.policy.reset_params()
                    self.agent.policy.update_params(best_params)
                    self.agent.constraint = constraint_target
                    # self.agent.lambda_param = args.init_lambda
                    print('New Constraint:', constraint_target)
                    cvar_alphas = []
                    self.agent.policy_lr = args.train_type.policy_lr
                    tolerance = 1.0
                    p_t = 0.99
                    best_cvar = 0
                    for train_step in range(args.train_type.mid_train_steps):
                        converged = False
                        num_iters = 0
                        highest_objective = np.infty * -1.
                        best_iter = 0
                        best_cvar = 0
                        while not converged and (num_iters < 200):
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
                                    'k_value': args.k_value,
                                    'R_j': R_j,
                                    'P_j': P_j
                                }
                            )
                            converged = (grad_norm < 1e-5)
                            num_iters += 1
                            print(num_iters, vf_model)
                            
                            if args.train_type.type in ['upper-cvar-opt-cvar', 'max-opt-cvar']:
                                objective = v_alpha_quantile #not really var, should clean this up, this quantity takes on a different value for each of these objectives so that I can track them on the objective they are optimizing
                            else:
                                objective = vf_model

                            if objective > highest_objective:
                                highest_objective = objective
                                print(highest_objective)
                                best_iter = num_iters
                                best_params = self.agent.policy.get_params()
                                best_cvar = cvar_alpha

                        self.agent.policy.update_params(best_params)
                        self.agent.lambda_param = np.clip(self.agent.lambda_param - (best_cvar - self.agent.constraint)/tolerance, a_min=0., a_max=None) #BUG: cvar_alpha is not the one corresponding to best_params!!!!!!, should recalculate cvar_alpha or save the one from the best iters: fixed I think 
                        tolerance = p_t * tolerance
                        
                        print(f'train_step: {train_steps}, lambda: {self.agent.lambda_param:.3f}, grad_norm: {grad_norm:.3f}, tolerance: {tolerance:.3f} distance to constraint: {best_cvar - self.agent.constraint :.3f}')
                        train_steps += 1
                    # best_iter = pre_train_best_iter

                if ep % args.log_freq == 0:
                    if args.env.terminates:
                        env_returns = self.evaluate_agent(args.num_eps_eval, args.env.traj_len)
                    else:
                        env_returns = sum_rewards/ep
                    # regret, bayesian_regret = self.regret_evaluate_agent()
                    stats = {   
                            'ep' : ep,
                            'av-V-model-pi' : vf_model,
                            'av-V-env-pi' : env_returns,
                            'v-alpha-quantile' : 0,
                            'cvar-alpha' : cvar_alpha,
                            'cvar-constraint-lambda' : self.agent.lambda_param,
                            'grad-norm'  : grad_norm,
                            'best_iter': best_iter,
                            'samples_taken' : samples_taken
                        }
                    if self.use_csv:
                        self.logger.writerow(stats)
                    else:
                        for key, value in stats.items():
                            self.writer.add_scalar(key, value, ep)
                    print(f'Iter: {ep}, Env rets: {env_returns:.3f}')
                
                if args.reset_params:
                    self.agent.lambda_param = args.init_lambda
                    self.agent.policy_lr = args.train_type.policy_lr
                    self.agent.lambda_lr = args.train_type.lambda_lr
        if self.use_csv:
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
            # self.logger.writerow(
            #     {
            #         'av-V-model-pi' : vf_model,
            #         'av-V-env-pi' : env_returns,
            #         'v-alpha-quantile' : v_alpha_quantile,
            #         'cvar-constraint-lambda' : self.agent.lambda_param,
            #         'cvar-alpha' : cvar_alpha,
            #         'regret'     : regret,
            #         'bayesian-regret' : bayesian_regret
            #         }
            #     )
            print(f'MC2PS: {ep}, Model Value fn: {vf_model}, Env rets: {env_returns}')
            # self.logger.close()
            return

        for p_step in range(100):
            # R, P = self.agent.get_CE_model()
            vf_model, v_alpha_quantile, cvar_alpha = self.agent.grad_step(self.train_type, **self.args)

            # print(self.agent.policy.get_params())
            if p_step % self.log_freq == 0:
                env_returns = self.evaluate_agent(args.num_eps_eval, args.env.traj_len)
                regret, bayesian_regret = self.regret_evaluate_agent()

            # self.logger.writerow(
            #     {
            #         'av-V-model-pi' : vf_model,
            #         'av-V-env-pi' : env_returns,
            #         'v-alpha-quantile' : v_alpha_quantile,
            #         'cvar-constraint-lambda' : self.agent.lambda_param,
            #         'cvar-alpha' : cvar_alpha,
            #         'regret'     : regret,
            #         'bayesian-regret' : bayesian_regret
            #         }
            #     )
            print(f'Iter: {p_step}, Model Value fn: {vf_model}, Env rets: {env_returns}, cvar: {cvar_alpha}')
            
        # self.logger.close()
