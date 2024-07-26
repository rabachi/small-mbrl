import jax.numpy as jnp
from src.utils import *
import numpy as np
from src.logger import CSVLogger
import wandb
from src.utils import get_policy
import pdb

class MBRLLoop():
    def __init__(self, 
                env,
                agent,
                nState,
                nAction,
                initial_distribution,
                data_dir,
                p_params_baseline,
                baseline_true_perf,
                seed=0, 
                wandb_entity="rabachi"
                ):
        self.env = env
        self.nState = nState
        self.nAction = nAction
        self.initial_distribution = initial_distribution
        self.agent = agent
        self.use_csv = False
        self.p_params_baseline = p_params_baseline
        self.baseline_true_perf = baseline_true_perf
        # self.vf_models = [] #we need the true value functions but because we don't have them, we lower bound them either using posterior model or using confidence intervals
        # self.lb_vf_models = []
        self.executed_policy_params = []
        self.curr_budget_limit = 0.
        args_dict = {
                    'ep' : 0,
                    'av-V-model-pi' : 0,
                    'av-V-env-pi' : 0,
                    'objective': 0,
                    'cvar-alpha' : 0,
                    'cvar-constraint-lambda': 0,
                    'policy-entropy': 0,
                    'grad-norm': 0
                }

        if self.use_csv:
            self.logger = CSVLogger(
                fieldnames=args_dict,
                filename=data_dir+'_csvlog'
            )
        else:
            run = wandb.init(
                project="small_mbrl",
                entity=wandb_entity,
                config=args_dict
            )
            # Tensorboard
            # self.writer = SummaryWriter(data_dir)

        self.rng = np.random.RandomState(seed)

    def Q_learning(self, args, discount):
        qf = np.zeros((self.nState, self.nAction))
        outcomes = []
        epsilon = 1.0
        for ep in range(args.num_eps):
            state = self.env.reset()
            done = False
            while not done:
                rnd = self.rng.random()
                if rnd < epsilon:
                    action = self.rng.choice(np.arange(self.nAction))
                else:
                    action = np.argmax(qf[state])

                next_state, reward, done, _ = self.env.step(action)
                qf[state, action] = qf[state, action] + \
                                        args.train_type.alpha * (reward + discount * np.max(qf[next_state]) - qf[state, action])
                state = next_state

                if reward > 0.5:
                    outcomes.append(1)
            # Update epsilon
            epsilon = max(epsilon - args.train_type.epsilon_decay, 0)
            env_returns = self.evaluate_qf(qf, args.num_eps_eval)
            stats = {   
                        'ep' : ep,
                        'av-V-env-pi' : env_returns,
                        'successes' : sum(outcomes)
                    }
            wandb.log(stats, step=ep)
            # for key, value in stats.items():
            #     self.writer.add_scalar(key, value, ep)
            print(f'Iter: {ep}, Env rets: {env_returns:.3f}, outcomes: {outcomes}')
        return env_returns

    # @profile
    def training_loop(self, args):
        env_returns = 0
        vf_model = 0
        grad_norm, cvar_alpha = 0, 0
        objective = 0
        regret, bayesian_regret = 0, 0
        constraints = []
        cvars = []
        objectives = []
        policy_returns = []
        print(args.num_eps)

        if not args.env.terminates:
            num_eps = args.num_eps * args.env.traj_len
            step = 0
            state = self.env.reset()
            sum_rewards = 0.
        else:
            num_eps = args.num_eps
        
        env_returns = self.evaluate_agent(args.num_eps_eval, args.env.traj_len)
        stats = {   
                    'ep' : 0,
                    'av-V-model-pi' : 0,
                    'av-V-env-pi' : env_returns,
                    'objective' : 0,
                    'cvar-alpha' : 0,
                    'cvar-constraint-lambda' : self.agent.lambda_param,
                    'grad-norm'  : 0
                }
        if self.use_csv:
            self.logger.writerow(stats)
        else:
            wandb.log(stats, step=0)
            # for key, value in stats.items():
            #     self.writer.add_scalar(key, value, 0)

        print(f'Iter: {0}, Env rets: {env_returns:.3f}')
        train_iter = 0
        for ep in range(1, num_eps + 1):
            #generate data
            if args.env.terminates:
                done = False
                step = 0
                state = self.env.reset()
                while not done:
                    action = jax.lax.stop_gradient(self.agent.policy(state))
                    next_state, reward, done, _ = self.env.step(action)
                    self.agent.update_obs(state, action, reward, next_state, False)
                    # self.agent.update_obs(state, action, reward, next_state, done)
                    state = next_state
                    step += 1
                    if (step >= args.env.traj_len):
                        break
                if done:
                    while (step < args.env.traj_len):
                        action = self.agent.rng.choice(self.nAction)
                        reward = self.env.terminal_reward()
                        self.agent.update_obs(state, action, reward, state, False)
                        step += 1
            else:
                action = jax.lax.stop_gradient(self.agent.policy(state))
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
                #reset policy params here
                if args.reset_params:
                    self.agent.policy.reset_params()

                if args.train_type.type in ['max-opt-cvar', 'upper-cvar-opt-cvar', 'pg-cvar']:
                    if args.optimization_type in ['conservative']:
                        (
                            vf_model, 
                            cvar_alpha, 
                            objective, 
                            grad_norm 
                        ) = self.train_conservative_agent(args, self.p_params_baseline, self.baseline_true_perf)
                    else:
                        (
                            vf_model, 
                            cvar_alpha, 
                            objective, 
                            grad_norm,
                            constraint 
                        ) = self.train_constrained_agent(args, self.p_params_baseline, self.baseline_true_perf)
                        constraints.append(constraint)
                        cvars.append(cvar_alpha)
                        objectives.append(objective)
                        policy_perfo = self.evaluate_agent(args.num_eps_eval, args.env.traj_len)
                        policy_returns.append(policy_perfo)
                        # np.save(self.returns_dir + "/contraints.npy", constraints)
                        # np.save(self.returns_dir + "/objectives.npy", objectives)
                        # np.save(self.returns_dir + "/cvars.npy", cvars)
                        # np.save(self.returns_dir + "/policy_returns.npy", policy_returns)

                        train_iter += 1
                elif args.train_type.type in ['psrl', 'psrl-opt-cvar']: #TODO: Only PSRL implemented
                    (
                        vf_model, 
                        cvar_alpha, 
                        objective, 
                        grad_norm 
                    ) = self.train_psrl_agent(args)
                else:
                    (
                        vf_model, 
                        cvar_alpha, 
                        objective, 
                        grad_norm 
                    ) = self.train_agent(args)

                if ep % args.log_freq == 0:
                    if args.env.terminates:
                        env_returns = self.evaluate_agent(args.num_eps_eval, args.env.traj_len)
                    else:
                        env_returns = sum_rewards/ep
                    stats = {   
                            'ep' : ep,
                            'av-V-model-pi' : vf_model,
                            'av-V-env-pi' : env_returns,
                            'objective' : objective,
                            'cvar-alpha' : cvar_alpha,
                            'cvar-constraint-lambda' : self.agent.lambda_param,
                            'policy-entropy' : self.agent.policy.entropy(state), 
                            'grad-norm'  : grad_norm
                        }
                    if self.use_csv:
                        self.logger.writerow(stats)
                    else:
                        wandb.log(stats, step=ep)
                        # for key, value in stats.items():
                        #     print(key, value)
                        #     self.writer.add_scalar(key, value, ep)
                    print(f'Iter: {ep}, Env rets: {env_returns:.3f}')
                
                if args.reset_params:
                    self.agent.lambda_param = args.init_lambda
                    self.agent.policy_lr = args.train_type.policy_lr
                    self.agent.lambda_lr = args.train_type.lambda_lr
        if self.use_csv:
            self.logger.close()
        return env_returns

    def calculate_budget(self, args, p_params_baseline, baseline_true_perf, R_j, P_j):
        # recompute lower bounds for all policies in self.executed_policy_params
        cvars = np.zeros((len(self.executed_policy_params), 1))
        for idx in range(len(self.executed_policy_params)):
            if (np.equal(self.executed_policy_params[idx] - p_params_baseline, 0.)).all() and False:
                cvars[idx] = baseline_true_perf
            else:
                _, _, _, cvar_alpha, _ = self.agent.posterior_sampling(self.executed_policy_params[idx],
                                                                       args.num_samples_plan, args.risk_threshold,
                                                                       R_j=R_j, P_j=P_j)
                #cvars[idx] = max(0, cvar_alpha)  # can it be negative?
                cvars[idx] = cvar_alpha
        # sum over lower bounds
        # lb_sum = np.sum(cvars)
        # recalculate V of baseline policy
        # U_pi_baseline, _, _, _, _ = self.posterior_sampling(p_params_baseline, args.num_samples_plan, args.risk_threshold, R_j=R_j, P_j=P_j)
        budget = 0 if cvars.shape[0] == 0 else np.max(cvars) #lb_sum - (len(self.executed_policy_params) + 1) * args.alpha_baseline * baseline_true_perf
        # this should be less than curr_budget_limit - new_policy_lowerbound
        print(budget)
        return budget  # , U_pi_baseline
        
    def train_conservative_agent(self, args, p_params_baseline, baseline_true_perf):
        grad_norm = 5.
        best_params = self.agent.policy.get_params()
        R_j, P_j = self.agent.multiple_sample_mdp(args.num_samples_plan)

        # self.agent.policy.update_params(best_params) # why did I have this here ..?
        budget = 0.
        budget = self.calculate_budget(args, p_params_baseline, baseline_true_perf, R_j, P_j)
        # if budget less that curr_budget_limit - new_policy_lowerbound, play baseline, else, play UCB arm/policy
        
        # print(budget <= self.curr_budget_limit)
        # if budget <= self.curr_budget_limit: # if this is true no need to find UCB arm
        #     self.agent.policy.update_params(p_params_baseline)
        #     self.curr_budget_limit = self.curr_budget_limit + args.alpha_baseline * baseline_true_perf
        #     return baseline_true_perf, 0, baseline_true_perf, 0
        
        self.agent.policy_lr = args.train_type.policy_lr    
        num_iters = 0
        highest_objective = np.infty * -1.
        best_cvar = 0
        best_vf_model = 0
        train_type = "upper-cvar"
        print(train_type)
        for train_step in range(args.train_type.mid_train_steps):
            (
                vf_model, 
                upper_objective, 
                cvar_alpha, 
                grad_norm
            ) = self.agent.grad_step(
                train_type, 
                **{
                    'num_samples_plan': args.num_samples_plan,
                    'risk_threshold': args.risk_threshold,
                    'k_value': args.k_value,
                    'R_j': R_j,
                    'P_j': P_j
                }
            )
            if args.train_type.type in ['upper-cvar-opt-cvar', 'max-opt-cvar']:
                objective = upper_objective 
            else:
                objective = vf_model

            if objective > highest_objective:
                highest_objective = objective
                print(highest_objective)
                best_vf_model = vf_model
                best_params = self.agent.policy.get_params()
                best_cvar = cvar_alpha
            
        print(budget + best_cvar, args.alpha_baseline * baseline_true_perf, baseline_true_perf)
        if budget + best_cvar < 0:
            played_baseline = True
            print("played baseline")
            self.agent.policy.update_params(p_params_baseline)
            self.executed_policy_params.append(p_params_baseline)
        else:
            print("False")
            played_baseline = False
            self.agent.policy.update_params(best_params)
            self.executed_policy_params.append(best_params)
            
        print(f'train_step: {train_step}, budget: {budget + best_cvar}, played_baseline?: {played_baseline}, budget_limit: {self.curr_budget_limit:.3f}')
        return best_vf_model, best_cvar, highest_objective, grad_norm

    def train_constrained_agent(self, args, p_params_baseline, baseline_true_perf):
        grad_norm = 5.

        best_params = self.agent.policy.get_params()
        R_j, P_j = self.agent.multiple_sample_mdp(args.num_samples_plan)

        self.agent.policy.update_params(best_params)
        budget = self.calculate_budget(args, p_params_baseline, baseline_true_perf, R_j, P_j)
        # budget is lb_sum - baseline term, need to add newpolicylb
        if len(self.executed_policy_params) == 0:
            budget = -1000
        self.agent.constraint = budget
        self.agent.policy_lr = args.train_type.policy_lr
        tolerance = 1.0
        p_t = 0.92
        best_cvar = 0
        # if args.const_lambda:
            # args.train_type.mid_train_steps = 1
        for train_step in range(args.train_type.mid_train_steps):
            converged = False
            num_iters = 0
            highest_objective = np.infty * -1.
            best_iter = 0
            best_cvar = 0
            best_vf_model = 0
            while not converged and (num_iters < args.train_type.low_train_steps):
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
                converged = (grad_norm < 1e-5)
                num_iters += 1
                
                if args.train_type.type in ['upper-cvar-opt-cvar', 'max-opt-cvar']:
                    objective = upper_objective 
                else:
                    objective = vf_model

                if objective > highest_objective:
                    highest_objective = objective
                    best_iter = num_iters
                    best_vf_model = vf_model
                    best_params = self.agent.policy.get_params()
                    best_cvar = cvar_alpha

            self.agent.policy.update_params(best_params)
            # if not args.const_lambda:
            print(self.agent.lambda_param - (cvar_alpha - self.agent.constraint) / tolerance)
            print(self.agent.constraint)
            print(cvar_alpha)
            self.agent.lambda_param = np.clip(self.agent.lambda_param -
                                                  (cvar_alpha - self.agent.constraint) / tolerance,
                                                    a_min=0., a_max=None)
            tolerance = p_t * tolerance
            
            print(
                f'train_step: {train_step}, lambda: {self.agent.lambda_param:.3f}, grad_norm: {grad_norm:.3f},'
                f' tolerance: {tolerance:.3f} distance to constraint: {best_cvar - self.agent.constraint :.3f}')
        
        self.executed_policy_params.append(best_params) #is this part correct? should we play baseline on purpose sometimes or is that covered by the constrained opt?
        return best_vf_model, best_cvar, highest_objective, grad_norm, budget
            
    def train_psrl_agent(self, args):
        #TODO: only psrl implemented and needs testing. Constrained version removed for now due to changing constraints etc.
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
                        'R_j': R_j,
                        'P_j': P_j
                    }
            )
            # print(f'lambda: {self.agent.lambda_param}, cvar: {cvar_alpha}')
            best_iter = train_step
            print(f'Train_step: {train_step}, Model Value fn: {vf_model:.3f}, grad_norm: {grad_norm:.3f}, cvar: {cvar_alpha:.3f}')
        return vf_model, cvar_alpha, vf_model, grad_norm

    def train_agent(self, args):                    
        grad_norm = 5.0 #just a random number to start with
        # while not np.isclose(grad_norm, 0.0, atol=5e-3):
        highest_objective = np.infty * -1.
        best_vf_model = 0
        best_cvar = 0
        best_params = self.agent.policy.get_params()
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
                    'R_j' : R_j,
                    'P_j' : P_j
                }
            )
            if args.train_type.type in ['pg', 'pg-CE']:
                objective = vf_model
            elif args.train_type.type in ['upper-cvar', 'max-opt']:
                objective = upper_objective #not really var, should clean this up, this quantity takes on a different value for each of these objectives so that I can track them on the objective they are optimizing
            elif args.train_type.type in ['CVaR']:
                objective = cvar_alpha
            elif args.train_type.type in ['both-max-CVaR']:
                objective = cvar_alpha + args.init_lambda * upper_objective

            if objective > highest_objective:
                highest_objective = objective
                # print(highest_objective)
                best_cvar = cvar_alpha
                best_vf_model = vf_model
                best_params = self.agent.policy.get_params()
            # print(f'lambda: {self.agent.lambda_param}, cvar: {cvar_alpha}')
            print(f'Train_step: {train_step}, Model Value fn: {vf_model:.3f}, grad_norm: {grad_norm:.3f}, cvar: {cvar_alpha:.3f}')
        self.agent.policy.update_params(best_params)

        return best_vf_model, best_cvar, highest_objective, grad_norm

    def regret_evaluate_agent(self):
        regret = 0
        bayesian_regret = 0
        return regret, bayesian_regret

    def evaluate_qf(self, qf, num_eps_eval):
        nb_success = 0
        all_rewards = np.zeros((num_eps_eval,1))
        # Evaluation
        for ep in range(num_eps_eval):
            ep_rewards = 0
            state = self.env.reset()
            done = False
            while not done:
                action = np.argmax(qf[state])
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                ep_rewards += reward
            # ep_rewards += reward
            print(ep_rewards)
            all_rewards[ep] = ep_rewards
        return np.mean(all_rewards)

    def evaluate_agent(self, num_eps_eval, traj_len):
        eval_rewards = np.zeros((num_eps_eval, 1))
        num_goals_reached = 0
        num_fails = 0
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
                # print(next_state, reward, done)
            if done:
                if reward > self.env.goal_reward: #self.env.max_reward:
                    num_goals_reached += 1
                elif reward <= self.env.hole_reward:
                    num_fails += 1
            if done:
                while (step < traj_len):
                    action = self.agent.rng.choice(self.nAction)
                    reward = self.env.terminal_reward()
                    ep_rewards.append(reward)
                    step += 1
            eval_rewards[ep] = discount(ep_rewards, self.agent.discount)[0]
            # eval_rewards[ep] = sum(ep_rewards)
        return_avg = np.mean(eval_rewards)
        # avg_goals = num_goals_reached/num_eps_eval
        # avg_fails = num_fails / num_eps_eval
        print(f"REACHED GOAL: {num_goals_reached} times, FAILED: {num_fails} TIMES")
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
            print(f'MC2PS: {ep}, Model Value fn: {vf_model}, Env rets: {env_returns}')
            return

        for p_step in range(100):
            vf_model, v_alpha_quantile, cvar_alpha = self.agent.grad_step(self.train_type, **self.args)

            if p_step % self.log_freq == 0:
                env_returns = self.evaluate_agent(args.num_eps_eval, args.env.traj_len)
                regret, bayesian_regret = self.regret_evaluate_agent()

            print(f'Iter: {p_step}, Model Value fn: {vf_model}, Env rets: {env_returns}, cvar: {cvar_alpha}')
            