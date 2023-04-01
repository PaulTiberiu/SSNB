import sys
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import gym
import hydra
import optuna
from omegaconf import DictConfig

from bbrl.utils.chrono import Chrono
from bbrl import get_arguments, get_class
from bbrl.workspace import Workspace
from bbrl.agents import Agents, TemporalAgent
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.workspace import Workspace

from ssnb.models.actors import ContinuousDeterministicActor
from ssnb.models.critics import ContinuousQAgent
from ssnb.models.exploration_agents import AddGaussianNoise
from ssnb.models.loggers import Logger, RewardLogger
from ssnb.models.shared_models import soft_update_params

# HYDRA_FULL_ERROR = 1
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances

import matplotlib
matplotlib.use("TkAgg")
assets_path = os.getcwd() + "/../assets/"

# Create the TD3 Agent
def create_td3_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    actor = ContinuousDeterministicActor(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    # target_actor = copy.deepcopy(actor)
    noise_agent = AddGaussianNoise(cfg.algorithm.action_noise)
    tr_agent = Agents(train_env_agent, actor, noise_agent)
    ev_agent = Agents(eval_env_agent, actor)
    critic_1 = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
    )
    target_critic_1 = copy.deepcopy(critic_1)
    critic_2 = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
    )
    target_critic_2 = copy.deepcopy(critic_2)
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)
    return (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    )


def make_gym_env(env_name, xml_file):
    xml_file = assets_path + xml_file
    return gym.make(env_name, xml_file=xml_file)

# Configure the optimizer
def setup_optimizers(cfg, actor, critic_1, critic_2):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = nn.Sequential(critic_1, critic_2).parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer


def compute_critic_loss(cfg, reward, must_bootstrap, q_values_1, q_values_2, q_next):
    # Compute temporal difference
    target = (
        reward[:-1][0] + cfg.algorithm.discount_factor * q_next * must_bootstrap.int()
    )
    td_1 = target - q_values_1.squeeze(-1)
    td_2 = target - q_values_2.squeeze(-1)
    td_error_1 = td_1**2
    td_error_2 = td_2**2
    critic_loss_1 = td_error_1.mean()
    critic_loss_2 = td_error_2.mean()
    return critic_loss_1, critic_loss_2


def compute_actor_loss(q_values):
    actor_loss = -q_values
    return actor_loss.mean()


def run_td3(cfg, reward_logger):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9
    delta_list = []
    # 2) Create the environment agents
    train_env_agent = AutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.n_envs,
        cfg.algorithm.seed,
    )
    eval_env_agent = NoAutoResetGymAgent(
        get_class(cfg.gym_env),
        get_arguments(cfg.gym_env),
        cfg.algorithm.nb_evals,
        cfg.algorithm.seed,
    )
    # 3) Create the TD3 Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic_1,
        target_critic_1,
        critic_2,
        target_critic_2,
    ) = create_td3_agent(cfg, train_env_agent, eval_env_agent)
    ag_actor = TemporalAgent(actor)
    # ag_target_actor = TemporalAgent(target_actor)
    q_agent_1 = TemporalAgent(critic_1)
    target_q_agent_1 = TemporalAgent(target_critic_1)
    q_agent_2 = TemporalAgent(critic_2)
    target_q_agent_2 = TemporalAgent(target_critic_2)
    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)
    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic_1, critic_2)
    nb_steps = 0
    tmp_steps = 0
    is_pruned = False
    # Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps)
        else:
            train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        if epoch > 0 or cfg.algorithm.n_steps > 1:
            rb.put(transition_workspace)
            # rb.print_obs()
            
        for _ in range(cfg.algorithm.n_updates):
            # print(f"done {done}, reward {reward}, action {action}")
            if nb_steps > cfg.algorithm.learning_starts:
                rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)
                done, truncated, reward = rb_workspace[
                    "env/done", "env/truncated", "env/reward"
                ]
                # Determines whether values of the critic should be propagated
                # True if the episode reached a time limit or if the task was not done
                # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
                must_bootstrap = torch.logical_or(~done[1], truncated[1])
                # Critic update
                # compute q_values: at t, we have Q(s,a) from the (s,a) in the RB
                q_agent_1(rb_workspace, t=0, n_steps=1)
                q_values_rb_1 = rb_workspace["q_value"]
                q_agent_2(rb_workspace, t=0, n_steps=1)
                q_values_rb_2 = rb_workspace["q_value"]
                with torch.no_grad():
                    # replace the action at t+1 in the RB with \pi(s_{t+1}), to compute Q(s_{t+1}, \pi(s_{t+1}) below
                    ag_actor(rb_workspace, t=1, n_steps=1)
                    # compute q_values: at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
                    target_q_agent_1(rb_workspace, t=1, n_steps=1)
                    post_q_values_1 = rb_workspace["q_value"]
                    target_q_agent_2(rb_workspace, t=1, n_steps=1)
                    post_q_values_2 = rb_workspace["q_value"]
                post_q_values = torch.min(post_q_values_1, post_q_values_2).squeeze(-1)
                # Compute critic loss
                critic_loss_1, critic_loss_2 = compute_critic_loss(
                    cfg,
                    reward,
                    must_bootstrap,
                    q_values_rb_1[0],
                    q_values_rb_2[0],
                    post_q_values[1],
                )
                logger.add_log("critic_loss_1", critic_loss_1, nb_steps)
                logger.add_log("critic_loss_2", critic_loss_2, nb_steps)
                critic_loss = critic_loss_1 + critic_loss_2
                # Actor update
                # Now we determine the actions the current policy would take in the states from the RB
                ag_actor(rb_workspace, t=0, n_steps=1)
                # We determine the Q values resulting from actions of the current policy
                # We arbitrarily chose to update the actor with respect to critic_1
                # and we back-propagate the corresponding loss to maximize the Q values
                q_agent_1(rb_workspace, t=0, n_steps=1)
                q_values_1 = rb_workspace["q_value"]
                q_agent_2(rb_workspace, t=0, n_steps=1)
                q_values_2 = rb_workspace["q_value"]
                current_q_values = torch.min(q_values_1, q_values_2).squeeze(-1)
                actor_loss = compute_actor_loss(current_q_values)
                logger.add_log("actor_loss", actor_loss, nb_steps)
                # Actor update part ###################################################################
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), cfg.algorithm.max_grad_norm
                )
                actor_optimizer.step()
                # Critic update part ############################################################
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    critic_1.parameters(), cfg.algorithm.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    critic_2.parameters(), cfg.algorithm.max_grad_norm
                )
                critic_optimizer.step()
                # Soft update of target q function
                tau = cfg.algorithm.tau_target
                soft_update_params(critic_1, target_critic_1, tau)
                soft_update_params(critic_2, target_critic_2, tau)
                # soft_update_params(actor, target_actor, tau)
        
        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(eval_workspace, t=0, stop_variable="env/done", render=cfg.render_agents) 
            rewards = eval_workspace["env/cumulated_reward"]
            q_agent_1(eval_workspace, t=0, stop_variable="env/done")
            q_values = eval_workspace["q_value"].squeeze()
            delta = q_values - rewards
            maxi_delta = delta.max(axis=0)[0].detach().numpy()
            delta_list.append(maxi_delta)
            mean = rewards[-1].mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"nb_steps: {nb_steps}, reward: {mean}")
            reward_logger.add(nb_steps, mean)
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = "./td3_agent/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = (
                    directory
                    + cfg.gym_env.env_name
                    + "#td3#T1_T2#"
                    + str(mean.item())
                    + ".agt"
                )
                eval_agent.save_model(filename)

            trial.report(mean, epoch)
            if trial.should_prune():
                is_pruned = True
                return _, _, mean, is_pruned            
           
                
    delta_list_mean = np.array(delta_list).mean(axis=1)
    delta_list_std = np.array(delta_list).std(axis=1)
    return delta_list_mean, delta_list_std, mean, is_pruned

"""
@hydra.main(
    config_path="./configs/td3/",
    config_name="td3_swimmer3.yaml",
    # config_name="td3_cartpolecontinuous.yaml",
    # config_name="td3_lunar_lander_continuous.yaml",
    # config_name="td3_pendulum.yaml",
)
"""

cfg = OmegaConf.load("./configs/td3_swimmer_values.yaml")

def sample_td3_params(trial):
    """Sampler for TD3 hyperparameters."""
    
    params = cfg.copy()
    
    # discount factor between 0.9 and 0.9999
    params.algorithm.discount_factor = trial.suggest_float("discount_factor", cfg.algorithm.discount_factor.min, cfg.algorithm.discount_factor.max, log=True)
    
    # n_steps 128, 256, 512, ...
    n_steps = 2 ** trial.suggest_int("n_steps", cfg.algorithm.n_steps.min, cfg.algorithm.n_steps.max)
    
    # buffer_size between 1e5 and 1e6
    params.algorithm.buffer_size = trial.suggest_int("buffer_size", cfg.algorithm.buffer_size.min, cfg.algorithm.buffer_size.max)
    
    # batch_size between 100 and 300
    params.algorithm.batch_size = trial.suggest_int("batch_size", cfg.algorithm.batch_size.min, cfg.algorithm.batch_size.max)
    
    # tau_target between 0.05 and 0.005
    params.algorithm.tau_target = trial.suggest_float("tau_target", cfg.algorithm.tau_target.min, cfg.algorithm.tau_target.max, log=True)
    
    # action_noise between 0 and 0.1
    params.algorithm.action_noise = trial.suggest_float("action_std", cfg.algorithm.action_noise.min, cfg.algorithm.action_noise.max, log=True)
    
    # actor hidden size between [32, 32] and [256, 256]
    ahs = 2 ** trial.suggest_int("actor_hidden_size", cfg.algorithm.architecture.actor_hidden_size.min, cfg.algorithm.architecture.actor_hidden_size.max)
    params.algorithm.architecture.actor_hidden_size = [ahs, ahs]
    
    # critic hidden size between [32, 32] and [256, 256]
    chs = 2 ** trial.suggest_int("critic_hidden_size", cfg.algorithm.architecture.critic_hidden_size.min, cfg.algorithm.architecture.critic_hidden_size.max)
    params.algorithm.architecture.critic_hidden_size = [chs, chs]
    
    # actor learning rate between 1e-5 and 1
    params.actor_optimizer.lr = trial.suggest_float("actor_lr", cfg.actor_optimizer.lr.min, cfg.actor_optimizer.lr.max, log=True)
    # critic learning rate between 1e-5 and 1
    params.critic_optimizer.lr = trial.suggest_float("critic_lr", cfg.critic_optimizer.lr.min, cfg.critic_optimizer.lr.max, log=True)

    params.algorithm.n_steps = n_steps
    params.algorithm.max_epochs = int(params.algorithm.n_timesteps // n_steps) # to have a run of n_timesteps
    
    return params
    
def td3_objective(trial):
    mean = 0
    is_pruned = False
    
    cfg = sample_td3_params(trial)
 
    nan_encountered = False
    
    try:
        # Train the model
        logdir = "./plot/"
        reward_logger = RewardLogger(logdir + "td3.steps", logdir + "td3.rwd")
        torch.manual_seed(cfg.algorithm.seed)
        _, _, mean, is_pruned = run_td3(cfg, reward_logger, trial)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN
        print(e)
        nan_encountered = True

    # Tell the optimizer that the trial failed
    if nan_encountered:
        return float("nan")
    
    if is_pruned:
        raise optuna.exceptions.TrialPruned()
    
    return mean

def tune(objective):
	# Création et lancement de l'étude
	sampler = TPESampler(n_startup_trials=cfg.study.n_startup_trials)
	
	pruner = MedianPruner(n_startup_trials=cfg.study.n_startup_trials, n_warmup_steps=cfg.study.n_warmup_steps // 3)
    
	study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
	
	try:
		study.optimize(objective, n_trials=cfg.study.n_trials, n_jobs=cfg.study.n_jobs, timeout=cfg.study.timeout)
	except KeyboardInterrupt:
		pass
	
	print("Number of finished trials: ", len(study.trials))

	print("Best trial:")
	trial = study.best_trial

	print(f"  Value: {trial.value}")

	print("  Params: ")
	for key, value in trial.params.items():
		print(f"    {key}: {value}")
	
	print("  User attrs:")
	for key, value in trial.user_attrs.items():
		print(f"    {key}: {value}")

	# Write report
	study.trials_dataframe().to_csv("study_results_td3_swimmer.csv")

	fig1 = plot_optimization_history(study)
	fig2 = plot_param_importances(study)

	fig1.show()
	fig2.show()

def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    tune(td3_objective)
    # main_loop(cfg)


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main(cfg)
