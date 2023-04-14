import sys
import os
import copy
import numpy as np
import torch
import torch.nn as nn
import gym
import hydra

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

class TD3_agent:
    def __init__(self, cfg):
        self.logger = Logger(cfg)
        self.best_reward = -10e9
        self.delta_list = []
        # 2) Create the environment agents
        self.train_env_agent = AutoResetGymAgent(
            get_class(cfg.gym_env),
            get_arguments(cfg.gym_env),
            cfg.algorithm.n_envs,
            cfg.algorithm.seed,
        )
        self.eval_env_agent = NoAutoResetGymAgent(
            get_class(cfg.gym_env),
            get_arguments(cfg.gym_env),
            cfg.algorithm.nb_evals,
            cfg.algorithm.seed,
        )
        # 3) Create the TD3 Agent
        (
            self.train_agent,
            self.eval_agent,
            self.actor,
            self.critic_1,
            self.target_critic_1,
            self.critic_2,
            self.target_critic_2,
        ) = create_td3_agent(cfg, self.train_env_agent, self.eval_env_agent)
        self.ag_actor = TemporalAgent(self.actor)
        # ag_target_actor = TemporalAgent(target_actor)
        self.q_agent_1 = TemporalAgent(self.critic_1)
        self.target_q_agent_1 = TemporalAgent(self.target_critic_1)
        self.q_agent_2 = TemporalAgent(self.critic_2)
        self.target_q_agent_2 = TemporalAgent(self.target_critic_2)
        self.train_workspace = Workspace()
        self.rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)
        # Configure the optimizer
        self.actor_optimizer, self.critic_optimizer = setup_optimizers(cfg, self.actor, self.critic_1, self.critic_2)
        self.nb_steps = 0
        self.tmp_steps = 0
        self.is_eval = bool
        self.last_mean_reward = float

def run_td3(cfg, reward_logger, agent):
    # 1)  Build the  logger
    if not agent :
        agent = TD3_agent(cfg)
    agent.is_eval = False
    # Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            agent.train_workspace.zero_grad()
            agent.train_workspace.copy_n_last_steps(1)
            agent.train_agent(
                agent.train_workspace, t=1, n_steps=cfg.algorithm.n_steps
            )  # check if it should be n_steps=cfg.algorithm.n_steps - 1

            # below, was n_steps=cfg.algorithm.n_steps - 1
            agent.train_agent(agent.train_workspace, t=1, n_steps=cfg.algorithm.n_steps)
        else:
            agent.train_agent(agent.train_workspace, t=0, n_steps=cfg.algorithm.n_steps)

        transition_workspace = agent.train_workspace.get_transitions()
        action = transition_workspace["action"]
        agent.nb_steps += action[0].shape[0]
        if epoch > 0 or cfg.algorithm.n_steps > 1:
            agent.rb.put(transition_workspace)
            # rb.print_obs()
        for _ in range(cfg.algorithm.n_updates):
            # print(f"done {done}, reward {reward}, action {action}")
            if agent.nb_steps > cfg.algorithm.learning_starts:
                rb_workspace = agent.rb.get_shuffled(cfg.algorithm.batch_size)
                done, truncated, reward = rb_workspace[
                    "env/done", "env/truncated", "env/reward"
                ]
                # Determines whether values of the critic should be propagated
                # True if the episode reached a time limit or if the task was not done
                # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
                must_bootstrap = torch.logical_or(~done[1], truncated[1])
                # Critic update
                # compute q_values: at t, we have Q(s,a) from the (s,a) in the RB
                agent.q_agent_1(rb_workspace, t=0, n_steps=1)
                q_values_rb_1 = rb_workspace["q_value"]
                agent.q_agent_2(rb_workspace, t=0, n_steps=1)
                q_values_rb_2 = rb_workspace["q_value"]
                with torch.no_grad():
                    # replace the action at t+1 in the RB with \pi(s_{t+1}), to compute Q(s_{t+1}, \pi(s_{t+1}) below
                    agent.ag_actor(rb_workspace, t=1, n_steps=1)
                    # compute q_values: at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
                    agent.target_q_agent_1(rb_workspace, t=1, n_steps=1)
                    post_q_values_1 = rb_workspace["q_value"]
                    agent.target_q_agent_2(rb_workspace, t=1, n_steps=1)
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
                agent.logger.add_log("critic_loss_1", critic_loss_1, agent.nb_steps)
                agent.logger.add_log("critic_loss_2", critic_loss_2, agent.nb_steps)
                critic_loss = critic_loss_1 + critic_loss_2
                # Actor update
                # Now we determine the actions the current policy would take in the states from the RB
                agent.ag_actor(rb_workspace, t=0, n_steps=1)
                # We determine the Q values resulting from actions of the current policy
                # We arbitrarily chose to update the actor with respect to critic_1
                # and we back-propagate the corresponding loss to maximize the Q values
                agent.q_agent_1(rb_workspace, t=0, n_steps=1)
                q_values_1 = rb_workspace["q_value"]
                agent.q_agent_2(rb_workspace, t=0, n_steps=1)
                q_values_2 = rb_workspace["q_value"]
                current_q_values = torch.min(q_values_1, q_values_2).squeeze(-1)
                actor_loss = compute_actor_loss(current_q_values)
                agent.logger.add_log("actor_loss", actor_loss, agent.nb_steps)
                # Actor update part ###################################################################
                agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.actor.parameters(), cfg.algorithm.max_grad_norm
                )
                agent.actor_optimizer.step()
                # Critic update part ############################################################
                agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    agent.critic_1.parameters(), cfg.algorithm.max_grad_norm
                )
                torch.nn.utils.clip_grad_norm_(
                    agent.critic_2.parameters(), cfg.algorithm.max_grad_norm
                )
                agent.critic_optimizer.step()
                # Soft update of target q function
                tau = cfg.algorithm.tau_target
                soft_update_params(agent.critic_1, agent.target_critic_1, tau)
                soft_update_params(agent.critic_2, agent.target_critic_2, tau)
                # soft_update_params(actor, target_actor, tau)

        if agent.nb_steps - agent.tmp_steps > cfg.algorithm.eval_interval:
            agent.tmp_steps = agent.nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            agent.eval_agent(eval_workspace, t=0, stop_variable="env/done", render=cfg.render_agents) #????????
            rewards = eval_workspace["env/cumulated_reward"]
            agent.q_agent_1(eval_workspace, t=0, stop_variable="env/done")
            q_values = eval_workspace["q_value"].squeeze()
            delta = q_values - rewards
            maxi_delta = delta.max(axis=0)[0].detach().numpy()
            agent.delta_list.append(maxi_delta)
            mean = rewards[-1].mean()
            agent.logger.add_log("reward", mean, agent.nb_steps)
            print(f"nb_steps: {agent.nb_steps}, reward: {mean}")
            reward_logger.add(agent.nb_steps, mean)
            agent.is_eval = True
            agent.last_mean_reward = mean
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
                agent.eval_agent.save_model(filename)

    delta_list_mean = np.array(agent.delta_list).mean(axis=1)
    delta_list_std = np.array(agent.delta_list).std(axis=1)
    return delta_list_mean, delta_list_std


@hydra.main(
    config_path="./configs/td3/",
    config_name="td3_swimmer3.yaml",
    # config_name="td3_cartpolecontinuous.yaml",
    # config_name="td3_lunar_lander_continuous.yaml",
    # config_name="td3_pendulum.yaml",
)


def main(cfg: DictConfig):
    # print(OmegaConf.to_yaml(cfg))
    chrono = Chrono()
    logdir = "./plot/"
    reward_logger = RewardLogger(logdir + "td3.steps", logdir + "td3.rwd")
    torch.manual_seed(cfg.algorithm.seed)
    run_td3(cfg, reward_logger)
    #study = optuna.create_study()
    #study.optimize(objective, n_trials=100)
    chrono.stop()
    # main_loop(cfg)
if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()