import os
import functools
import time
from typing import Tuple
from omegaconf import OmegaConf
import torch
import gym
import bbrl
import bbrl_gym
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.distributions import Normal


from bbrl.agents.agent import Agent
from bbrl import get_arguments, get_class, instantiate_class
from bbrl.visu.visu_policies import plot_policy
from bbrl.visu.visu_critics import plot_critic

# The workspace is the main class in BBRL, this is where all data is collected and stored
from bbrl.workspace import Workspace

# Agents(agent1,agent2,agent3,...) executes the different agents the one after the other
# TemporalAgent(agent) executes an agent over multiple timesteps in the workspace, 
# or until a given condition is reached
from bbrl.agents import Agents, RemoteAgent, TemporalAgent

# AutoResetGymAgent is an agent able to execute a batch of gym environments
# with auto-resetting. These agents produce multiple variables in the workspace: 
# ’env/env_obs’, ’env/reward’, ’env/timestep’, ’env/done’, ’env/initial_state’, ’env/cumulated_reward’, 
# ... When called at timestep t=0, then the environments are automatically reset. 
# At timestep t>0, these agents will read the ’action’ variable in the workspace at time t − 1
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
# Not present in the A2C version...
from bbrl.utils.logger import TFLogger
from bbrl.utils.replay_buffer import ReplayBuffer


params={
  "save_best": True,
  "plot_agents": True,
  "logger":{
    "classname": "bbrl.utils.logger.TFLogger",
    "log_dir": "./tmp/" + str(time.time()),
    "cache_size": 10000,
    "every_n_seconds": 10,
    "verbose": False,  
    },

  "algorithm":{
    "seed": 1,
    "max_grad_norm": 0.5,
    "epsilon": 0.02,
    "n_envs": 1,
    "n_steps": 100,
    "n_updates": 100,
    "eval_interval": 2000,
    "nb_evals": 10,
    "gae": 0.8,
    "max_epochs": 21000,
    "discount_factor": 0.98,
    "buffer_size": 2e5,
    "batch_size": 64,
    "tau_target": 0.05,
    "learning_starts": 10000,
    "action_noise": 0.1,
    "architecture":{
        "actor_hidden_size": [400, 300],
        "critic_hidden_size": [400, 300],
        },
  },
  "gym_env":{
    "classname": "__main__.make_gym_env",
    "env_name": "Swimmer-v3",
  },
  "actor_optimizer":{
    "classname": "torch.optim.Adam",
    "lr": 1e-3,
  },
  "critic_optimizer":{
    "classname": "torch.optim.Adam",
    "lr": 1e-3,
  }
}


def build_mlp(sizes, activation, output_activation=nn.Identity()):
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act]
    return nn.Sequential(*layers)

def make_gym_env(env_name):
    return gym.make(env_name)

class ContinuousQAgent(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        self.is_q_function = True
        self.model = build_mlp(
            [state_dim + action_dim] + list(hidden_layers) + [1], activation=nn.ReLU()
        )

    def forward(self, t, detach_actions=False):
        obs = self.get(("env/env_obs", t))
        action = self.get(("action", t))
        if detach_actions:
            action = action.detach()
        osb_act = torch.cat((obs, action), dim=1)
        q_value = self.model(osb_act)
        self.set(("q_value", t), q_value)

    def predict_value(self, obs, action):
        osb_act = torch.cat((obs, action), dim=0)
        q_value = self.model(osb_act)
        return q_value

class ContinuousDeterministicActor(Agent):
    def __init__(self, state_dim, hidden_layers, action_dim):
        super().__init__()
        layers = [state_dim] + list(hidden_layers) + [action_dim]
        self.model = build_mlp(
            layers, activation=nn.ReLU(), output_activation=nn.Tanh()
        )

    def forward(self, t,render=None):
        obs = self.get(("env/env_obs", t))
        action = self.model(obs)
        self.set(("action", t), action)

    def predict_action(self, obs, stochastic):
        assert (
            not stochastic
        ), "ContinuousDeterministicActor cannot provide stochastic predictions"
        return self.model(obs)

class AddGaussianNoise(Agent):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        dist = Normal(act, self.sigma)
        action = dist.sample()
        self.set(("action", t), action)

class AddOUNoise(Agent):
    """
    Ornstein Uhlenbeck process noise for actions as suggested by DDPG paper
    """

    def __init__(self, std_dev, theta=0.15, dt=1e-2):
        self.theta = theta
        self.std_dev = std_dev
        self.dt = dt
        self.x_prev = 0

    def forward(self, t, **kwargs):
        act = self.get(("action", t))
        x = (
            self.x_prev
            + self.theta * (act - self.x_prev) * self.dt
            + self.std_dev * math.sqrt(self.dt) * torch.randn(act.shape)
        )
        self.x_prev = x
        self.set(("action", t), x)

def get_env_agents(cfg):
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
    return train_env_agent, eval_env_agent

# Create the DDPG Agent
def create_ddpg_agent(cfg, train_env_agent, eval_env_agent):
    obs_size, act_size = train_env_agent.get_obs_and_actions_sizes()
    critic = ContinuousQAgent(
        obs_size, cfg.algorithm.architecture.critic_hidden_size, act_size
    )
    target_critic = copy.deepcopy(critic)
    actor = ContinuousDeterministicActor(
        obs_size, cfg.algorithm.architecture.actor_hidden_size, act_size
    )
    # target_actor = copy.deepcopy(actor) # not used in practice, though described in the paper
    noise_agent = AddGaussianNoise(cfg.algorithm.action_noise) # alternative : AddOUNoise
    tr_agent = Agents(train_env_agent, actor, noise_agent)  
    ev_agent = Agents(eval_env_agent, actor)

    # Get an agent that is executed on a complete workspace
    train_agent = TemporalAgent(tr_agent)
    eval_agent = TemporalAgent(ev_agent)
    train_agent.seed(cfg.algorithm.seed)
    return train_agent, eval_agent, actor, critic, target_critic  # , target_actor

class Logger():

  def __init__(self, cfg):
    self.logger = instantiate_class(cfg.logger)

  def add_log(self, log_string, loss, epoch):
    self.logger.add_scalar(log_string, loss.item(), epoch)

  # Log losses
  def log_losses(self, cfg, epoch, critic_loss, entropy_loss, a2c_loss):
    self.add_log("critic_loss", critic_loss, epoch)
    self.add_log("entropy_loss", entropy_loss, epoch)
    self.add_log("a2c_loss", a2c_loss, epoch)

# Configure the optimizers
def setup_optimizers(cfg, actor, critic):
    actor_optimizer_args = get_arguments(cfg.actor_optimizer)
    parameters = actor.parameters()
    actor_optimizer = get_class(cfg.actor_optimizer)(parameters, **actor_optimizer_args)
    critic_optimizer_args = get_arguments(cfg.critic_optimizer)
    parameters = critic.parameters()
    critic_optimizer = get_class(cfg.critic_optimizer)(
        parameters, **critic_optimizer_args
    )
    return actor_optimizer, critic_optimizer

def compute_critic_loss(cfg, reward, must_bootstrap, q_values, target_q_values):
    # Compute temporal difference
    q_next = target_q_values
    target = (
        reward[:-1][0]
        + cfg.algorithm.discount_factor * q_next.squeeze(-1) * must_bootstrap.int()
    )
    td = target - q_values.squeeze(-1)
    # Compute critic loss
    td_error = td**2
    critic_loss = td_error.mean()
    exit()
    return critic_loss

def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

def compute_actor_loss(q_values):
    return -q_values.mean()

def run_ddpg(cfg):
    # 1)  Build the  logger
    logger = Logger(cfg)
    best_reward = -10e9

    # 2) Create the environment agent
    train_env_agent, eval_env_agent = get_env_agents(cfg)

    # 3) Create the DDPG Agent
    (
        train_agent,
        eval_agent,
        actor,
        critic,
        # target_actor,
        target_critic,
    ) = create_ddpg_agent(cfg, train_env_agent, eval_env_agent)
    ag_actor = TemporalAgent(actor)
    # ag_target_actor = TemporalAgent(target_actor)
    q_agent = TemporalAgent(critic)
    target_q_agent = TemporalAgent(target_critic)

    train_workspace = Workspace()
    rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

    # Configure the optimizer
    actor_optimizer, critic_optimizer = setup_optimizers(cfg, actor, critic)
    nb_steps = 0
    tmp_steps = 0

    # Training loop
    for epoch in range(cfg.algorithm.max_epochs):
        # Execute the agent in the workspace
        if epoch > 0:
            train_workspace.zero_grad()
            train_workspace.copy_n_last_steps(1)
            train_agent(train_workspace, t=1, n_steps=cfg.algorithm.n_steps - 1)
        else:
            train_agent(train_workspace, t=0, n_steps=cfg.algorithm.n_steps)

        transition_workspace = train_workspace.get_transitions()
        action = transition_workspace["action"]
        nb_steps += action[0].shape[0]
        rb.put(transition_workspace)

        for _ in range(cfg.algorithm.n_updates): 
            rb_workspace = rb.get_shuffled(cfg.algorithm.batch_size)

            done, truncated, reward, action = rb_workspace[
                "env/done", "env/truncated", "env/reward", "action"
            ]
            if nb_steps > cfg.algorithm.learning_starts:
                # Determines whether values of the critic should be propagated
                # True if the episode reached a time limit or if the task was not done
                # See https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj?usp=sharing
                must_bootstrap = torch.logical_or(~done[1], truncated[1])

                # Critic update
                # compute q_values: at t, we have Q(s,a) from the (s,a) in the RB
                q_agent(rb_workspace, t=0, n_steps=1)
                q_values = rb_workspace["q_value"]

                with torch.no_grad():
                    # replace the action at t+1 in the RB with \pi(s_{t+1}), to compute Q(s_{t+1}, \pi(s_{t+1}) below
                    ag_actor(rb_workspace, t=1, n_steps=1)
                    # compute q_values: at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
                    target_q_agent(rb_workspace, t=1, n_steps=1)
                  # q_agent(rb_workspace, t=1, n_steps=1)
                # finally q_values contains the above collection at t=0 and t=1
                post_q_values = rb_workspace["q_value"]

                # Compute critic loss
                critic_loss = compute_critic_loss(
                    cfg, reward, must_bootstrap, q_values[0], post_q_values[1]
                )
                logger.add_log("critic_loss", critic_loss, nb_steps)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), cfg.algorithm.max_grad_norm
                )
                critic_optimizer.step()

                # Actor update
                # Now we determine the actions the current policy would take in the states from the RB
                ag_actor(rb_workspace, t=0, n_steps=1)
                # We determine the Q values resulting from actions of the current policy
                q_agent(rb_workspace, t=0, n_steps=1)
                # and we back-propagate the corresponding loss to maximize the Q values
                q_values = rb_workspace["q_value"]
                actor_loss = compute_actor_loss(q_values)
                logger.add_log("actor_loss", actor_loss, nb_steps)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), cfg.algorithm.max_grad_norm
                )
                actor_optimizer.step()

                # Soft update of target q function
                tau = cfg.algorithm.tau_target
                soft_update_params(critic, target_critic, tau)
                # soft_update_params(actor, target_actor, tau)

        if nb_steps - tmp_steps > cfg.algorithm.eval_interval:
            tmp_steps = nb_steps
            eval_workspace = Workspace()  # Used for evaluation
            eval_agent(eval_workspace, t=0, stop_variable="env/done",render=True)
            rewards = eval_workspace["env/cumulated_reward"][-1]
            mean = rewards.mean()
            logger.add_log("reward", mean, nb_steps)
            print(f"nb_steps: {nb_steps}, reward: {mean}")
            if cfg.save_best and mean > best_reward:
                best_reward = mean
                directory = "./ddpg_agent/"
                if not os.path.exists(directory):
                    os.makedirs(directory)
                filename = directory + "ddpg_" + str(mean.item()) + ".agt"
                eval_agent.save_model(filename)
"""
                if cfg.plot_agents:
                    plot_policy(
                        actor,
                        eval_env_agent,
                        "./ddpg_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                        stochastic=False,
                    )
                    plot_critic(
                        q_agent.agent,
                        eval_env_agent,
                        "./ddpg_plots/",
                        cfg.gym_env.env_name,
                        best_reward,
                    )
"""

config=OmegaConf.create(params)
torch.manual_seed(config.algorithm.seed)
run_ddpg(config)
