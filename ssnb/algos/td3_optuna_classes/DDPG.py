import copy
import gym
import hydra
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
import torch.nn as nn

from omegaconf import DictConfig

from bbrl import get_arguments, get_class
from bbrl.agents import Agents, TemporalAgent
from bbrl.agents.gymb import AutoResetGymAgent, NoAutoResetGymAgent
from bbrl.utils.chrono import Chrono
from bbrl.utils.replay_buffer import ReplayBuffer
from bbrl.workspace import Workspace

from ssnb.models.actors import ContinuousDeterministicActor
from ssnb.models.critics import ContinuousQAgent
from ssnb.models.exploration_agents import AddGaussianNoise
from ssnb.models.loggers import Logger, RewardLogger
from ssnb.models.plotters import Plotter

assets_path = os.getcwd() + '/../../assets/'

class DDPG:
    def __init__(self, cfg):
        torch.manual_seed(cfg.algorithm.seed)
        self.cfg = cfg
        self.agent = {}
        self.env_agent = {}
        self.best_reward = -10e9
        self.policy_filename = None

        # Create the environment agents
        self.env_agent['train_env_agent'] = AutoResetGymAgent(
            get_class(self.cfg.gym_env),
            get_arguments(self.cfg.gym_env),
            self.cfg.algorithm.n_envs,
            self.cfg.algorithm.seed,
        )

        self.env_agent['eval_env_agent'] = NoAutoResetGymAgent(
            get_class(self.cfg.gym_env),
            get_arguments(self.cfg.gym_env),
            self.cfg.algorithm.nb_evals,
            self.cfg.algorithm.seed,
        )

        # Create the DDPG Agent
        obs_size, act_size = self.env_agent['train_env_agent'].get_obs_and_actions_sizes()
        self.agent['actor'] = ContinuousDeterministicActor(obs_size, self.cfg.algorithm.architecture.actor_hidden_size, act_size)

        noise_agent = AddGaussianNoise(self.cfg.algorithm.action_noise)
        tr_agent = Agents(self.env_agent['train_env_agent'], self.agent['actor'], noise_agent)
        ev_agent = Agents(self.env_agent['eval_env_agent'], self.agent['actor'])

        self.agent['critic'] = ContinuousQAgent(obs_size, self.cfg.algorithm.architecture.critic_hidden_size, act_size)
        self.agent['target_critic'] = copy.deepcopy(self.agent['critic'])

        self.agent['train_agent'] = TemporalAgent(tr_agent)
        self.agent['eval_agent'] = TemporalAgent(ev_agent)
        self.agent['train_agent'].seed(self.cfg.algorithm.seed)

        self.agent['ag_actor'] = TemporalAgent(self.agent['actor'])
        self.agent['q_agent'] = TemporalAgent(self.agent['critic'])
        self.agent['target_q_agent'] = TemporalAgent(self.agent['target_critic'])


    @classmethod
    def create_agent(cls, cfg):
        return cls(cfg)


    def soft_update_params(self, net, target_net, tau):
        for param, target_param in zip(net.parameters(), target_net.parameters()):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


    # Configure the optimizer
    def setup_optimizers(cfg, actor, critic):
        actor_optimizer_args = get_arguments(self.cfg.actor_optimizer)
        parameters = self.agent['actor'].parameters()
        actor_optimizer = get_class(self.cfg.actor_optimizer)(parameters, **actor_optimizer_args)
        critic_optimizer_args = get_arguments(self.cfg.critic_optimizer)
        parameters = self.agent['critic'].parameters()
        critic_optimizer = get_class(self.cfg.critic_optimizer)(parameters, **critic_optimizer_args)
        return actor_optimizer, critic_optimizer


    def compute_actor_loss(self, q_values):
        return -q_values.mean()


    def compute_critic_loss(self, reward, must_bootstrap, q_values, target_q_values):
        # Compute temporal difference
        q_next = target_q_values
        target = (reward[:-1].squeeze() + self.cfg.algorithm.discount_factor * q_next.squeeze(-1) * must_bootstrap.int())
        mse = nn.MSELoss()
        critic_loss = mse(target, q_values.squeeze(-1))
        return critic_loss


    def run(self, budget):
        try:
            if self.policy_filename:
                self.agent['eval_agent'].load_model(self.policy_filename)
                self.agent['train_agent'].load_model(self.policy_filename)
            
            # Build the logger
            logger = Logger(cfg)
            logdir = "./plot/"
            reward_logger = RewardLogger(logdir + "ddpg.steps", logdir + "ddpg.rwd")

            train_workspace = Workspace()
            rb = ReplayBuffer(max_size=cfg.algorithm.buffer_size)

            # Configure the optimizer
            actor_optimizer, critic_optimizer = self.setup_optimizers(cfg)
            nb_steps = 0
            tmp_steps = 0

            # Training loop
            for epoch in range(self.cfg.algorithm.max_epochs):
                # Check the remaining training budget
                if nb_steps >= budget:
                    break

                # Execute the agent in the workspace
                if epoch > 0:
                    train_workspace.zero_grad()
                    train_workspace.copy_n_last_steps(1)
                    self.agent['train_agent'](train_workspace, t=1, n_steps=self.cfg.algorithm.n_steps - 1)

                else:
                    self.agent['train_agent'](train_workspace, t=0, n_steps=self.cfg.algorithm.n_steps)

                transition_workspace = train_workspace.get_transitions()
                action = transition_workspace["action"]
                nb_steps += action[0].shape[0]
                rb.put(transition_workspace)

                for _ in range(self.cfg.algorithm.n_updates):
                    rb_workspace = rb.get_shuffled(self.cfg.algorithm.batch_size)
                    done, truncated, reward, action = rb_workspace["env/done", "env/truncated", "env/reward", "action"]

                    if nb_steps > self.cfg.algorithm.learning_starts:
                        """
                        Determines whether values of the critic should be propagated
                        True if the episode reached a time limit or if the task was not done
                        See https://colab.research.google.com/drive/1erLbRKvdkdDy0Zn1X_JhC01s1QAt4BBj?usp=sharing
                        """
                        must_bootstrap = torch.logical_or(~done[1], truncated[1])

                        """
                        Critic update
                        compute q_values: at t, we have Q(s,a) from the (s,a) in the RB
                        the detach_actions=True changes nothing in the results
                        """
                        self.agent['q_agent'](rb_workspace, t=0, n_steps=1, detach_actions=True)
                        q_values = rb_workspace["q_value"]

                        with torch.no_grad():
                            # replace the action at t+1 in the RB with \pi(s_{t+1}), to compute Q(s_{t+1}, \pi(s_{t+1}) below
                            ag_actor(rb_workspace, t=1, n_steps=1)

                            # compute q_values: at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
                            self.agent['target_q_agent'](rb_workspace, t=1, n_steps=1, detach_actions=True)

                        # finally q_values contains the above collection at t=0 and t=1
                        post_q_values = rb_workspace["q_value"]

                        # Compute critic loss
                        critic_loss = self.compute_critic_loss(self.cfg, reward, must_bootstrap, q_values[0], post_q_values[1])
                        logger.add_log("critic_loss", critic_loss, nb_steps)
                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.agent['critic'].parameters(), self.cfg.algorithm.max_grad_norm)
                        critic_optimizer.step()

                        # Actor update : now we determine the actions the current policy would take in the states from the RB
                        ag_actor(rb_workspace, t=0, n_steps=1)

                        # We determine the Q values resulting from actions of the current policy
                        self.agent['q_agent'](rb_workspace, t=0, n_steps=1)

                        # And we back-propagate the corresponding loss to maximize the Q values
                        q_values = rb_workspace["q_value"]
                        actor_loss = self.compute_actor_loss(q_values)
                        logger.add_log("actor_loss", actor_loss, nb_steps)

                        # if -25 < actor_loss < 0 and nb_steps > 2e5:
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.agent['actor'].parameters(), self.cfg.algorithm.max_grad_norm)
                        actor_optimizer.step()

                        # Soft update of target q function
                        tau = self.cfg.algorithm.tau_target
                        self.soft_update_params(self.agent['critic'], self.agent['target_critic'], tau)

                if nb_steps - tmp_steps > self.cfg.algorithm.eval_interval:
                    tmp_steps = nb_steps
                    eval_workspace = Workspace()  # Used for evaluation
                    self.agent['eval_agent'](eval_workspace, t=0, stop_variable="env/done", render=self.cfg.render_agents) # Used for render
                    rewards = eval_workspace["env/cumulated_reward"]
                    self.agent['q_agent'](eval_workspace, t=0, stop_variable="env/done")
                    q_values = eval_workspace["q_value"].squeeze()
                    mean = rewards[-1].mean()
                    logger.add_log("reward", mean, nb_steps)
                    print(f"nb_steps: {nb_steps}, reward: {mean}")
                    reward_logger.add(nb_steps, mean)

                    if self.cfg.save_best and mean > self.best_reward:
                        self.best_reward = mean
                        directory = "./ddpg_agent/"

                        if not os.path.exists(directory):
                            os.makedirs(directory)
                        
                        filename = (
                            directory
                            + self.cfg.gym_env.env_name
                            + "#ddpg#T1_T2#"
                            + str(mean.item())
                            + ".agt"
                        )

                        self.agent['eval_agent'].save_model(filename)

            if not self.policy_filename:
                self.policy_filename = "./ddpg_agent/"  + self.cfg.gym_env.env_name + "#ddpg#T1_T2#" + str(mean.item()) + ".agt"

            self.agent['train_agent'].save_model(self.policy_filename)
            return mean
        
        except KeyboardInterrupt:
            print('\nAlgorithm interrupted by user before terminating')
            return None


def make_gym_env(env_name, xml_file):
    xml_file = assets_path + xml_file
    return gym.make(env_name, xml_file=xml_file)


@hydra.main(
    config_path="../configs/ddpg/",
    config_name="ddpg_swimmer3.yaml",
)


def main(cfg):
    chrono = Chrono()
    a = DDPG(cfg)
    a.run(250000)
    chrono.stop()


if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()