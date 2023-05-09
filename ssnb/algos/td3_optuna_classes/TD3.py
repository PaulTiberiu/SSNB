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

assets_path = os.getcwd() + '/../../assets/'


class TD3:
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

        # Create the TD3 Agent
        obs_size, act_size = self.env_agent['train_env_agent'].get_obs_and_actions_sizes()
        self.agent['actor'] = ContinuousDeterministicActor(obs_size, self.cfg.algorithm.architecture.actor_hidden_size, act_size)

        noise_agent = AddGaussianNoise(self.cfg.algorithm.action_noise)
        tr_agent = Agents(self.env_agent['train_env_agent'], self.agent['actor'], noise_agent)
        ev_agent = Agents(self.env_agent['eval_env_agent'], self.agent['actor'])

        self.agent['critic_1'] = ContinuousQAgent(obs_size, self.cfg.algorithm.architecture.critic_hidden_size, act_size)
        self.agent['target_critic_1'] = copy.deepcopy(self.agent['critic_1'])
        self.agent['critic_2'] = ContinuousQAgent(obs_size, self.cfg.algorithm.architecture.critic_hidden_size, act_size)
        self.agent['target_critic_2'] = copy.deepcopy(self.agent['critic_2'])

        self.agent['train_agent'] = TemporalAgent(tr_agent)
        self.agent['eval_agent'] = TemporalAgent(ev_agent)
        self.agent['train_agent'].seed(self.cfg.algorithm.seed)


        self.agent['ag_actor'] = TemporalAgent(self.agent['actor'])
        self.agent['q_agent_1'] = TemporalAgent(self.agent['critic_1'])
        self.agent['target_q_agent_1'] = TemporalAgent(self.agent['target_critic_1'])
        self.agent['q_agent_2'] = TemporalAgent(self.agent['critic_2'])
        self.agent['target_q_agent_2'] = TemporalAgent(self.agent['target_critic_2'])


    @classmethod
    def create_agent(cls, cfg):
        return cls(cfg)


    def setup_optimizers(self):
        actor_optimizer_args = get_arguments(self.cfg.actor_optimizer)
        parameters = self.agent['actor'].parameters()
        actor_optimizer = get_class(self.cfg.actor_optimizer)(parameters, **actor_optimizer_args)
        critic_optimizer_args = get_arguments(self.cfg.critic_optimizer)
        parameters = nn.Sequential(self.agent['critic_1'], self.agent['critic_2']).parameters()
        critic_optimizer = get_class(self.cfg.critic_optimizer)(parameters, **critic_optimizer_args)
        return actor_optimizer, critic_optimizer


    def compute_critic_loss(self, reward, must_bootstrap, q_values_1, q_values_2, q_next):
        # Compute temporal difference
        target = (reward[:-1][0] + self.cfg.algorithm.discount_factor * q_next * must_bootstrap.int())
        td_1 = target - q_values_1.squeeze(-1)
        td_2 = target - q_values_2.squeeze(-1)
        td_error_1 = td_1**2
        td_error_2 = td_2**2
        critic_loss_1 = td_error_1.mean()
        critic_loss_2 = td_error_2.mean()
        return critic_loss_1, critic_loss_2


    def compute_actor_loss(self, q_values):
        actor_loss = -q_values
        return actor_loss.mean()


    def run(self, budget):
        try:
            if self.policy_filename:
                self.agent['eval_agent'].load_model(self.policy_filename)
                self.agent['train_agent'].load_model(self.policy_filename)

            # Build the loggers
            logger = Logger(self.cfg)
            logdir = "./plot/"
            reward_logger = RewardLogger(logdir + "td3.steps", logdir + "td3.rwd")

            train_workspace = Workspace()
            rb = ReplayBuffer(max_size=self.cfg.algorithm.buffer_size)

            # Configure the optimizer
            actor_optimizer, critic_optimizer = self.setup_optimizers()
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
                    self.agent['train_agent'](train_workspace, t=1, n_steps=self.cfg.algorithm.n_steps)  # check if it should be n_steps=cfg.algorithm.n_steps - 1

                else:
                    self.agent['train_agent'](train_workspace, t=0, n_steps=self.cfg.algorithm.n_steps)

                transition_workspace = train_workspace.get_transitions()
                action = transition_workspace["action"]
                nb_steps += action[0].shape[0]

                if epoch > 0 or self.cfg.algorithm.n_steps > 1:
                    rb.put(transition_workspace)

                for _ in range(self.cfg.algorithm.n_updates):
                    if nb_steps > self.cfg.algorithm.learning_starts:
                        rb_workspace = rb.get_shuffled(self.cfg.algorithm.batch_size)
                        done, truncated, reward = rb_workspace["env/done", "env/truncated", "env/reward"]

                        # Determines whether values of the critic should be propagated
                        # True if the episode reached a time limit or if the task was not done
                        # See https://colab.research.google.com/drive/1W9Y-3fa6LsPeR6cBC1vgwBjKfgMwZvP5?usp=sharing
                        must_bootstrap = torch.logical_or(~done[1], truncated[1])

                        # Critic update
                        # compute q_values: at t, we have Q(s,a) from the (s,a) in the RB
                        self.agent['q_agent_1'](rb_workspace, t=0, n_steps=1)
                        q_values_rb_1 = rb_workspace["q_value"]
                        self.agent['q_agent_2'](rb_workspace, t=0, n_steps=1)
                        q_values_rb_2 = rb_workspace["q_value"]

                        with torch.no_grad():
                            # Replace the action at t+1 in the RB with \pi(s_{t+1}), to compute Q(s_{t+1}, \pi(s_{t+1}) below
                            self.agent['ag_actor'](rb_workspace, t=1, n_steps=1)

                            # Compute q_values: at t+1 we have Q(s_{t+1}, \pi(s_{t+1})
                            self.agent['target_q_agent_1'](rb_workspace, t=1, n_steps=1)
                            post_q_values_1 = rb_workspace["q_value"]
                            self.agent['target_q_agent_2'](rb_workspace, t=1, n_steps=1)
                            post_q_values_2 = rb_workspace["q_value"]

                        post_q_values = torch.min(post_q_values_1, post_q_values_2).squeeze(-1)
                        # Compute critic loss
                        critic_loss_1, critic_loss_2 = self.compute_critic_loss(
                            reward,
                            must_bootstrap,
                            q_values_rb_1[0],
                            q_values_rb_2[0],
                            post_q_values[1],
                        )

                        logger.add_log("critic_loss_1", critic_loss_1, nb_steps)
                        logger.add_log("critic_loss_2", critic_loss_2, nb_steps)
                        critic_loss = critic_loss_1 + critic_loss_2

                        # Critic update part
                        critic_optimizer.zero_grad()
                        critic_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.agent['critic_1'].parameters(), self.cfg.algorithm.max_grad_norm)
                        torch.nn.utils.clip_grad_norm_(self.agent['critic_2'].parameters(), self.cfg.algorithm.max_grad_norm)
                        critic_optimizer.step()

                        # Actor update
                        # Now we determine the actions the current policy would take in the states from the RB
                        self.agent['ag_actor'](rb_workspace, t=0, n_steps=1)

                        # We determine the Q values resulting from actions of the current policy
                        # We arbitrarily chose to update the actor with respect to critic_1
                        # and we back-propagate the corresponding loss to maximize the Q values
                        self.agent['q_agent_1'](rb_workspace, t=0, n_steps=1)
                        q_values_1 = rb_workspace["q_value"]
                        # self.agent['q_agent_2'](rb_workspace, t=0, n_steps=1)
                        # q_values_2 = rb_workspace["q_value"]
                        current_q_values = q_values_1.squeeze(-1)
                        actor_loss = self.compute_actor_loss(current_q_values)
                        logger.add_log("actor_loss", actor_loss, nb_steps)

                        # Actor update part
                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.agent['actor'].parameters(), self.cfg.algorithm.max_grad_norm)
                        actor_optimizer.step()

                        # Soft update of target q function
                        tau = self.cfg.algorithm.tau_target
                        soft_update_params(self.agent['critic_1'], self.agent['target_critic_1'], tau)
                        soft_update_params(self.agent['critic_2'], self.agent['target_critic_2'], tau)

                if nb_steps - tmp_steps > self.cfg.algorithm.eval_interval:
                    tmp_steps = nb_steps
                    eval_workspace = Workspace()  # Used for evaluation
                    self.agent['eval_agent'](eval_workspace, t=0, stop_variable="env/done", render=self.cfg.render_agents)
                    rewards = eval_workspace["env/cumulated_reward"]
                    self.agent['q_agent_1'](eval_workspace, t=0, stop_variable="env/done")
                    q_values = eval_workspace["q_value"].squeeze()
                    mean = rewards[-1].mean()
                    logger.add_log("reward", mean, nb_steps)
                    print(f"nb_steps: {nb_steps}, reward: {mean}")
                    reward_logger.add(nb_steps, mean)

                    if self.cfg.save_best and mean > self.best_reward:
                        self.best_reward = mean
                        directory = "./td3_agent/"

                        if not os.path.exists(directory):
                            os.makedirs(directory)

                        filename = (
                            directory
                            + self.cfg.gym_env.env_name
                            + "#td3#T1_T2#"
                            + str(mean.item())
                            + ".agt"
                        )

                        self.agent['eval_agent'].save_model(filename)
                        
                epoch += 1
            
            if not self.policy_filename:
                self.policy_filename = "./td3_agent/"  + self.cfg.gym_env.env_name + "#td3#T1_T2#" + str(mean.item()) + ".agt"

            self.agent['train_agent'].save_model(self.policy_filename)
            return mean
            
        except KeyboardInterrupt:
            print('\nAlgorithm interrupted by user before terminating')
            return None


def make_gym_env(env_name, xml_file):
    xml_file = assets_path + xml_file
    return gym.make(env_name, xml_file=xml_file)

@hydra.main(
    config_path="../configs/td3/",
    config_name="td3_swimmer3.yaml",
)

def main(cfg):
    chrono = Chrono()
    a = TD3(cfg)
    a.run(10000)
    a.run(10000)
    chrono.stop()

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
