import sys
import os
import torch
import gym
import hydra
import optuna

from omegaconf import OmegaConf

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances

from bbrl.utils.chrono import Chrono

from ssnb.algos.td3_optuna_classes.TD3 import TD3

assets_path = os.getcwd() + '/../../assets/'


class Optimize:
    def __init__(self, cfg):
        self.cfg = cfg
        agent_cfg = OmegaConf.load(cfg.agent.config)

        if cfg.agent.classname == 'ssnb.algos.TD3.TD3':
            self.agent = TD3(agent_cfg)

        else:
            self.agent = None    
    
    def parseSampling(self, trial, paramName, paramConfig):
        if paramName == 'discount_factor':
            # discount factor between 0.9 and 0.9999
            return trial.suggest_float('discount_factor', paramConfig.min, paramConfig.max, log=True)

        elif paramName == 'n_steps':
            # n_steps 128, 256, 512, ...
            return 2 ** trial.suggest_int('n_steps', paramConfig.min, paramConfig.max)

        elif paramName == 'buffer_size':
            # buffer_size between 1e5 and 1e6
            return trial.suggest_int('buffer_size', paramConfig.min, paramConfig.max)

        elif paramName == 'batch_size':
            # batch_size between 100 and 300
            return trial.suggest_int("batch_size", paramConfig.min, paramConfig.max)

        elif paramName == 'tau_target':
            # tau_target between 0.05 and 0.005
            return trial.suggest_float("tau_target", paramConfig.min, paramConfig.max, log=True)

        elif paramName == 'action_noise':
            # action_noise between 0 and 0.1
            return trial.suggest_float("action_noise", paramConfig.min, paramConfig.max, log=True)

        elif paramName == 'architecture':
            # actor hidden size between [32, 32] and [256, 256]
            ahs = 2 ** trial.suggest_int("actor_hidden_size", paramConfig.min, paramConfig.max)
            chs = 2 ** trial.suggest_int("critic_hidden_size", paramConfig.min, paramConfig.max)
            return {'actor_hidden_size': [ahs, ahs], 'critic_hidden_size': [chs, chs]}

        else:
            print(f'Hyperparameter {paramName} is not supported')


    def sample_params(self, trial):
        # cf. actor_optimizer, critic_optimizer et architecture
        config = self.agent.cfg.copy()

        for paramName, paramConfig in self.cfg.params.items():
            #eval('suggested_value = trial.suggest_' + paramConfig.type + '("' + paramName + '", ' + paramConfig.min + ', ' + paramConfig.max + ', log=' + paramConfig.log + ')')
            #config.algorithm[paramName] = suggested_value
            suggested_value = self.parseSampling(trial, paramName, paramConfig)
            config.algorithm[paramName] = suggested_value

        #config.algorithm.max_epochs = int(config.algorithm.n_timesteps // config.algorithm.n_steps) # to have a run of n_timesteps
        return config


    def objective(self, trial):
        mean = []
        is_pruned = False
        nan_encountered = False

        config = self.sample_params(trial)
        trial_agent = self.agent.create_agent(config)

        for session in range(self.cfg.study.n_sessions):
            mean_session = trial_agent.run(self.cfg.study.n_steps_per_trial // self.cfg.study.n_sessions)

            if mean_session is None:
                raise KeyboardInterrupt

            mean.append(mean_session)
            trial.report(mean_session, session)
            if trial.should_prune():
                is_pruned = True
                break

        # Tell the optimizer that the trial failed
        if nan_encountered:
            return float("nan")

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        if len(mean) == 0:
            return None

        return max(mean)


    def tune(self):
        study = optuna.create_study(direction="maximize")

        try:
            study.optimize(self.objective, n_trials=self.cfg.study.n_trials)

        except KeyboardInterrupt:
            pass

        print(f'Number of finished trials: {len(study.trials)}\n')

        trial = study.best_trial
        print(f'=== Best trial ===\nValue: {trial.value}\nParams: ')
        for key, value in trial.params.items():
            print(f"{key}: {value}\t")

        print('\nUser attributes: ')
        for key, value in trial.user_attrs.items():
            print(f"{key}: {value}\t")

        # Write report
        study.trials_dataframe().to_csv("study_results_td3_swimmer.csv")
        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)
        fig1.show()
        fig2.show()


def make_gym_env(env_name, xml_file):
    xml_file = assets_path + xml_file
    return gym.make(env_name, xml_file=xml_file)


@hydra.main(
    config_path="../configs/td3/",
    config_name="optimize_swimmer3.yaml",
)

def main(cfg):
    chrono = Chrono()
    Optimize(cfg).tune()
    chrono.stop()

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
