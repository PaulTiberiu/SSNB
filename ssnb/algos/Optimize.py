import sys
import os
import torch
import gym
import hydra
import optuna
import matplotlib.pyplot as plt

from omegaconf import OmegaConf

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

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
            return trial.suggest_float('discount_factor', paramConfig.min, paramConfig.max, log=True)

        elif paramName == 'buffer_size':
            return 10 ** trial.suggest_int('buffer_size', paramConfig.min, paramConfig.max)

        elif paramName == 'batch_size':
            return 10 ** trial.suggest_int('batch_size', paramConfig.min, paramConfig.max)

        elif paramName == 'tau_target':
            return trial.suggest_float('tau_target', paramConfig.min, paramConfig.max, log=True)

        elif paramName == 'action_noise':
            return trial.suggest_float('action_noise', paramConfig.min, paramConfig.max, log=True)

        elif paramName == 'n_steps':
            return 10 ** trial.suggest_int('n_steps', paramConfig.min, paramConfig.max)

        elif paramName == 'architecture':
            ahs = trial.suggest_int('actor_hidden_size', paramConfig.min, paramConfig.max)
            chs = trial.suggest_int('critic_hidden_size', paramConfig.min, paramConfig.max)
            return {'actor_hidden_size': [ahs, ahs], 'critic_hidden_size': [chs, chs]}

        elif paramName == 'actor_optimizer_lr':
            return trial.suggest_float('actor_optimizer_lr', paramConfig.min, paramConfig.max)

        elif paramName == 'critic_optimizer_lr':
            return trial.suggest_float('critic_optimizer_lr', paramConfig.min, paramConfig.max)

        else:
            print(f'Hyperparameter {paramName} is not supported')


    def sample_params(self, trial):
        config = self.agent.cfg.copy()

        for paramName, paramConfig in self.cfg.params.items():
            #eval('suggested_value = trial.suggest_' + paramConfig.type + '("' + paramName + '", ' + paramConfig.min + ', ' + paramConfig.max + ', log=' + paramConfig.log + ')')
            #config.algorithm[paramName] = suggested_value
            suggested_value = self.parseSampling(trial, paramName, paramConfig)
            
            if paramName == 'architecture':
                config.algorithm.architecture['actor_hidden_size'] = suggested_value['actor_hidden_size']
                config.algorithm.architecture['critic_hidden_size'] = suggested_value['critic_hidden_size']

            elif paramName =='actor_optimizer_lr':
                config.actor_optimizer.lr = suggested_value

            elif paramName == 'critic_optimizer_lr':
                config.critic_optimizer.lr = suggested_value

            else:
                config.algorithm[paramName] = suggested_value

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
        plt.show()


def make_gym_env(env_name, xml_file):
    xml_file = assets_path + xml_file
    return gym.make(env_name, xml_file=xml_file)


@hydra.main(
    config_path="./configs/",
    config_name="optimize_swimmer.yaml",
)

def main(cfg):
    chrono = Chrono()
    Optimize(cfg).tune()
    chrono.stop()

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
