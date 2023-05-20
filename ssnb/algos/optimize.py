import sys
import os
import torch
import gym
import hydra
import optuna
import time
import matplotlib.pyplot as plt
import numpy as np

from omegaconf import OmegaConf

from optuna.samplers import TPESampler
from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

from bbrl.utils.chrono import Chrono

from ssnb.algos.td3 import TD3
from ssnb.algos.ddpg import DDPG

assets_path = os.getcwd() + '/../assets/'

class Optimize:
    def __init__(self, cfg):
        self.cfg = cfg
        self.seeds = []
        agent_cfg = OmegaConf.load(cfg.agent.config)

        if cfg.agent.classname:
            exec("self.agent = " + cfg.agent.classname + "(agent_cfg)")
        
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

        elif paramName == 'actor_hidden_size':
            ahs = trial.suggest_int('actor_hidden_size', paramConfig.min, paramConfig.max)
            return {'actor_hidden_size': [ahs, ahs]}
        
        elif paramName == 'critic_hidden_size':
            chs = trial.suggest_int('critic_hidden_size', paramConfig.min, paramConfig.max)
            return {'critic_hidden_size': [chs, chs]}

        elif paramName == 'actor_optimizer_lr':
            return trial.suggest_float('actor_optimizer_lr', paramConfig.min, paramConfig.max)

        elif paramName == 'critic_optimizer_lr':
            return trial.suggest_float('critic_optimizer_lr', paramConfig.min, paramConfig.max)

        else:
            print(f'Hyperparameter {paramName} is not supported')


    def sample_params(self, trial):
        config = self.agent.cfg.copy()

        for paramName, paramConfig in self.cfg.params.items():
            #exec('suggested_value = trial.suggest_' + paramConfig.type + '("' + paramName + '", ' + paramConfig.min + ', ' + paramConfig.max + ', log=' + paramConfig.log + ')')
            #config.algorithm[paramName] = suggested_value
            suggested_value = self.parseSampling(trial, paramName, paramConfig)
            
            if paramName == 'actor_hidden_size':
                config.algorithm.architecture['actor_hidden_size'] = suggested_value['actor_hidden_size']
                
            elif paramName == 'critic_hidden_size':
                config.algorithm.architecture['critic_hidden_size'] = suggested_value['critic_hidden_size']

            elif paramName =='actor_optimizer_lr':
                config.actor_optimizer.lr = suggested_value

            elif paramName == 'critic_optimizer_lr':
                config.critic_optimizer.lr = suggested_value

            else:
                config.algorithm[paramName] = suggested_value

        return config
    

    def generate(self):
        self.seeds.append(int(time.time()))
        np.random.seed(self.seeds[0])
        
        for i in range(self.cfg.study.nb_seeds):
            random = np.random.randint(1, np.iinfo(np.int32).max)
            self.seeds.append(random)
            
    
    def objective(self, trial):
        nan_encountered = False

        config = self.sample_params(trial)
        mean_list = []

        for seed in range(1, len(self.seeds)):
            config.algorithm.seed = self.seeds[seed]
            trial_agent = self.agent.create_agent(config)
            print(f'\n=== Trial {trial.number} in progress with seed {self.seeds[seed]} ===\n')

            mean = trial_agent.run()
            
            if mean is None:
                raise KeyboardInterrupt
            
            trial.report(mean, seed)

            # Tell the optimizer that the trial failed
            if nan_encountered:
                return float("nan")

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

            mean_list.append(mean)

        return np.mean(mean_list)


    def tune(self):
        # Generate seeds
        self.generate() 
        print(f'Seeds used for this study: {self.seeds[1:]}\nGenerated with seed: {self.seeds[0]}\n')
            
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
        study.trials_dataframe().to_csv("study_results_" + self.cfg.agent.classname + "_swimmer.csv")
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
