import sys
import os
import torch
import gym
import hydra
import optuna
import omegaconf

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances

from bbrl.utils.chrono import Chrono

from ssnb.algos.td3_optuna_classes.TD3 import TD3

assets_path = os.getcwd() + '/../../assets/'


def parseSampling(trial, paramName, paramConfig):
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


def sample_params(trial, agent):
    # cf. actor_optimizer, critic_optimizer et architecture
    config = omegaconf.create(
        {
        'render_agents': False,
        'save_best': False, 
        'logger': 
            {
            'classname': 'bbrl.utils.logger.TFLogger',
            'log_dir': './td3_logs/',
            'verbose': False,
            'every_n_seconds': 10
            }
        'algorithm': {},
        'gym_env':
            {
                'classname': '__main__.make_gym_env',
                'env_name': 'Swimmer-v3',
                'xml_file': 'swimmer3.xml'
            },
        'actor_optimizer':
            {
                'classname': torch.optim.Adam
            },
        'critic_optimizer':
            {
                'classname': torch.optim.Adam 
            } 
        }
    )

    for key, value in agent.cfg.algorithm.items():
        config.algorithm[key] = value

    for paramName, paramConfig in agent.cfg.trial.items():
        suggested_value = parseSampling(trial, paramName, paramConfig)
        config.algorithm[paramName] = suggested_value

    config.actor_optimizer['lr'] = trial.suggest_float("actor_optimizer_lr", agent.cfg.trial_actor_optimizer.min, agent.cfg.trial_actor_optimizer.max, log=True)
    config.critic_optimizer['lr'] = trial.suggest_float("critic_optimizer_lr", agent.cfg.trial_critic_optimizer.min, agent.cfg.trial_critic_optimizer.max, log=True)

    #config.algorithm.max_epochs = int(config.algorithm.n_timesteps // config.algorithm.n_steps) # to have a run of n_timesteps
    return config


def objective_agent(trial, agent):
        mean = 0
        is_pruned = False
        nan_encountered = False

        #config = sample_params(trial, agent)
        trial_agent = agent.create_agent(agent.cfg)

        try:
            for epoch in range(1):
                mean = trial_agent.run()
                trial.report(mean, epoch)
                if trial.should_prune():
                    is_pruned = True
                    break

        except AssertionError as e:
            # Sometimes, random hyperparams can generate NaN
            nan_encountered = True

        # Tell the optimizer that the trial failed
        if nan_encountered:
            return float("nan")

        if is_pruned:
            raise optuna.exceptions.TrialPruned()

        return mean



def tune(agent):
    # Création et lancement de l'étude
    sampler = TPESampler(n_startup_trials=agent.cfg.study.n_startup_trials)
    pruner = MedianPruner(n_startup_trials=agent.cfg.study.n_startup_trials, n_warmup_steps=agent.cfg.study.n_warmup_steps // 3)
    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")

    objective = lambda trial: objective_agent(trial, agent)

    try:
        study.optimize(objective, n_trials=agent.cfg.study.n_trials, n_jobs=agent.cfg.study.n_jobs, timeout=agent.cfg.study.timeout)
    except KeyboardInterrupt:
        print('\nStudy interrupted by user')
        
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
    config_name="td3_swimmer_optuna2.yaml",
)

def main(cfg):
    chrono = Chrono()
    a = TD3(cfg)
    tune(a)
    chrono.stop()

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
