import hydra
import optuna

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances


def sample_params(trial):
    """Sampler for hyperparameters."""

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


def objective_agent(trial, agent):
        mean = 0
        is_pruned = False
        nan_encountered = False
        nb_epoch_per_step = 1

        config = sample_params(trial)
        trial_agent = agent.create_agent(config)

        try:
            for epoch in range(trial_agent.cfg.algorithm.max_epochs):
                mean = trial_agent.run(nb_epoch_per_step)
                trial.report(mean, epoch)
                if trial.should_prune():
                    is_pruned = True
                    break

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


@hydra.main(
    config_path="../configs/td3/",
    config_name="td3_swimmer_optuna.yaml",
)

def main(cfg):
    chrono = Chrono()
    a = TD3(cfg)
    tune(a)
    chrono.stop()

if __name__ == "__main__":
    sys.path.append(os.getcwd())
    main()
