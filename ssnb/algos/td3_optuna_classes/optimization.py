import optuna

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances


def objective_agent(trial, agent):
        mean = 0
        is_pruned = False
        nan_encountered = False
        nb_epoch_per_step = 1

        try:
            for epoch in range(agent.cfg.algorithm.max_epochs):
                mean = agent.run(nb_epoch_per_step)
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
