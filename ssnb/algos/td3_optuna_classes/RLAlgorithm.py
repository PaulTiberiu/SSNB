import optuna

from abc import ABC, abstractmethod

from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from optuna.visualization import plot_optimization_history, plot_param_importances

class RLAlgorithm(ABC):
    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def create_agent(self):
        pass

    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def objective(self, trial):
        pass


    def tune(self):
        # Création et lancement de l'étude
        sampler = TPESampler(n_startup_trials=self.cfg.study.n_startup_trials)
        pruner = MedianPruner(n_startup_trials=self.cfg.study.n_startup_trials, n_warmup_steps=self.cfg.study.n_warmup_steps // 3)
        study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
        
        try:
            study.optimize(self.objective, n_trials=self.cfg.study.n_trials, n_jobs=self.cfg.study.n_jobs, timeout=self.cfg.study.timeout)
        except KeyboardInterrupt:
            pass
        
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")

        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        print("  User attrs:")

        for key, value in trial.user_attrs.items():
            print(f"    {key}: {value}")

        # Write report
        study.trials_dataframe().to_csv("study_results_td3_swimmer.csv")

        fig1 = plot_optimization_history(study)
        fig2 = plot_param_importances(study)

        fig1.show()
        fig2.show()