"""P7: validation-only hyperparameter search with Optuna."""
import json
from pathlib import Path
from .layers import OPERATOR_NAMES
from .train import EXPERIMENTAL_OPERATORS
from .train import run_training

def run_study(dataset, operators=("baseline", "softmax_beta"), trials=12, epochs=8,
              seed=42, output_dir="storage/tuning", timeout=None):
    try:
        import optuna
    except ImportError as exc:
        raise ImportError("Install the lab extras: pip install -e '.[lab]'") from exc
    if trials < 1 or epochs < 1 or not operators:
        raise ValueError("Positive trials/epochs and at least one operator required")
    if any(op not in ("baseline", *OPERATOR_NAMES, *EXPERIMENTAL_OPERATORS) for op in operators):
        raise ValueError("Unknown operator")
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    sampler = optuna.samplers.TPESampler(seed=seed)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=min(5, trials), n_warmup_steps=1)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    def objective(trial):
        operator = trial.suggest_categorical("operator", list(operators))
        lr = trial.suggest_float("lr", 1e-4, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
        dim = trial.suggest_categorical("dim", [32, 64, 128])
        def callback(epoch_row):
            trial.report(epoch_row["val_macro_f1"], epoch_row["epoch"])
            if trial.should_prune():
                raise optuna.TrialPruned()
        result = run_training(dataset=dataset, operator=operator, epochs=epochs, seed=seed,
                              batch_size=batch_size, lr=lr, dim=dim,
                              output_dir=str(root / "trials"), evaluate_test=False,
                              epoch_callback=callback)
        trial.set_user_attr("run_id", result["run_id"])
        trial.set_user_attr("best_val_loss", result["best_val_loss"])
        return result["best_val_macro_f1"]
    study.optimize(objective, n_trials=trials, timeout=timeout, gc_after_trial=True)
    report = {"dataset": str(dataset), "seed": seed, "trials": trials,
              "epochs_budget": epochs, "best_params": study.best_params if study.best_trials else {},
              "best_validation_macro_f1": study.best_value if study.best_trials else None,
              "results": [{"number":t.number, "state":t.state.name, "value":t.value,
                           "params":t.params} for t in study.trials],
              "warning": "Validation-only search. A final held-out test must be run once after selection."}
    (root / "study.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
