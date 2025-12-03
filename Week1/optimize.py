import optuna
import numpy as np
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from main import cross_validate, test_final
from utils import load_dataset
import json, datetime, os


def objective(trial):
    codebook_size = trial.suggest_int("codebook_size", 20, 200, step=10)
    detector_type = trial.suggest_categorical("detector_type", ["AKAZE", "SIFT", "ORB"])
    dense_sift = trial.suggest_categorical("dense_sift", [False, True]) if detector_type == "SIFT" else False
    sift_step = trial.suggest_int("sift_step", 5, 20, step=5) if dense_sift else 10
    sift_scales = trial.suggest_int("sift_scales", 1, 3) if dense_sift else 1
    use_scaler = trial.suggest_categorical("use_scaler", [False, True])
    
    classifier_type = trial.suggest_categorical("classifier_type", ["logreg", "svm_linear", "svm_rbf"])
    
    bovw_params = {
        "detector_type": detector_type,
        "codebook_size": codebook_size,
        "dense_sift": dense_sift,
        "sift_step": sift_step,
        "sift_scales": sift_scales,
        "use_scaler": use_scaler
    }
    
    classifier_params = {}
    if classifier_type == "svm_linear":
        classifier_params["C"] = trial.suggest_float("C", 0.1, 100, log=True)
    elif classifier_type == "svm_rbf":
        classifier_params["C"] = trial.suggest_float("C", 0.1, 100, log=True)
        classifier_params["gamma"] = trial.suggest_float("gamma", 0.001, 1, log=True)
    
    try:
        mean_acc, _ = cross_validate(train_data, bovw_params, classifier_params, 
                                     n_splits=3, classifier_type=classifier_type)
        return mean_acc
    except Exception as e:
        print(f"Trial failed: {e}")
        return 0.0

if __name__ == "__main__":
    train_data = load_dataset("../data/MIT_split/train")
    test_data = load_dataset("../data/MIT_split/test")
    
    print(f"Train samples: {len(train_data)}, Test samples: {len(test_data)}\n")
    
    # Configure a TPESampler: seeded, allow multivariate modeling and start with several random trials for robustness
    sampler = TPESampler(seed=42, n_startup_trials=10, multivariate=True)
    pruner = MedianPruner()
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(objective, n_trials=50, show_progress_bar=True)
    
    print("\n=== Best Trial ===")
    print(f"Best accuracy: {study.best_value:.4f}")
    print(f"Best params: {study.best_params}\n")
    
    # Persist best trial config to disk
    out_dir = os.path.join(os.path.dirname(__file__), "study_results")
    os.makedirs(out_dir, exist_ok=True)
    summary = {
        "best_value": study.best_value,
        "best_params": study.best_params,
        "datetime": datetime.datetime.now().isoformat()
    }
    summary_path = os.path.join(out_dir, "best_trial.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    best_bovw_params = {k: v for k, v in study.best_params.items() 
                        if k in ["detector_type", "codebook_size", "dense_sift", "sift_step", "sift_scales"]}
    # include scaler option if present in best params
    if "use_scaler" in study.best_params:
        best_bovw_params["use_scaler"] = study.best_params["use_scaler"]
    best_classifier_type = study.best_params.get("classifier_type", "logreg")
    best_classifier_params = {k: v for k, v in study.best_params.items() 
                              if k in ["C", "gamma"]}
    
    print("=== Final Test with Best Params ===")
    train_acc, test_acc, _, _ = test_final(train_data, test_data, 
                                           best_bovw_params, best_classifier_params,
                                           classifier_type=best_classifier_type)
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    # Save final evaluation results
    final_results = {
        "train_accuracy": float(train_acc),
        "test_accuracy": float(test_acc),
        "best_bovw_params": best_bovw_params,
        "best_classifier_type": best_classifier_type,
        "best_classifier_params": best_classifier_params,
        "datetime": datetime.datetime.now().isoformat()
    }
    final_path = os.path.join(out_dir, "final_results.json")
    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2)
