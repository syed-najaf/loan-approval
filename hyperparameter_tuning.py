import os
import sys
import json
import logging
import pandas as pd
import numpy as np
import joblib

# Ensure project root is in sys.path for execution from any working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from src.utils.training_utils import (
    load_and_split_data,
    calculate_metrics,
    logger,
    RANDOM_STATE,
    METRICS_DIR,
    EVALUATION_JSON_PATH
)

# Tuned Output Paths (Strictly isolated from frozen baseline artifacts)
TUNED_MODELS_DIR = os.path.join(PROJECT_ROOT, "models", "tuned_models")
TUNED_EVALUATION_JSON_PATH = os.path.join(METRICS_DIR, "tuned_model_evaluation.json")
TUNING_RESULTS_JSON_PATH = os.path.join(METRICS_DIR, "tuning_results.json")
TUNED_COMPARISON_REPORT_PATH = os.path.join(METRICS_DIR, "tuned_comparison_report.md")


class TunedWeightedSoftVotingEnsemble:
    """
    Weighted Soft Voting Ensemble combining tuned base model estimators.
    Probability predictions are weighted strictly by each tuned model's internal 5-fold CV ROC-AUC score on X_train.
    The held-out X_val / y_val is NEVER used to derive or optimize ensemble weights.
    """
    def __init__(self, models_dict: dict, weights_dict: dict):
        self.models_dict = models_dict
        self.weights_dict = weights_dict

    def predict_proba(self, X) -> np.ndarray:
        weighted_p1 = np.zeros(len(X))
        for name, model in self.models_dict.items():
            weight = self.weights_dict[name]
            p1 = model.predict_proba(X)[:, 1]
            weighted_p1 += weight * p1
        p0 = 1.0 - weighted_p1
        return np.vstack((p0, weighted_p1)).T

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)


def initialize_tuned_directories():
    """Ensure tuned model output directory exists."""
    if not os.path.exists(TUNED_MODELS_DIR):
        os.makedirs(TUNED_MODELS_DIR, exist_ok=True)
        logger.info(f"Created tuned models directory: {TUNED_MODELS_DIR}")


def load_frozen_baseline_metrics() -> dict:
    """
    Read frozen baseline evaluation metrics from models/metrics/model_evaluation.json (read-only).
    """
    if not os.path.exists(EVALUATION_JSON_PATH):
        raise FileNotFoundError(f"Frozen baseline metrics not found at {EVALUATION_JSON_PATH}")

    with open(EVALUATION_JSON_PATH, "r") as f:
        baseline_metrics = json.load(f)
    return baseline_metrics


def extract_top_candidates_summary(search_obj, top_k: int = 3) -> list:
    """
    Extract the top K candidate hyperparameter configurations and their mean/std CV ROC-AUC scores
    from search_obj.cv_results_ for auditable reproducibility without bloated artifacts.
    """
    cv_results = search_obj.cv_results_
    ranks = cv_results['rank_test_score']
    
    top_indices = np.argsort(ranks)[:top_k]
    summary = []
    for idx in top_indices:
        summary.append({
            "rank": int(ranks[idx]),
            "mean_cv_roc_auc": float(cv_results['mean_test_score'][idx]),
            "std_cv_roc_auc": float(cv_results['std_test_score'][idx]),
            "params": cv_results['params'][idx]
        })
    return summary


def tune_logistic_regression(X_train, y_train, cv) -> tuple:
    """
    Tune Logistic Regression using GridSearchCV over 48 parameter combinations.
    """
    logger.info("--- Tuning Logistic Regression (GridSearchCV: 48 combinations) ---")
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        estimator=LogisticRegression(random_state=RANDOM_STATE, max_iter=1000),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Logistic Regression Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    logger.info(f"Logistic Regression Best Params: {grid_search.best_params_}")
    return grid_search, param_grid, 48


def tune_decision_tree(X_train, y_train, cv) -> tuple:
    """
    Tune Decision Tree using GridSearchCV over 384 parameter combinations.
    """
    logger.info("--- Tuning Decision Tree (GridSearchCV: 384 combinations) ---")
    param_grid = {
        'max_depth': [3, 4, 5, 6, 7, 8],
        'min_samples_split': [2, 5, 10, 15],
        'min_samples_leaf': [1, 2, 4, 6],
        'criterion': ['gini', 'entropy'],
        'class_weight': [None, 'balanced']
    }
    
    grid_search = GridSearchCV(
        estimator=DecisionTreeClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Decision Tree Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    logger.info(f"Decision Tree Best Params: {grid_search.best_params_}")
    return grid_search, param_grid, 384


def tune_random_forest(X_train, y_train, cv) -> tuple:
    """
    Tune Random Forest using RandomizedSearchCV (n_iter=50, random_state=42).
    """
    logger.info("--- Tuning Random Forest (RandomizedSearchCV: 50 sampled candidates) ---")
    param_distributions = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [4, 6, 8, 10, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    }
    
    random_search = RandomizedSearchCV(
        estimator=RandomForestClassifier(random_state=RANDOM_STATE),
        param_distributions=param_distributions,
        n_iter=50,
        scoring='roc_auc',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True
    )
    random_search.fit(X_train, y_train)
    logger.info(f"Random Forest Best CV ROC-AUC: {random_search.best_score_:.4f}")
    logger.info(f"Random Forest Best Params: {random_search.best_params_}")
    return random_search, param_distributions, 50


def tune_xgboost(X_train, y_train, cv) -> tuple:
    """
    Tune XGBoost using RandomizedSearchCV (n_iter=60, random_state=42).
    """
    logger.info("--- Tuning XGBoost (RandomizedSearchCV: 60 sampled candidates) ---")
    param_distributions = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'max_depth': [3, 4, 5, 6],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2],
        'scale_pos_weight': [1.0, 2.0]
    }
    
    random_search = RandomizedSearchCV(
        estimator=XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss'),
        param_distributions=param_distributions,
        n_iter=60,
        scoring='roc_auc',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        refit=True
    )
    random_search.fit(X_train, y_train)
    logger.info(f"XGBoost Best CV ROC-AUC: {random_search.best_score_:.4f}")
    logger.info(f"XGBoost Best Params: {random_search.best_params_}")
    return random_search, param_distributions, 60


def tune_neural_network(X_train, y_train, cv) -> tuple:
    """
    Tune MLP Neural Network using GridSearchCV with conditional parameter grids for solver compatibility.
    100 combinations (Adam) + 50 combinations (LBFGS) = 150 total combinations.
    """
    logger.info("--- Tuning Neural Network (GridSearchCV: 150 conditional combinations) ---")
    
    param_grid = [
        # Adam Solver Grid (100 combinations)
        {
            'solver': ['adam'],
            'hidden_layer_sizes': [(16,), (32,), (64,), (32, 16), (16, 8)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'learning_rate_init': [0.001, 0.01],
            'early_stopping': [False]
        },
        # LBFGS Quasi-Newton Solver Grid (50 combinations)
        {
            'solver': ['lbfgs'],
            'hidden_layer_sizes': [(16,), (32,), (64,), (32, 16), (16, 8)],
            'activation': ['relu', 'tanh'],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'max_iter': [500]
        }
    ]
    
    grid_search = GridSearchCV(
        estimator=MLPClassifier(random_state=RANDOM_STATE),
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        refit=True
    )
    grid_search.fit(X_train, y_train)
    logger.info(f"Neural Network Best CV ROC-AUC: {grid_search.best_score_:.4f}")
    logger.info(f"Neural Network Best Params: {grid_search.best_params_}")
    return grid_search, param_grid, 150


def evaluate_refitted_estimator(estimator, X_val, y_val) -> dict:
    """
    Evaluate an already-refitted best estimator once on held-out X_val/y_val.
    Note: refit=True on GridSearchCV/RandomizedSearchCV ensures estimator is already refitted on complete X_train.
    No redundant .fit() call is made.
    """
    y_pred = estimator.predict(X_val)
    y_prob = estimator.predict_proba(X_val)[:, 1] if hasattr(estimator, "predict_proba") else None
    return calculate_metrics(y_val, y_pred, y_prob)


def run_hyperparameter_tuning_pipeline():
    """
    Main orchestration pipeline for hyperparameter optimization and tuned evaluation.
    Strictly isolated from frozen baseline artifacts.
    """
    logger.info("==================================================")
    logger.info(" Starting Hyperparameter Tuning Pipeline ")
    logger.info("==================================================")

    initialize_tuned_directories()

    # 1. Load data split (X_val/y_val is held-out and completely isolated during CV & ensemble weighting)
    X_train, X_val, y_train, y_val = load_and_split_data()
    frozen_baseline_metrics = load_frozen_baseline_metrics()

    # 2. Define Stratified 5-Fold Cross-Validation for internal X_train tuning
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    tuned_estimators = {}
    tuned_cv_scores = {}
    tuned_validation_metrics = {}
    tuning_reproducibility_records = {}

    # Define base candidate tuning configurations
    tuning_tasks = [
        ("logistic_regression", tune_logistic_regression, "GridSearchCV"),
        ("decision_tree", tune_decision_tree, "GridSearchCV"),
        ("random_forest", tune_random_forest, "RandomizedSearchCV"),
        ("xgboost", tune_xgboost, "RandomizedSearchCV"),
        ("neural_network", tune_neural_network, "GridSearchCV")
    ]

    # Execute tuning for each base model
    for model_name, tune_fn, search_method in tuning_tasks:
        search_obj, search_space_def, n_candidates = tune_fn(X_train, y_train, cv)
        
        # Best estimator is ALREADY refitted on full X_train because refit=True
        best_estimator = search_obj.best_estimator_
        best_cv_score = float(search_obj.best_score_)
        tuned_cv_scores[model_name] = best_cv_score

        # Evaluate already-refitted best estimator once on held-out X_val (No redundant .fit() call)
        val_metrics = evaluate_refitted_estimator(best_estimator, X_val, y_val)
        
        # Save fitted tuned estimator artifact separately under models/tuned_models/
        tuned_model_path = os.path.join(TUNED_MODELS_DIR, f"{model_name}.joblib")
        joblib.dump(best_estimator, tuned_model_path)
        logger.info(f"Saved tuned model artifact to {tuned_model_path}")

        tuned_estimators[model_name] = best_estimator
        tuned_validation_metrics[model_name] = val_metrics

        # Baseline comparative metrics
        base_roc_auc = float(frozen_baseline_metrics[model_name]["ROC_AUC"])
        tuned_roc_auc = float(val_metrics["ROC_AUC"])
        roc_auc_delta = float(tuned_roc_auc - base_roc_auc)
        is_improved = bool(roc_auc_delta > 0.0)

        # Structure auditable optimization record for tuning_results.json
        top_candidates = extract_top_candidates_summary(search_obj, top_k=3)
        tuning_reproducibility_records[model_name] = {
            "model_name": model_name,
            "search_method": search_method,
            "optimization_metric": "ROC_AUC",
            "cv_strategy": "StratifiedKFold",
            "cv_folds": 5,
            "random_state": RANDOM_STATE,
            "n_candidates_evaluated": n_candidates,
            "n_cv_fits": n_candidates * 5,
            "search_space": search_space_def,
            "best_params": search_obj.best_params_,
            "best_cv_score": best_cv_score,
            "top_candidates_summary": top_candidates,
            "validation_metrics": val_metrics,
            "baseline_comparison": {
                "baseline_roc_auc": base_roc_auc,
                "roc_auc_delta": roc_auc_delta,
                "is_improved": is_improved
            }
        }
        logger.info(
            f"{model_name} Best CV ROC-AUC: {best_cv_score:.4f} | "
            f"Held-Out Val ROC-AUC: {tuned_roc_auc:.4f} (Baseline: {base_roc_auc:.4f}, Delta: {roc_auc_delta:+.4f})"
        )

    # 3. Construct Tuned Weighted Soft Voting Ensemble
    # CRITICAL LEAKAGE-FREE CORRECTION: Weights derived EXCLUSIVELY from internal 5-fold CV ROC-AUC on X_train.
    # X_val / y_val is NEVER used to derive or optimize ensemble weights.
    logger.info("--- Constructing Tuned Weighted Soft Voting Ensemble (Weights derived from X_train 5-fold CV) ---")
    cv_weights = {
        name: tuned_cv_scores[name]
        for name in tuned_estimators.keys()
    }
    total_cv_w = sum(cv_weights.values())
    normalized_weights = {k: float(v / total_cv_w) for k, v in cv_weights.items()}
    logger.info(f"Soft Voting Normalized Weights (from internal CV): {normalized_weights}")

    soft_voting_ensemble = TunedWeightedSoftVotingEnsemble(
        models_dict=tuned_estimators,
        weights_dict=normalized_weights
    )
    
    # Evaluate soft voting ensemble ONCE on held-out X_val
    voting_y_pred = soft_voting_ensemble.predict(X_val)
    voting_y_prob = soft_voting_ensemble.predict_proba(X_val)[:, 1]
    voting_metrics = calculate_metrics(y_val, voting_y_pred, voting_y_prob)
    
    tuned_validation_metrics["tuned_soft_voting"] = voting_metrics
    joblib.dump(soft_voting_ensemble, os.path.join(TUNED_MODELS_DIR, "tuned_soft_voting.joblib"))
    logger.info(f"Tuned Soft Voting Ensemble Validation Metrics: {voting_metrics}")

    tuning_reproducibility_records["tuned_soft_voting"] = {
        "model_name": "tuned_soft_voting",
        "search_method": "Weighted Soft Voting Consensus",
        "weight_derivation_method": "Internal Stratified 5-Fold CV ROC-AUC scores on X_train",
        "internal_cv_roc_auc_scores": cv_weights,
        "normalized_weights": normalized_weights,
        "validation_used_for_weights": False,
        "validation_metrics": voting_metrics
    }

    # 4. Construct Leakage-Free Tuned Stacking Ensemble (Out-of-Fold CV on X_train)
    # StackingClassifier generates out-of-fold predictions on X_train to fit LogisticRegression meta-learner.
    # X_val / y_val is NEVER used during meta-learner fitting.
    logger.info("--- Constructing Leakage-Free Tuned Stacking Ensemble (Out-of-Fold CV on X_train) ---")
    stacking_estimators = [(name, model) for name, model in tuned_estimators.items()]
    stacking_ensemble = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(random_state=RANDOM_STATE),
        cv=cv,
        n_jobs=-1
    )
    
    stacking_ensemble.fit(X_train, y_train)
    
    # Evaluate stacking ensemble ONCE on held-out X_val
    stacking_y_pred = stacking_ensemble.predict(X_val)
    stacking_y_prob = stacking_ensemble.predict_proba(X_val)[:, 1]
    stacking_metrics = calculate_metrics(y_val, stacking_y_pred, stacking_y_prob)
    
    tuned_validation_metrics["tuned_stacking"] = stacking_metrics
    joblib.dump(stacking_ensemble, os.path.join(TUNED_MODELS_DIR, "tuned_stacking.joblib"))
    logger.info(f"Tuned Stacking Ensemble Validation Metrics: {stacking_metrics}")

    tuning_reproducibility_records["tuned_stacking"] = {
        "model_name": "tuned_stacking",
        "search_method": "Stacking (Out-of-Fold Meta-Learner)",
        "base_estimators": list(tuned_estimators.keys()),
        "meta_estimator": "LogisticRegression(random_state=42)",
        "cv_strategy": "Out-of-Fold 5-Fold Stratified CV on X_train",
        "validation_used_for_meta_fitting": False,
        "validation_metrics": stacking_metrics
    }

    # 5. Persist Tuned Evaluation Artifacts
    # A. Save tuned_model_evaluation.json
    with open(TUNED_EVALUATION_JSON_PATH, "w") as f:
        json.dump(tuned_validation_metrics, f, indent=4)
    logger.info(f"Saved tuned metrics to {TUNED_EVALUATION_JSON_PATH}")

    # B. Save tuning_results.json
    with open(TUNING_RESULTS_JSON_PATH, "w") as f:
        json.dump(tuning_reproducibility_records, f, indent=4)
    logger.info(f"Saved tuning reproducibility log to {TUNING_RESULTS_JSON_PATH}")

    # C. Generate Tuned Comparative Report (Markdown & LaTeX)
    generate_tuned_comparison_report(
        frozen_baseline_metrics,
        tuned_validation_metrics,
        tuning_reproducibility_records
    )

    logger.info("==================================================")
    logger.info(" Hyperparameter Tuning Pipeline Completed Successfully ")
    logger.info("==================================================")
    return tuned_validation_metrics, tuning_reproducibility_records


def generate_tuned_comparison_report(baseline_metrics: dict, tuned_metrics: dict, tuning_records: dict):
    """
    Generate Markdown comparison report and IEEE LaTeX tables comparing baseline vs tuned performance.
    """
    md = []
    md.append("# Hyperparameter Optimization & Model Tuning Report")
    md.append("\n**Phase:** Post-Tuning Evaluation")
    md.append("**Optimization Objective:** Stratified 5-Fold Cross-Validation ROC-AUC on `X_train`\n")

    md.append("## 1. Frozen Baseline vs. Tuned Models Metrics Comparison\n")
    md.append("| Model Candidate | Baseline ROC-AUC | Internal CV ROC-AUC | Tuned Val ROC-AUC | ROC-AUC Delta | Tuned Accuracy | Tuned F1 | Tuning Result |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |")

    base_models = ["logistic_regression", "decision_tree", "random_forest", "xgboost", "neural_network"]
    for model in base_models:
        base_roc = baseline_metrics[model]["ROC_AUC"]
        cv_roc = tuning_records[model]["best_cv_score"]
        tuned_roc = tuned_metrics[model]["ROC_AUC"]
        delta = tuned_roc - base_roc
        acc = tuned_metrics[model]["Accuracy"]
        f1 = tuned_metrics[model]["F1_Score"]
        status = "Improved" if delta > 0 else ("Unchanged" if delta == 0 else "Regressed")
        md.append(
            f"| **{model}** | {base_roc:.4f} | {cv_roc:.4f} | {tuned_roc:.4f} | {delta:+.4f} | "
            f"{acc:.4f} | {f1:.4f} | {status} |"
        )

    md.append("\n## 2. Tuned Ensembles Standing\n")
    md.append(
        f"- **Tuned Soft Voting Ensemble:** Val ROC-AUC = {tuned_metrics['tuned_soft_voting']['ROC_AUC']:.4f}, "
        f"Val F1 Score = {tuned_metrics['tuned_soft_voting']['F1_Score']:.4f} *(Weights derived strictly from X_train 5-fold CV)*"
    )
    md.append(
        f"- **Tuned Stacking Ensemble:** Val ROC-AUC = {tuned_metrics['tuned_stacking']['ROC_AUC']:.4f}, "
        f"Val F1 Score = {tuned_metrics['tuned_stacking']['F1_Score']:.4f} *(Meta-learner trained on X_train out-of-fold predictions)*"
    )

    md.append("\n## 3. Detailed Tuning Search Summary\n")
    for model in base_models:
        record = tuning_records[model]
        md.append(f"### `{model}` ({record['search_method']})")
        md.append(f"- **Evaluated Candidates:** {record['n_candidates_evaluated']} ({record['n_cv_fits']} CV fits)")
        md.append(f"- **Best Internal 5-Fold CV ROC-AUC:** {record['best_cv_score']:.4f}")
        md.append(f"- **Selected Best Parameters:** `{json.dumps(record['best_params'])}`\n")

    md.append("## 4. IEEE Experimental Results Table (LaTeX Snippet)\n")
    md.append("```latex")
    md.append("\\begin{table}[h]")
    md.append("\\centering")
    md.append("\\caption{Comparison of Baseline and Hyperparameter-Tuned Models}")
    md.append("\\label{tab:tuned_comparison}")
    md.append("\\begin{tabular}{lcccccc}")
    md.append("\\hline")
    md.append("Model & Base ROC-AUC & CV ROC-AUC & Val ROC-AUC & Delta & Val F1 & Val Acc \\\\")
    md.append("\\hline")
    for model in base_models:
        name_tex = model.replace("_", " ").title()
        base_roc = baseline_metrics[model]["ROC_AUC"]
        cv_roc = tuning_records[model]["best_cv_score"]
        tuned_roc = tuned_metrics[model]["ROC_AUC"]
        delta = tuned_roc - base_roc
        acc = tuned_metrics[model]["Accuracy"]
        f1 = tuned_metrics[model]["F1_Score"]
        md.append(
            f"{name_tex} & {base_roc:.4f} & {cv_roc:.4f} & {tuned_roc:.4f} & {delta:+.4f} & "
            f"{f1:.4f} & {acc:.4f} \\\\"
        )
    md.append("\\hline")
    md.append("\\end{tabular}")
    md.append("\\end{table}")
    md.append("```\n")

    report_str = "\n".join(md)
    with open(TUNED_COMPARISON_REPORT_PATH, "w") as f:
        f.write(report_str)
    logger.info(f"Saved tuned comparison report to {TUNED_COMPARISON_REPORT_PATH}")


if __name__ == "__main__":
    run_hyperparameter_tuning_pipeline()
