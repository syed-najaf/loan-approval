import sys
import os
import json
import joblib
import numpy as np

# Ensure the root project directory is in the Python path
# This allows execution from any directory (e.g., `python src/train_ensemble.py` or `cd src && python train_ensemble.py`)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.training_utils import (
    load_and_split_data,
    calculate_metrics,
    save_metrics,
    logger,
    MODELS_DIR,
    EVALUATION_JSON_PATH
)

BASE_MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "neural_network"
]

class WeightedSoftVotingEnsemble:
    """
    Custom Weighted Soft Voting Ensemble combining frozen base models.
    Aggregates predicted probabilities weighted by each model's normalized ROC-AUC score.
    """
    def __init__(self, models_dict, weights_dict):
        """
        Args:
            models_dict (dict): Mapping of model name -> loaded model object.
            weights_dict (dict): Mapping of model name -> normalized weight (float).
        """
        self.models_dict = models_dict
        self.weights_dict = weights_dict

    def predict_proba(self, X):
        """
        Compute weighted soft voting probabilities for class 0 and class 1.
        Returns:
            np.ndarray: Probability matrix of shape (n_samples, 2).
        """
        weighted_p1 = np.zeros(len(X))

        for name, model in self.models_dict.items():
            weight = self.weights_dict[name]
            p1 = model.predict_proba(X)[:, 1]
            weighted_p1 += weight * p1

        p0 = 1.0 - weighted_p1
        return np.vstack((p0, weighted_p1)).T

    def predict(self, X, threshold=0.5):
        """
        Generate binary class predictions based on aggregated probability threshold.
        Returns:
            np.ndarray: Binary class array (0 or 1).
        """
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= threshold).astype(int)

def main():
    logger.info("Initializing Hybrid Weighted Soft Voting Ensemble Pipeline...")

    try:
        # 1. Load validation data split
        _, X_val, _, y_val = load_and_split_data()

        # 2. Read evaluation metrics JSON to retrieve frozen ROC-AUC scores
        logger.info(f"Reading evaluation metrics from {EVALUATION_JSON_PATH}")
        if not os.path.exists(EVALUATION_JSON_PATH):
            raise FileNotFoundError(f"Metrics file not found at {EVALUATION_JSON_PATH}")

        with open(EVALUATION_JSON_PATH, 'r') as f:
            evaluation_metrics = json.load(f)

        # Extract ROC-AUC scores for each base model
        roc_auc_scores = {}
        for name in BASE_MODEL_NAMES:
            if name not in evaluation_metrics or "ROC_AUC" not in evaluation_metrics[name]:
                raise KeyError(f"ROC_AUC metric for '{name}' not found in {EVALUATION_JSON_PATH}")
            roc_auc_scores[name] = evaluation_metrics[name]["ROC_AUC"]
            logger.info(f"Loaded ROC-AUC for {name}: {roc_auc_scores[name]:.4f}")

        # Compute normalized weights (sum to 1.0)
        total_roc_auc = sum(roc_auc_scores.values())
        weights = {name: score / total_roc_auc for name, score in roc_auc_scores.items()}
        for name, weight in weights.items():
            logger.info(f"Calculated normalized weight for {name}: {weight:.4f}")

        # 3. Load pre-trained frozen base model artifacts
        logger.info(f"Loading frozen base model artifacts from {MODELS_DIR}")
        models = {}
        for name in BASE_MODEL_NAMES:
            model_path = os.path.join(MODELS_DIR, f"{name}.joblib")
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model artifact not found: {model_path}")
            models[name] = joblib.load(model_path)
            logger.info(f"Successfully loaded {name} artifact from {model_path}")

        # 4. Construct Weighted Soft Voting Ensemble
        ensemble = WeightedSoftVotingEnsemble(models_dict=models, weights_dict=weights)

        # 5. Obtain probabilities and predictions on validation set
        logger.info("Generating ensemble soft voting predictions on validation set...")
        y_prob_matrix = ensemble.predict_proba(X_val)
        y_prob_class1 = y_prob_matrix[:, 1]
        y_pred = ensemble.predict(X_val)

        # 6. Calculate evaluation metrics
        metrics = calculate_metrics(y_val, y_pred, y_prob_class1)
        logger.info(f"Ensemble Metrics: {metrics}")

        # 7. Persist metrics using existing persistence mechanism
        save_metrics("ensemble", metrics)

        logger.info("Ensemble training and evaluation script executed successfully.")

    except Exception as e:
        logger.error(f"Ensemble execution failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
