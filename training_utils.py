import os
import sys
import json
import logging
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("training_utils")

# Global Constants
RANDOM_STATE = 42
# src/utils/training_utils.py -> BASE_DIR is three levels up: src/utils/ -> src/ -> root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "train_engineered.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models", "base_models")
METRICS_DIR = os.path.join(BASE_DIR, "models", "metrics")
EVALUATION_JSON_PATH = os.path.join(METRICS_DIR, "model_evaluation.json")

def initialize_directories():
    """Create required directories automatically if they are missing."""
    for directory in [MODELS_DIR, METRICS_DIR]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
            logger.info(f"Created directory: {directory}")

def load_and_split_data():
    """
    Load train_engineered.csv, verify target exists, and split into internal train/val sets.
    Returns:
        X_train, X_val, y_train, y_val as Pandas DataFrames/Series
    """
    logger.info(f"Loading data from {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Training data not found at {DATA_PATH}. Please run preprocessing first.")
        
    df = pd.read_csv(DATA_PATH)
    
    target_col = 'Loan_Status'
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in the training dataset.")
        
    # Separate X and y
    y = df.pop(target_col)
    X = df
    
    # Create internal stratified train/validation split
    logger.info(f"Performing stratified train_test_split with RANDOM_STATE={RANDOM_STATE}")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=0.2, 
        stratify=y, 
        random_state=RANDOM_STATE
    )
    
    logger.info(f"Training set size: {X_train.shape[0]} samples")
    logger.info(f"Validation set size: {X_val.shape[0]} samples")
    
    # Ensure Pandas format to preserve feature names for SHAP
    return X_train, X_val, y_train, y_val

def calculate_metrics(y_true, y_pred, y_prob):
    """
    Centralized metric calculation.
    """
    metrics = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1_Score": float(f1_score(y_true, y_pred, zero_division=0))
    }
    
    if y_prob is not None:
        metrics["ROC_AUC"] = float(roc_auc_score(y_true, y_prob))
    else:
        metrics["ROC_AUC"] = None
        
    return metrics

def save_model(model, model_name):
    """
    Serialize the trained model using joblib.
    """
    initialize_directories()
    model_path = os.path.join(MODELS_DIR, f"{model_name}.joblib")
    joblib.dump(model, model_path)
    logger.info(f"Successfully saved {model_name} artifact to {model_path}")

def save_metrics(model_name, metrics):
    """
    Persist metrics into a centralized JSON file.
    """
    initialize_directories()
    
    # Load existing metrics if the file exists
    evaluation_data = {}
    if os.path.exists(EVALUATION_JSON_PATH):
        with open(EVALUATION_JSON_PATH, 'r') as f:
            try:
                evaluation_data = json.load(f)
            except json.JSONDecodeError:
                logger.warning("Existing metrics JSON is corrupted. Overwriting.")
                
    # Update with new metrics
    evaluation_data[model_name] = metrics
    
    # Save back to file
    with open(EVALUATION_JSON_PATH, 'w') as f:
        json.dump(evaluation_data, f, indent=4)
        
    logger.info(f"Appended {model_name} metrics to {EVALUATION_JSON_PATH}")

def train_evaluate_and_save(model, model_name, X_train, X_val, y_train, y_val):
    """
    Master pipeline execution function to be called by base training scripts.
    """
    logger.info(f"--- Starting Pipeline for {model_name} ---")
    
    # 1. Fit the model (keeping DataFrames intact for feature names)
    logger.info(f"Fitting {model_name}...")
    model.fit(X_train, y_train)
    
    # 2. Generate Predictions
    logger.info("Generating validation predictions...")
    y_pred = model.predict(X_val)
    
    y_prob = None
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_val)[:, 1]
        
    # 3. Calculate metrics
    metrics = calculate_metrics(y_val, y_pred, y_prob)
    logger.info(f"{model_name} Metrics: {metrics}")
    
    # 4. Save artifacts and metrics
    save_model(model, model_name)
    save_metrics(model_name, metrics)
    
    logger.info(f"--- Completed Pipeline for {model_name} ---")
