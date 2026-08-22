import sys
import os
from xgboost import XGBClassifier

# Ensure the root project directory is in the Python path
# This allows execution from any directory (e.g., `python src/train_xgboost.py` or `cd src && python train_xgboost.py`)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.training_utils import (
    load_and_split_data, 
    train_evaluate_and_save, 
    RANDOM_STATE, 
    logger
)

def main():
    logger.info("Initializing XGBoost Baseline Implementation...")
    
    try:
        # 1. Load and split the preprocessed data
        X_train, X_val, y_train, y_val = load_and_split_data()
        
        # 2. Instantiate the model
        # Applying conservative baseline hyperparameters to prevent severe overfitting
        # n_jobs=-1 parallelizes tree construction
        model = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        
        # 3. Execute the standard training and evaluation pipeline
        model_name = "xgboost"
        train_evaluate_and_save(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val
        )
        
        logger.info("XGBoost script executed successfully.")
        
    except Exception as e:
        logger.error(f"XGBoost training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
