import sys
import os
from sklearn.linear_model import LogisticRegression

# Ensure the root project directory is in the Python path
# This allows execution from any directory (e.g., `python src/train_logistic.py` or `cd src && python train_logistic.py`)
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
    logger.info("Initializing Logistic Regression Reference Implementation...")
    
    try:
        # 1. Load and split the preprocessed data
        X_train, X_val, y_train, y_val = load_and_split_data()
        
        # 2. Instantiate the model
        # max_iter is set to 1000 to prevent ConvergenceWarnings on larger/complex datasets
        model = LogisticRegression(
            random_state=RANDOM_STATE,
            max_iter=1000
        )
        
        # 3. Execute the standard training and evaluation pipeline
        model_name = "logistic_regression"
        train_evaluate_and_save(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val
        )
        
        logger.info("Logistic Regression script executed successfully.")
        
    except Exception as e:
        logger.error(f"Logistic Regression training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
