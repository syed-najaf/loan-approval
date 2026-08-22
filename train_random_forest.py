import sys
import os
from sklearn.ensemble import RandomForestClassifier

# Ensure the root project directory is in the Python path
# This allows execution from any directory (e.g., `python src/train_random_forest.py` or `cd src && python train_random_forest.py`)
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
    logger.info("Initializing Random Forest Baseline Implementation...")
    
    try:
        # 1. Load and split the preprocessed data
        X_train, X_val, y_train, y_val = load_and_split_data()
        
        # 2. Instantiate the model
        # Applying conservative baseline hyperparameters to prevent severe overfitting
        # n_jobs=-1 parallelizes tree construction, bootstrap=True documents standard bagging
        model = RandomForestClassifier(
            criterion="gini",
            n_estimators=100,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            bootstrap=True
        )
        
        # 3. Execute the standard training and evaluation pipeline
        model_name = "random_forest"
        train_evaluate_and_save(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val
        )
        
        logger.info("Random Forest script executed successfully.")
        
    except Exception as e:
        logger.error(f"Random Forest training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
