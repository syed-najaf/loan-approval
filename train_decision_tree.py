import sys
import os
from sklearn.tree import DecisionTreeClassifier

# Ensure the root project directory is in the Python path
# This allows execution from any directory (e.g., `python src/train_decision_tree.py` or `cd src && python train_decision_tree.py`)
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
    logger.info("Initializing Decision Tree Baseline Implementation...")
    
    try:
        # 1. Load and split the preprocessed data
        X_train, X_val, y_train, y_val = load_and_split_data()
        
        # 2. Instantiate the model
        # Applying conservative baseline hyperparameters to prevent severe overfitting
        # that would naturally occur with unconstrained trees on a small dataset.
        model = DecisionTreeClassifier(
            criterion="gini",
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        )
        
        # 3. Execute the standard training and evaluation pipeline
        model_name = "decision_tree"
        train_evaluate_and_save(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val
        )
        
        logger.info("Decision Tree script executed successfully.")
        
    except Exception as e:
        logger.error(f"Decision Tree training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
