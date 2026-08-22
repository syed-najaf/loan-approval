import sys
import os
from sklearn.neural_network import MLPClassifier

# Ensure the root project directory is in the Python path
# This allows execution from any directory (e.g., `python src/train_neural_network.py` or `cd src && python train_neural_network.py`)
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
    logger.info("Initializing Neural Network (MLPClassifier) Baseline Implementation...")
    
    try:
        # 1. Load and split the preprocessed data
        X_train, X_val, y_train, y_val = load_and_split_data()
        
        # 2. Instantiate the model
        # Applying conservative baseline hyperparameters to prevent severe overfitting
        # early_stopping=True and L2 regularization (alpha) are critical for this small dataset
        model = MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            learning_rate='constant',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            random_state=RANDOM_STATE
        )
        
        # 3. Execute the standard training and evaluation pipeline
        model_name = "neural_network"
        train_evaluate_and_save(
            model=model,
            model_name=model_name,
            X_train=X_train,
            X_val=X_val,
            y_train=y_train,
            y_val=y_val
        )
        
        logger.info("Neural Network script executed successfully.")
        
    except Exception as e:
        logger.error(f"Neural Network training failed: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
