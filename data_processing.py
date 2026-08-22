"""
Data Processing Module for Intelligent Loan System

This module handles:
- Loading raw datasets
- Validating inputs
- Handling missing values (Imputation)
- Target encoding
- Serializing imputation and target artifacts
- Saving clean, imputed datasets

It ensures no data leakage by fitting imputation steps strictly on training data
and applying them to the test data.
Note: Scaling and categorical encoding are deferred to the feature engineering stage.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Constants for paths relative to project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PREPROCESSING_MODELS_DIR = os.path.join(BASE_DIR, "models", "preprocessing")

TRAIN_DATA_PATH = os.path.join(RAW_DATA_DIR, "train.csv")
TEST_DATA_PATH = os.path.join(RAW_DATA_DIR, "test.csv")

PROCESSED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_processed.csv")
PROCESSED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test_processed.csv")

def create_directories():
    """Create necessary directories if they do not exist."""
    directories = [PROCESSED_DATA_DIR, PREPROCESSING_MODELS_DIR]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.info(f"Created directory: {directory}")

def validate_and_load_data(filepath: str) -> pd.DataFrame:
    """
    Validate existence of the file and load it.
    
    Args:
        filepath (str): Path to the CSV file.
        
    Returns:
        pd.DataFrame: Loaded dataset.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.EmptyDataError: If the file is empty.
    """
    if not os.path.exists(filepath):
        logger.error(f"File not found: {filepath}")
        raise FileNotFoundError(f"Missing required file: {filepath}")
        
    try:
        df = pd.read_csv(filepath)
        if df.empty:
            raise pd.errors.EmptyDataError(f"The file {filepath} is empty.")
        logger.info(f"Successfully loaded {filepath} with shape {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error loading {filepath}: {str(e)}")
        raise

def process_data():
    """
    Main function to execute the data processing pipeline (Cleaning & Imputation).
    """
    logger.info("Starting data processing pipeline (Cleaning & Imputation)...")
    
    create_directories()
    
    # 1 & 2. Load and validate data
    train_df = validate_and_load_data(TRAIN_DATA_PATH)
    test_df = validate_and_load_data(TEST_DATA_PATH)
    
    # 3. Drop Loan_ID
    if 'Loan_ID' in train_df.columns:
        train_df = train_df.drop('Loan_ID', axis=1)
    if 'Loan_ID' in test_df.columns:
        test_df = test_df.drop('Loan_ID', axis=1)
    logger.info("Dropped 'Loan_ID' column from datasets.")
    
    # Separate features and target from training data
    target_col = 'Loan_Status'
    
    if target_col not in train_df.columns:
        logger.error(f"Target column '{target_col}' not found in training data.")
        raise ValueError(f"Target column '{target_col}' not found in training data.")
        
    X_train = train_df.drop(target_col, axis=1)
    y_train = train_df[target_col]
    
    X_test = test_df.copy()
    y_test = None
    if target_col in X_test.columns:
        y_test = X_test[target_col]
        X_test = X_test.drop(target_col, axis=1)

    # 4 & 5. Explicitly treat Credit_History and Dependents as categorical,
    # ensuring they are intentionally handled rather than relying solely on implicit dtype inference.
    explicit_categorical = ['Credit_History', 'Dependents']
    
    for df in [X_train, X_test]:
        for col in explicit_categorical:
            if col in df.columns:
                df[col] = df[col].astype(str).replace('nan', np.nan)

    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Guarantee explicitly defined categorical columns are in the categorical list
    for col in explicit_categorical:
        if col in numerical_cols:
            numerical_cols.remove(col)
        if col in X_train.columns and col not in categorical_cols:
            categorical_cols.append(col)
            
    logger.info(f"Numerical columns for imputation: {numerical_cols}")
    logger.info(f"Categorical columns for imputation: {categorical_cols}")
    
    # Initialize Imputers
    cat_imputer = SimpleImputer(strategy='most_frequent')
    num_imputer = SimpleImputer(strategy='median')
    
    # Fit and transform imputers (Leakage prevented)
    logger.info("Imputing missing values...")
    if categorical_cols:
        X_train[categorical_cols] = cat_imputer.fit_transform(X_train[categorical_cols])
        X_test[categorical_cols] = cat_imputer.transform(X_test[categorical_cols])
        
    if numerical_cols:
        X_train[numerical_cols] = num_imputer.fit_transform(X_train[numerical_cols])
        X_test[numerical_cols] = num_imputer.transform(X_test[numerical_cols])

    # 6. Encode Target feature
    logger.info("Encoding target variable...")
    target_encoder = LabelEncoder()
    y_train_encoded = target_encoder.fit_transform(y_train)
    
    if y_test is not None:
        y_test_encoded = target_encoder.transform(y_test)

    # 7. Serialize and save artifacts
    logger.info("Serializing imputation and target encoding artifacts...")
    joblib.dump(cat_imputer, os.path.join(PREPROCESSING_MODELS_DIR, "cat_imputer.joblib"))
    joblib.dump(num_imputer, os.path.join(PREPROCESSING_MODELS_DIR, "num_imputer.joblib"))
    joblib.dump(target_encoder, os.path.join(PREPROCESSING_MODELS_DIR, "target_encoder.joblib"))

    # Reattach target variables
    train_processed = X_train.copy()
    train_processed[target_col] = y_train_encoded
    
    test_processed = X_test.copy()
    if y_test is not None:
        test_processed[target_col] = y_test_encoded

    # 8. Save processed datasets
    logger.info("Saving imputed datasets...")
    train_processed.to_csv(PROCESSED_TRAIN_PATH, index=False)
    test_processed.to_csv(PROCESSED_TEST_PATH, index=False)
    
    logger.info(f"Imputed training data shape: {train_processed.shape}")
    logger.info(f"Imputed test data shape: {test_processed.shape}")
    logger.info("Data processing pipeline (Cleaning & Imputation) completed successfully.")

if __name__ == "__main__":
    try:
        process_data()
    except Exception as e:
        logger.error("Data processing failed.", exc_info=True)
        sys.exit(1)
