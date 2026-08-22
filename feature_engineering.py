"""
Feature Engineering Module for Intelligent Loan System

This module handles:
- Loading the processed datasets (imputed).
- Validating the existence of required source columns.
- Separating the target variable to prevent leakage.
- Creating engineered features (TotalIncome, EMI, BalanceIncome).
- Validating against negative values before log transformations.
- Applying native log(x + 1) transformations to skewed financial features.
- Applying OneHotEncoder to categorical features.
- Applying StandardScaler to numerical features.
- Ensuring identical schemas between train and test datasets.
- Saving the final engineered, encoded, and scaled datasets.
- Serializing the final preprocessing artifacts (scaler, encoder, final_preprocessor).
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

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
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")
PREPROCESSING_MODELS_DIR = os.path.join(BASE_DIR, "models", "preprocessing")

TRAIN_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "train_processed.csv")
TEST_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "test_processed.csv")

ENGINEERED_TRAIN_PATH = os.path.join(PROCESSED_DATA_DIR, "train_engineered.csv")
ENGINEERED_TEST_PATH = os.path.join(PROCESSED_DATA_DIR, "test_engineered.csv")

def validate_columns(df: pd.DataFrame, required_columns: list, dataset_name: str) -> None:
    """Check if all required columns exist in the DataFrame."""
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        logger.error(f"Missing required columns in {dataset_name} for feature engineering: {missing_cols}")
        raise ValueError(f"Missing required columns in {dataset_name}: {missing_cols}")

def engineer_features(df_train: pd.DataFrame, df_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Creates new engineered features and applies log transformations consistently across train and test sets.
    """
    train_eng = df_train.copy()
    test_eng = df_test.copy()
    
    required_cols = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
    validate_columns(train_eng, required_cols, "train")
    validate_columns(test_eng, required_cols, "test")
    
    # Preserve categorical explicitly before doing any math/extraction
    explicit_categorical = ['Credit_History', 'Dependents']
    for df in (train_eng, test_eng):
        for col in explicit_categorical:
            if col in df.columns:
                df[col] = df[col].astype(str)
                
    logger.info("Creating domain-specific features: TotalIncome, EMI, BalanceIncome...")
    for df in (train_eng, test_eng):
        # 1. TotalIncome
        df['TotalIncome'] = df['ApplicantIncome'] + df['CoapplicantIncome']
        
        # 2. EMI (preventing division by zero)
        safe_term = np.where(df['Loan_Amount_Term'] == 0, 1e-5, df['Loan_Amount_Term'])
        df['EMI'] = df['LoanAmount'] / safe_term
        
        # 3. BalanceIncome
        df['BalanceIncome'] = df['TotalIncome'] - df['EMI']
        
    # 4. Log transformations
    log_features = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'TotalIncome', 'EMI']
    logger.info(f"Applying log(x+1) transformation to features: {log_features}")
    
    # 5. Validate for negative values before log transforms
    for df_name, df in [("train", train_eng), ("test", test_eng)]:
        for col in log_features:
            if col in df.columns:
                if (df[col] < 0).any():
                    logger.error(f"Negative values detected in {df_name} dataset for feature {col}. Cannot apply log1p natively.")
                    raise ValueError(f"Unexpected negative values in {col} prior to log transformation.")
                    
    # Apply native np.log1p safely
    for df in (train_eng, test_eng):
        for col in log_features:
            if col in df.columns:
                df[f'{col}_Log'] = np.log1p(df[col])
            
    return train_eng, test_eng

def process_and_save():
    """Main execution function for feature engineering, encoding, and scaling."""
    logger.info("Starting feature engineering pipeline...")
    
    if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(TEST_DATA_PATH):
        logger.error(f"Processed datasets not found in {PROCESSED_DATA_DIR}. Please run data_processing.py first.")
        raise FileNotFoundError("Missing processed train/test datasets.")
        
    # Load processed data
    logger.info("Loading imputed datasets...")
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    test_df = pd.read_csv(TEST_DATA_PATH)
    
    logger.info(f"Initial train shape: {train_df.shape}")
    logger.info(f"Initial test shape: {test_df.shape}")
    
    train_initial_len = len(train_df)
    test_initial_len = len(test_df)
    
    # Temporarily separate target column to prevent leakage and encoding issues
    target_col = 'Loan_Status'
    if target_col not in train_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in training data.")
        
    y_train = train_df.pop(target_col)
    y_test = test_df.pop(target_col) if target_col in test_df.columns else None
    
    # Engineer features (includes validation and log transforms)
    train_engineered, test_engineered = engineer_features(train_df, test_df)
    
    # Identify categorical and numerical columns dynamically
    cat_cols = train_engineered.select_dtypes(include=['object', 'category']).columns.tolist()
    num_cols = train_engineered.select_dtypes(include=['int64', 'float64']).columns.tolist()
    logger.info(f"Identified categorical columns: {cat_cols}")
    logger.info(f"Identified numerical columns: {num_cols}")

    # Apply OneHotEncoder
    logger.info("Applying OneHotEncoder to categorical features...")
    try:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
    train_cat_encoded = encoder.fit_transform(train_engineered[cat_cols])
    test_cat_encoded = encoder.transform(test_engineered[cat_cols])
    
    cat_feature_names = encoder.get_feature_names_out(cat_cols)
    
    train_cat_df = pd.DataFrame(train_cat_encoded, columns=cat_feature_names, index=train_engineered.index)
    test_cat_df = pd.DataFrame(test_cat_encoded, columns=cat_feature_names, index=test_engineered.index)

    # Apply StandardScaler
    logger.info("Applying StandardScaler to numerical features...")
    scaler = StandardScaler()
    
    train_num_scaled = scaler.fit_transform(train_engineered[num_cols])
    test_num_scaled = scaler.transform(test_engineered[num_cols])
    
    train_num_df = pd.DataFrame(train_num_scaled, columns=num_cols, index=train_engineered.index)
    test_num_df = pd.DataFrame(test_num_scaled, columns=num_cols, index=test_engineered.index)

    # Combine encoded and scaled features
    train_final = pd.concat([train_num_df, train_cat_df], axis=1)
    test_final = pd.concat([test_num_df, test_cat_df], axis=1)

    # Reattach target variable
    train_final[target_col] = y_train
    if y_test is not None:
        test_final[target_col] = y_test

    # Validation checks
    if len(train_final) != train_initial_len or len(test_final) != test_initial_len:
        logger.error("Row counts changed during feature engineering/encoding.")
        raise ValueError("Row count mismatch. Row order/count must be strictly preserved.")
        
    train_features = set(train_final.columns) - {target_col}
    test_features = set(test_final.columns) - {target_col}
    
    if train_features != test_features:
        logger.error("Feature schemas for train and test do not match after encoding.")
        logger.error(f"Differences: {train_features.symmetric_difference(test_features)}")
        raise ValueError("Mismatched schemas between train and test datasets.")
        
    logger.info("Feature schemas and row counts successfully validated.")

    # Save artifacts
    logger.info("Serializing encoding and scaling artifacts...")
    if not os.path.exists(PREPROCESSING_MODELS_DIR):
        os.makedirs(PREPROCESSING_MODELS_DIR)
        
    joblib.dump(encoder, os.path.join(PREPROCESSING_MODELS_DIR, "encoder.joblib"))
    joblib.dump(scaler, os.path.join(PREPROCESSING_MODELS_DIR, "scaler.joblib"))
    
    # Create reusable preprocessor for inference
    try:
        cat_transformer = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    except TypeError:
        cat_transformer = OneHotEncoder(sparse=False, handle_unknown='ignore')
        
    final_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_cols),
            ('cat', cat_transformer, cat_cols)
        ]
    )
    # Fit the final_preprocessor so it is ready for deployment
    final_preprocessor.fit(train_engineered)
    joblib.dump(final_preprocessor, os.path.join(PREPROCESSING_MODELS_DIR, "final_preprocessor.joblib"))

    # Save engineered data
    logger.info("Saving engineered, scaled, and encoded datasets...")
    train_final.to_csv(ENGINEERED_TRAIN_PATH, index=False)
    test_final.to_csv(ENGINEERED_TEST_PATH, index=False)
    
    logger.info(f"Final train shape: {train_final.shape}")
    logger.info(f"Final test shape: {test_final.shape}")
    logger.info(f"Successfully saved to {ENGINEERED_TRAIN_PATH} and {ENGINEERED_TEST_PATH}")
    logger.info("Feature engineering pipeline completed successfully.")

if __name__ == "__main__":
    try:
        process_and_save()
    except Exception as e:
        logger.error("Feature engineering failed.", exc_info=True)
        sys.exit(1)
