# Refined Architecture: Intelligent Risk Assessment and Loan Approval System

## 1. Updated Architecture
The architecture is structured across three primary tiers, designed to facilitate a modular, robust, and reproducible machine learning pipeline:

- **Presentation Layer (Frontend):** Built with Streamlit, providing an interactive UI for users to input loan application details, view approval status, risk percentage, and SHAP-based explanations.
- **Application Layer (Backend Logic):** Python modules handling isolated responsibilities including data processing, feature engineering, independent model training, ensemble modeling, model comparison, inference, and risk scaling.
- **Data & Model Layer:** Maintains raw and processed data, trained models, evaluation metrics, and specifically serialized preprocessing artifacts (encoders, scalers, imputers) to ensure identical data transformations during training and inference.

**Formal Risk Assessment Methodology:**
- The probability output from the prediction model (probability of default or risk) is scaled to a percentage (0% to 100%).
- **Business Logic/Risk Categories:**
  - **Low Risk (0% - 30%):** High likelihood of repayment; standard approval criteria apply.
  - **Medium Risk (31% - 70%):** Conditional approval; may require manual review, higher interest rates, or additional collateral.
  - **High Risk (71% - 100%):** High likelihood of default; standard rejection or severe restrictive terms apply.

**Final Production Prediction Model Strategy:**
- The final deployed application will utilize the **Hybrid Ensemble model** (Voting/Stacking) by default to leverage the generalized performance of the base models.
- The system will allow fallback to the best-performing individual model if `comparison.py` identifies a base model that outperforms the ensemble significantly or offers drastically better latency without sacrificing substantial accuracy. The final selection is documented in `models/metrics/best_model_info.json`.

**Standardized Explainability:**
- **SHAP** (SHapley Additive exPlanations) is established as the primary explainability framework to provide local explanations (force plots for individual applicants) and global explanations (feature importance). LIME is optional.

## 2. Updated Folder Structure
The folder structure has been expanded to accommodate separated training scripts, comparison metrics, and preprocessing artifacts.

```text
intelligent-loan-system/
│
├── data/
│   ├── raw/                       # Original Kaggle dataset
│   └── processed/                 # Cleaned and engineered datasets
│
├── models/                        # Serialized artifacts
│   ├── preprocessing/             # Saved Encoders, Scalers, Imputers
│   ├── base_models/               # LR, DT, RF, XGB, NN (.pkl, .h5)
│   ├── ensemble_model/            # Stacking/Voting models
│   └── metrics/                   # Evaluation reports, comparison results, best model
│
├── src/                           # Core Python modules
│   ├── __init__.py
│   ├── data_processing.py         # Data cleaning, imputing, scaling, encoding
│   ├── feature_engineering.py     # Feature creation, selection
│   ├── train_logistic.py          # Logistic Regression training
│   ├── train_decision_tree.py     # Decision Tree training
│   ├── train_random_forest.py     # Random Forest training
│   ├── train_xgboost.py           # XGBoost training
│   ├── train_neural_network.py    # Neural Network training
│   ├── train_ensemble.py          # Hybrid model training
│   ├── comparison.py              # Model evaluation and comparison module
│   ├── inference.py               # Loads models and artifacts, makes predictions
│   ├── risk_calculator.py         # Logic for risk percentage calculation
│   └── explainability.py          # SHAP integration
│
├── notebooks/                     # Jupyter notebooks for EDA and experimentation
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Experiments.ipynb
│   └── 03_Ensemble_Tuning.ipynb
│
├── app/                           # Streamlit frontend application
│   ├── main.py                    # Streamlit entry point
│   ├── components.py              # Reusable UI components
│   └── utils.py                   # UI helper functions
│
├── requirements.txt               # Project dependencies
├── README.md                      # Project documentation
└── config.yaml                    # Configuration (hyperparameters, paths)
```

## 3. Updated Module Responsibilities

### `src/data_processing.py`
- **Inputs:** Raw Kaggle dataset (`data/raw/`).
- **Outputs:** Processed datasets (`data/processed/`); Serialized preprocessing artifacts (`models/preprocessing/` - imputers, scalers, encoders).
- **Dependencies:** `pandas`, `scikit-learn`, `joblib`.
- **Integration Points:** Feeds data into `feature_engineering.py`. Its saved artifacts are directly loaded by `inference.py`.

### `src/feature_engineering.py`
- **Inputs:** Cleaned datasets from `data_processing.py`.
- **Outputs:** Final modeling datasets with engineered features.
- **Dependencies:** `pandas`, `numpy`.
- **Integration Points:** Provides the final X and y datasets to all `train_*.py` modules.

### `src/train_logistic.py` through `src/train_neural_network.py`
- **Inputs:** Engineered train/test datasets.
- **Outputs:** Serialized model artifacts saved in `models/base_models/`.
- **Dependencies:** `scikit-learn`, `xgboost`, `tensorflow`/`pytorch`.
- **Integration Points:** Supplies base models to `train_ensemble.py` and `comparison.py`.

### `src/train_ensemble.py`
- **Inputs:** Engineered train dataset, serialized base models.
- **Outputs:** Serialized hybrid model artifact (Voting/Stacking) saved in `models/ensemble_model/`.
- **Dependencies:** `scikit-learn`.
- **Integration Points:** Supplies the final ensemble model to `comparison.py` and `inference.py`.

### `src/comparison.py`
- **Inputs:** Engineered test dataset, all serialized base and ensemble models.
- **Outputs:** Performance metrics (Accuracy, Precision, Recall, F1, ROC-AUC); Comparison tables and visualization plots saved to `models/metrics/`.
- **Dependencies:** `scikit-learn` metrics, `matplotlib`, `seaborn`.
- **Integration Points:** Serves as the ultimate decision gate to define the "Final Production Model" based on quantitative results.

### `src/inference.py`
- **Inputs:** Raw user input fields from the Streamlit frontend.
- **Outputs:** Prediction class (Yes/No) and default probability.
- **Dependencies:** `joblib`, `scikit-learn`.
- **Integration Points:** The core backend bridge. It loads the `models/preprocessing/` artifacts to transform raw user input, then loads the final production model to generate predictions, feeding the results to `risk_calculator.py`.

### `src/risk_calculator.py`
- **Inputs:** Prediction probabilities from `inference.py`.
- **Outputs:** Calculated Risk Percentage (0-100%) and Risk Category (Low, Medium, High).
- **Dependencies:** Core Python.
- **Integration Points:** Sits between the ML pipeline and the UI. Streamlit (`app/main.py`) calls this to display human-readable risk assessments.

### `src/explainability.py`
- **Inputs:** Transformed user instance, final production model.
- **Outputs:** SHAP values, force plots, and summary plots.
- **Dependencies:** `shap`, `matplotlib`.
- **Integration Points:** Streamlit (`app/main.py`) invokes this module to render local explainability graphics for the end user.

### `app/main.py`
- **Inputs:** User interaction via web interface.
- **Outputs:** Web dashboard displaying predictions, risk assessments, and SHAP charts.
- **Dependencies:** `streamlit`, `src.inference`, `src.risk_calculator`, `src.explainability`.
- **Integration Points:** The main entry point orchestrating user input collection and rendering outputs from the backend modules.
