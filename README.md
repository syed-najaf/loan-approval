# Intelligent Risk Assessment and Loan Approval System Using Machine Learning

A machine learning system that predicts loan approval outcomes, calculates a risk score, and categorizes applicants into risk tiers. Built as a college BE CSE project (VTU), with a full ML pipeline (baseline models → hyperparameter tuning → ensembling) already complete and an application layer (inference, explainability, UI) currently in progress.

> **Status: Work in Progress.** The model development phase is complete. The application/inference phase has just begun. See [Project Status](#project-status) below for exactly what is and isn't implemented.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Project Status](#project-status)
- [Results](#results)
- [Repository Structure](#repository-structure)
- [Roadmap](#roadmap)

---

## Overview

The system takes applicant information (income, credit history, loan amount, etc.) and predicts:

1. Loan approval outcome
2. Approval probability / risk score
3. Risk category (Low / Medium / High)
4. An explanation of the prediction (planned, via SHAP)

The intended final system is a three-tier architecture:

- **Presentation Layer** — Streamlit UI
- **Application Layer** — inference, risk calculation, explainability
- **Data/Model Layer** — preprocessing, feature engineering, trained models

**Risk categories:**

| Category | Probability Range |
|----------|-------------------|
| Low Risk | 0–30% |
| Medium Risk | 31–70% |
| High Risk | 71–100% |

## Architecture

```
                    USER
                     |
                     v
             Streamlit Interface
                     |
                     v
              Applicant Inputs
                     |
                     v
              Inference Module
                     |
                     v
        Preprocessing / Feature Engineering
                     |
                     v
          Tuned Stacking Ensemble
                     |
          +----------+----------+
          |                     |
          v                     v
   Loan Prediction        Probability Score
                                |
                                v
                       Risk Calculator
                                |
                                v
                      Risk Percentage
                                |
                                v
                       Risk Category
                  /        |        \
                 /         |         \
              Low       Medium       High
                                |
                                v
                       SHAP Explainability
                                |
                                v
                      User-Friendly Results
```

*This diagram represents the target final architecture. Components not yet implemented are marked in [Project Status](#project-status).*

## Dataset

[Kaggle Loan Prediction Dataset](https://www.kaggle.com/)

| | Rows | Columns |
|---|---|---|
| Training | 614 | 13 |
| Test | 367 | 12 |
| Engineered training data | 614 | 30 |

Train/validation split: `test_size=0.2`, `random_state=42`, stratified on `Loan_Status` → 491 train / 123 validation samples.

## Project Status

### ✅ Completed

- [x] Architecture design & planning
- [x] Dataset validation & leakage audit
- [x] Data cleaning / imputation pipeline (`src/data_processing.py`)
- [x] Feature engineering — TotalIncome, EMI, BalanceIncome, log transforms (`src/feature_engineering.py`)
- [x] 5 baseline models: Logistic Regression, Decision Tree, Random Forest, XGBoost, Neural Network
- [x] Baseline soft-voting ensemble & comparison report
- [x] Hyperparameter tuning (GridSearch / RandomizedSearch, 5-fold stratified CV, leakage-free)
- [x] Tuned ensembles: soft voting + stacking
- [x] Final model selection audit

### 🚧 In Progress / Not Yet Implemented

- [ ] `src/inference.py` — **current focus.** Raw-input → prediction pipeline
- [ ] `src/risk_calculator.py` — probability → risk category mapping
- [ ] `src/explainability.py` — SHAP-based explanations
- [ ] Streamlit app (`app/main.py`, `app/components.py`, `app/utils.py`)
- [ ] `config.yaml`
- [ ] Notebooks (EDA, model experiments, ensemble tuning)

### ⚠️ Known Issues

- `models/metrics/best_model_info.json` currently still points to the baseline Logistic Regression model rather than the tuned stacking ensemble — needs updating during the application phase.
- The custom `TunedWeightedSoftVotingEnsemble` class must be importable wherever `joblib.load()` deserializes the soft-voting model.

## Results

### Baseline Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8618 | 0.8400 | 0.9882 | 0.9081 | **0.8529** |
| Decision Tree | 0.8618 | 0.8542 | 0.9647 | 0.9061 | 0.7808 |
| Random Forest | 0.8537 | 0.8317 | 0.9882 | 0.9032 | 0.8161 |
| XGBoost | 0.8537 | 0.8384 | 0.9765 | 0.9022 | 0.7944 |
| Neural Network | 0.7154 | 0.7083 | 1.0000 | 0.8293 | 0.6093 |
| Soft Voting Ensemble | 0.8537 | 0.8317 | 0.9882 | 0.9032 | 0.8359 |

### Tuned Validation Results

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.8374 | 0.8824 | 0.8824 | 0.8824 | 0.8545 |
| Decision Tree | 0.8130 | 0.8605 | 0.8706 | 0.8655 | 0.7672 |
| Random Forest | 0.8455 | 0.8511 | 0.9412 | 0.8939 | 0.8279 |
| XGBoost | 0.8374 | 0.8495 | 0.9294 | 0.8876 | 0.8214 |
| Neural Network | 0.8537 | 0.8526 | 0.9529 | 0.9000 | **0.8579** |
| **Tuned Soft Voting** | 0.8780 | 0.8646 | 0.9765 | 0.9171 | 0.8393 |
| **Tuned Stacking (Production)** | **0.8780** | 0.8646 | **0.9765** | **0.9171** | 0.8443 |

**Production model:** `models/tuned_models/tuned_stacking.joblib` — selected for the best overall balance of Accuracy, F1, and Recall, even though the tuned Neural Network has a marginally higher ROC-AUC (0.8579).

Stacking architecture: base estimators (Logistic Regression, Decision Tree, Random Forest, XGBoost, Neural Network) → Logistic Regression meta-learner, trained on out-of-fold predictions from `X_train` only.

## Repository Structure

```
├── data/
│   └── processed/
│       └── train_engineered.csv
├── models/
│   ├── preprocessing/       # imputers, encoders, scaler
│   ├── base_models/         # 5 baseline models
│   ├── tuned_models/        # tuned models + ensembles
│   └── metrics/             # evaluation reports, comparisons
├── src/
│   ├── data_processing.py
│   ├── feature_engineering.py
│   ├── train_logistic.py
│   ├── train_decision_tree.py
│   ├── train_random_forest.py
│   ├── train_xgboost.py
│   ├── train_neural_network.py
│   ├── train_ensemble.py
│   ├── hyperparameter_tuning.py
│   ├── comparison.py
│   ├── inference.py         # not yet implemented
│   ├── risk_calculator.py   # not yet implemented
│   └── explainability.py    # not yet implemented
├── app/                      # Streamlit app — not yet implemented
│   ├── main.py
│   ├── components.py
│   └── utils.py
├── notebooks/                 # optional, not yet implemented
├── config.yaml                # not yet implemented
└── README.md
```

## Roadmap

```
Planning → Architecture → Data Processing → Feature Engineering →
Base Models → Baseline Ensemble → Baseline Evaluation →
Hyperparameter Tuning → Final Model Selection →
[ Inference Layer ] ← current phase
→ Risk Calculator → SHAP Explainability → Streamlit Frontend →
Integration Testing → Documentation → Final Project
```

---

## Tech Stack

- **ML:** scikit-learn, XGBoost
- **Explainability (planned):** SHAP
- **Frontend (planned):** Streamlit
- **Language:** Python

## License

*(Add your license here — check the Kaggle dataset's terms before redistributing raw data files.)*
