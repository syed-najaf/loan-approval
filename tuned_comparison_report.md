# Hyperparameter Optimization & Model Tuning Report

**Phase:** Post-Tuning Evaluation
**Optimization Objective:** Stratified 5-Fold Cross-Validation ROC-AUC on `X_train`

## 1. Frozen Baseline vs. Tuned Models Metrics Comparison

| Model Candidate | Baseline ROC-AUC | Internal CV ROC-AUC | Tuned Val ROC-AUC | ROC-AUC Delta | Tuned Accuracy | Tuned F1 | Tuning Result |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **logistic_regression** | 0.8529 | 0.7433 | 0.8545 | +0.0015 | 0.8374 | 0.8824 | Improved |
| **decision_tree** | 0.7808 | 0.7366 | 0.7672 | -0.0136 | 0.8130 | 0.8655 | Regressed |
| **random_forest** | 0.8161 | 0.7881 | 0.8279 | +0.0118 | 0.8455 | 0.8939 | Improved |
| **xgboost** | 0.7944 | 0.7785 | 0.8214 | +0.0269 | 0.8374 | 0.8876 | Improved |
| **neural_network** | 0.6093 | 0.7731 | 0.8579 | +0.2486 | 0.8537 | 0.9000 | Improved |

## 2. Tuned Ensembles Standing

- **Tuned Soft Voting Ensemble:** Val ROC-AUC = 0.8393, Val F1 Score = 0.9171 *(Weights derived strictly from X_train 5-fold CV)*
- **Tuned Stacking Ensemble:** Val ROC-AUC = 0.8443, Val F1 Score = 0.9171 *(Meta-learner trained on X_train out-of-fold predictions)*

## 3. Detailed Tuning Search Summary

### `logistic_regression` (GridSearchCV)
- **Evaluated Candidates:** 48 (240 CV fits)
- **Best Internal 5-Fold CV ROC-AUC:** 0.7433
- **Selected Best Parameters:** `{"C": 1.0, "class_weight": "balanced", "penalty": "l2", "solver": "liblinear"}`

### `decision_tree` (GridSearchCV)
- **Evaluated Candidates:** 384 (1920 CV fits)
- **Best Internal 5-Fold CV ROC-AUC:** 0.7366
- **Selected Best Parameters:** `{"class_weight": null, "criterion": "gini", "max_depth": 7, "min_samples_leaf": 6, "min_samples_split": 15}`

### `random_forest` (RandomizedSearchCV)
- **Evaluated Candidates:** 50 (250 CV fits)
- **Best Internal 5-Fold CV ROC-AUC:** 0.7881
- **Selected Best Parameters:** `{"n_estimators": 200, "min_samples_split": 10, "min_samples_leaf": 1, "max_features": 0.5, "max_depth": 8}`

### `xgboost` (RandomizedSearchCV)
- **Evaluated Candidates:** 60 (300 CV fits)
- **Best Internal 5-Fold CV ROC-AUC:** 0.7785
- **Selected Best Parameters:** `{"subsample": 0.6, "scale_pos_weight": 1.0, "n_estimators": 100, "max_depth": 6, "learning_rate": 0.05, "gamma": 0.2, "colsample_bytree": 0.8}`

### `neural_network` (GridSearchCV)
- **Evaluated Candidates:** 150 (750 CV fits)
- **Best Internal 5-Fold CV ROC-AUC:** 0.7731
- **Selected Best Parameters:** `{"activation": "relu", "alpha": 0.1, "early_stopping": false, "hidden_layer_sizes": [16], "learning_rate_init": 0.001, "solver": "adam"}`

## 4. IEEE Experimental Results Table (LaTeX Snippet)

```latex
\begin{table}[h]
\centering
\caption{Comparison of Baseline and Hyperparameter-Tuned Models}
\label{tab:tuned_comparison}
\begin{tabular}{lcccccc}
\hline
Model & Base ROC-AUC & CV ROC-AUC & Val ROC-AUC & Delta & Val F1 & Val Acc \\
\hline
Logistic Regression & 0.8529 & 0.7433 & 0.8545 & +0.0015 & 0.8824 & 0.8374 \\
Decision Tree & 0.7808 & 0.7366 & 0.7672 & -0.0136 & 0.8655 & 0.8130 \\
Random Forest & 0.8161 & 0.7881 & 0.8279 & +0.0118 & 0.8939 & 0.8455 \\
Xgboost & 0.7944 & 0.7785 & 0.8214 & +0.0269 & 0.8876 & 0.8374 \\
Neural Network & 0.6093 & 0.7731 & 0.8579 & +0.2486 & 0.9000 & 0.8537 \\
\hline
\end{tabular}
\end{table}
```
