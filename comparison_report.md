# Baseline Model Comparison & Selection Report

**Phase:** Baseline Model Evaluation (Frozen Checkpoint)
**Primary Selection Metrics:** ROC-AUC, F1 Score

## 1. Quantitative Metrics Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **logistic_regression** | 0.8618 | 0.8400 | 0.9882 | 0.9081 | 0.8529 |
| decision_tree | 0.8618 | 0.8542 | 0.9647 | 0.9061 | 0.7808 |
| random_forest | 0.8537 | 0.8317 | 0.9882 | 0.9032 | 0.8161 |
| xgboost | 0.8537 | 0.8384 | 0.9765 | 0.9022 | 0.7944 |
| neural_network | 0.7154 | 0.7083 | 1.0000 | 0.8293 | 0.6093 |
| ensemble | 0.8537 | 0.8317 | 0.9882 | 0.9032 | 0.8359 |

## 2. Individual Metric Leaders

- **Highest Accuracy:** `logistic_regression, decision_tree` (0.8618)
- **Highest Precision:** `decision_tree` (0.8542)
- **Highest Recall:** `neural_network` (1.0000)
- **Highest F1_Score:** `logistic_regression` (0.9081)
- **Highest ROC_AUC:** `logistic_regression` (0.8529)

## 3. Objective Ensemble Analysis

- **Ensemble Status:** Treated as a first-class candidate alongside base models.
- **Ensemble ROC-AUC:** 0.8359
- **Ensemble F1 Score:** 0.9032
- **Performance Relative to Top Performer:** ROC-AUC Delta of -0.0170 vs `logistic_regression`.

*Analysis Note:* Soft-voting ensemble performance reflects equal/weighted consensus. Without hyperparameter tuning on base estimators (specifically the uncalibrated neural network), the ensemble does not automatically surpass the linear baseline.

## 4. Current Baseline Selection & Rationale

**Current Baseline Winner:** `logistic_regression`

> Selected 'logistic_regression' as the current baseline checkpoint leader. It achieved the highest ROC-AUC (0.8529) and highest F1 Score (0.9081) among all six evaluated approaches. The soft-voting ensemble achieved an ROC-AUC of 0.8359 and F1 Score of 0.9032. While competitive, the ensemble was slightly degraded by the baseline Neural Network (ROC-AUC 0.6093), demonstrating that ensembles are not inherently superior to well-tuned single linear models. IMPORTANT: This selection represents the baseline model checkpoint prior to hyperparameter tuning. It does NOT represent the final production deployment model, which will be determined after the planned hyperparameter optimization phase.

## 5. IEEE Experimental Results Table (LaTeX Snippet)

```latex
\begin{table}[h]
\centering
\caption{Baseline Machine Learning Model Performance Comparison}
\label{tab:model_comparison}
\begin{tabular}{lccccc}
\hline
Model & Accuracy & Precision & Recall & F1-Score & ROC-AUC \\
\hline
Logistic Regression & 0.8618 & 0.8400 & 0.9882 & 0.9081 & 0.8529 \\
Decision Tree & 0.8618 & 0.8542 & 0.9647 & 0.9061 & 0.7808 \\
Random Forest & 0.8537 & 0.8317 & 0.9882 & 0.9032 & 0.8161 \\
Xgboost & 0.8537 & 0.8384 & 0.9765 & 0.9022 & 0.7944 \\
Neural Network & 0.7154 & 0.7083 & 1.0000 & 0.8293 & 0.6093 \\
Ensemble & 0.8537 & 0.8317 & 0.9882 & 0.9032 & 0.8359 \\
\hline
\end{tabular}
\end{table}
```
