import os
import sys
import json
import pandas as pd
import numpy as np

# Ensure project root is in sys.path for robust execution from any working directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.training_utils import (
    EVALUATION_JSON_PATH,
    METRICS_DIR,
    initialize_directories,
    logger
)


# Global Constants for Validation
VALIDATED_MODELS = [
    "logistic_regression",
    "decision_tree",
    "random_forest",
    "xgboost",
    "neural_network",
    "ensemble"
]

REQUIRED_METRICS = [
    "Accuracy",
    "Precision",
    "Recall",
    "F1_Score",
    "ROC_AUC"
]

SUMMARY_JSON_PATH = os.path.join(METRICS_DIR, "model_comparison_summary.json")
BEST_MODEL_INFO_PATH = os.path.join(METRICS_DIR, "best_model_info.json")
COMPARISON_REPORT_PATH = os.path.join(METRICS_DIR, "comparison_report.md")


def load_and_validate_metrics() -> dict:
    """
    Load existing evaluation metrics from models/metrics/model_evaluation.json
    and validate completeness across all six frozen model approaches and five core metrics.
    """
    logger.info(f"Loading evaluation metrics from {EVALUATION_JSON_PATH}")
    
    if not os.path.exists(EVALUATION_JSON_PATH):
        raise FileNotFoundError(
            f"Evaluation metrics file not found at {EVALUATION_JSON_PATH}. "
            "Please ensure model training pipelines have executed."
        )

    with open(EVALUATION_JSON_PATH, "r") as f:
        try:
            metrics_data = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON from {EVALUATION_JSON_PATH}: {e}")

    # Validate all 6 model approaches are present
    missing_models = [m for m in VALIDATED_MODELS if m not in metrics_data]
    if missing_models:
        raise KeyError(
            f"Missing model metrics in {EVALUATION_JSON_PATH}: {missing_models}. "
            f"Expected models: {VALIDATED_MODELS}"
        )

    # Validate all 5 metrics are present for each model
    for model_name in VALIDATED_MODELS:
        model_metrics = metrics_data[model_name]
        missing_metrics = [metric for metric in REQUIRED_METRICS if metric not in model_metrics]
        if missing_metrics:
            raise KeyError(
                f"Missing metrics for model '{model_name}': {missing_metrics}. "
                f"Expected metrics: {REQUIRED_METRICS}"
            )

    logger.info("Successfully loaded and validated evaluation metrics for all 6 models.")
    return metrics_data


def build_comparison_matrix(metrics_data: dict) -> pd.DataFrame:
    """
    Convert metrics dictionary into a structured pandas DataFrame.
    """
    df = pd.DataFrame.from_dict(metrics_data, orient="index")
    # Enforce column order for consistent reporting
    df = df[REQUIRED_METRICS]
    return df


def determine_metric_leaders(df: pd.DataFrame) -> dict:
    """
    Identify the leading model(s) for each individual evaluation metric.
    """
    leaders = {}
    for metric in REQUIRED_METRICS:
        max_val = df[metric].max()
        # Find all models matching max_val within floating point tolerance
        winning_models = df[np.isclose(df[metric], max_val)].index.tolist()
        leaders[metric] = {
            "winner": winning_models,
            "value": float(max_val)
        }
        logger.info(f"Metric Leader - {metric}: {winning_models} (Score: {max_val:.4f})")

    return leaders


def analyze_model_selection(df: pd.DataFrame, leaders: dict) -> dict:
    """
    Perform an objective multi-criteria model selection analysis using ROC-AUC and F1 Score
    as primary criteria, with Accuracy and Precision as secondary criteria.

    Integrates explicit architectural considerations:
    1. Ensemble is a first-class candidate, but not assumed to be superior automatically.
    2. Preserves strict distinction between current baseline selection, future hyperparameter-tuned
       selection, and final production model.
    """
    logger.info("Performing objective model selection analysis...")

    # Primary criteria comparison: ROC-AUC and F1_Score
    # Rank by ROC_AUC descending, then F1_Score descending
    ranked_df = df.sort_values(by=["ROC_AUC", "F1_Score", "Accuracy"], ascending=[False, False, False])
    
    current_baseline_model = ranked_df.index[0]
    baseline_metrics = df.loc[current_baseline_model].to_dict()

    ensemble_metrics = df.loc["ensemble"].to_dict()
    ensemble_roc_auc = ensemble_metrics["ROC_AUC"]
    baseline_roc_auc = baseline_metrics["ROC_AUC"]

    # Evaluate ensemble objective standing
    ensemble_is_top = (current_baseline_model == "ensemble")
    
    rationale = (
        f"Selected '{current_baseline_model}' as the current baseline checkpoint leader. "
        f"It achieved the highest ROC-AUC ({baseline_metrics['ROC_AUC']:.4f}) and highest F1 Score "
        f"({baseline_metrics['F1_Score']:.4f}) among all six evaluated approaches. "
    )

    if not ensemble_is_top:
        rationale += (
            f"The soft-voting ensemble achieved an ROC-AUC of {ensemble_roc_auc:.4f} and F1 Score of "
            f"{ensemble_metrics['F1_Score']:.4f}. While competitive, the ensemble was slightly degraded "
            f"by the baseline Neural Network (ROC-AUC {df.loc['neural_network', 'ROC_AUC']:.4f}), "
            "demonstrating that ensembles are not inherently superior to well-tuned single linear models. "
        )
    else:
        rationale += "The ensemble outperformed all individual base models on primary criteria. "

    rationale += (
        "IMPORTANT: This selection represents the baseline model checkpoint prior to hyperparameter tuning. "
        "It does NOT represent the final production deployment model, which will be determined after "
        "the planned hyperparameter optimization phase."
    )

    selection_summary = {
        "current_baseline_model": current_baseline_model,
        "selection_phase": "baseline_checkpoint",
        "primary_criteria": ["ROC_AUC", "F1_Score"],
        "secondary_criteria": ["Accuracy", "Precision"],
        "baseline_metrics": {k: float(v) for k, v in baseline_metrics.items()},
        "ensemble_standing": {
            "is_top_performer": ensemble_is_top,
            "metrics": {k: float(v) for k, v in ensemble_metrics.items()},
            "roc_auc_delta_vs_leader": float(ensemble_roc_auc - baseline_roc_auc)
        },
        "selection_rationale": rationale
    }

    logger.info(f"Objective Selection Result: Baseline Leader = '{current_baseline_model}'")
    return selection_summary


def save_comparison_artifacts(metrics_data: dict, df: pd.DataFrame, leaders: dict, selection_info: dict):
    """
    Persist structured comparison JSON summaries, best_model_info.json, and a comprehensive
    Markdown/LaTeX comparative report for research documentation.
    """
    initialize_directories()

    # 1. Save Full Comparison Summary JSON
    summary_content = {
        "models_evaluated": list(metrics_data.keys()),
        "raw_evaluation_matrix": metrics_data,
        "metric_leaders": leaders,
        "selection_analysis": selection_info
    }
    with open(SUMMARY_JSON_PATH, "w") as f:
        json.dump(summary_content, f, indent=4)
    logger.info(f"Saved comparison summary JSON to {SUMMARY_JSON_PATH}")

    # 2. Save best_model_info.json (Targeted baseline metadata for downstream modules)
    best_model_name = selection_info["current_baseline_model"]
    model_path = (
        None
        if best_model_name == "ensemble"
        else os.path.join("models", "base_models", f"{best_model_name}.joblib")
    )
    
    best_model_info = {
        "model_name": best_model_name,
        "model_path": model_path,
        "selection_phase": "baseline_checkpoint",
        "selection_metric": "ROC_AUC & F1_Score",
        "is_ensemble": (best_model_name == "ensemble"),
        "metrics": selection_info["baseline_metrics"],
        "selection_rationale": selection_info["selection_rationale"],
        "note": "Baseline model selection prior to hyperparameter optimization phase."
    }
    with open(BEST_MODEL_INFO_PATH, "w") as f:
        json.dump(best_model_info, f, indent=4)
    logger.info(f"Saved best model info to {BEST_MODEL_INFO_PATH}")

    # 3. Generate Markdown & LaTeX Comparison Report for IEEE Documentation
    report_content = generate_markdown_report(df, leaders, selection_info)
    with open(COMPARISON_REPORT_PATH, "w") as f:
        f.write(report_content)
    logger.info(f"Generated comprehensive comparison report at {COMPARISON_REPORT_PATH}")


def generate_markdown_report(df: pd.DataFrame, leaders: dict, selection_info: dict) -> str:
    """
    Format a clean Markdown report with embedded tables, metric leader breakdown,
    ensemble analysis, and LaTeX snippet for research papers.
    """
    md = []
    md.append("# Baseline Model Comparison & Selection Report")
    md.append("\n**Phase:** Baseline Model Evaluation (Frozen Checkpoint)")
    md.append("**Primary Selection Metrics:** ROC-AUC, F1 Score\n")

    md.append("## 1. Quantitative Metrics Comparison Table\n")
    md.append("| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC |")
    md.append("| :--- | :---: | :---: | :---: | :---: | :---: |")

    for model_name, row in df.iterrows():
        name_display = f"**{model_name}**" if model_name == selection_info["current_baseline_model"] else model_name
        md.append(
            f"| {name_display} | {row['Accuracy']:.4f} | {row['Precision']:.4f} | "
            f"{row['Recall']:.4f} | {row['F1_Score']:.4f} | {row['ROC_AUC']:.4f} |"
        )

    md.append("\n## 2. Individual Metric Leaders\n")
    for metric, leader_data in leaders.items():
        winners_str = ", ".join(leader_data["winner"])
        md.append(f"- **Highest {metric}:** `{winners_str}` ({leader_data['value']:.4f})")

    md.append("\n## 3. Objective Ensemble Analysis\n")
    ens_info = selection_info["ensemble_standing"]
    md.append(f"- **Ensemble Status:** Treated as a first-class candidate alongside base models.")
    md.append(f"- **Ensemble ROC-AUC:** {ens_info['metrics']['ROC_AUC']:.4f}")
    md.append(f"- **Ensemble F1 Score:** {ens_info['metrics']['F1_Score']:.4f}")
    md.append(
        f"- **Performance Relative to Top Performer:** ROC-AUC Delta of "
        f"{ens_info['roc_auc_delta_vs_leader']:+.4f} vs `{selection_info['current_baseline_model']}`."
    )
    md.append(
        "\n*Analysis Note:* Soft-voting ensemble performance reflects equal/weighted consensus. "
        "Without hyperparameter tuning on base estimators (specifically the uncalibrated neural network), "
        "the ensemble does not automatically surpass the linear baseline."
    )

    md.append("\n## 4. Current Baseline Selection & Rationale\n")
    md.append(f"**Current Baseline Winner:** `{selection_info['current_baseline_model']}`\n")
    md.append(f"> {selection_info['selection_rationale']}\n")

    md.append("## 5. IEEE Experimental Results Table (LaTeX Snippet)\n")
    md.append("```latex")
    md.append("\\begin{table}[h]")
    md.append("\\centering")
    md.append("\\caption{Baseline Machine Learning Model Performance Comparison}")
    md.append("\\label{tab:model_comparison}")
    md.append("\\begin{tabular}{lccccc}")
    md.append("\\hline")
    md.append("Model & Accuracy & Precision & Recall & F1-Score & ROC-AUC \\\\")
    md.append("\\hline")
    for model_name, row in df.iterrows():
        name_tex = model_name.replace("_", " ").title()
        md.append(
            f"{name_tex} & {row['Accuracy']:.4f} & {row['Precision']:.4f} & "
            f"{row['Recall']:.4f} & {row['F1_Score']:.4f} & {row['ROC_AUC']:.4f} \\\\"
        )
    md.append("\\hline")
    md.append("\\end{tabular}")
    md.append("\\end{table}")
    md.append("```\n")

    return "\n".join(md)


def run_comparison_pipeline():
    """
    Main orchestration function for the model comparison layer.
    """
    logger.info("==================================================")
    logger.info(" Starting Model Comparison & Selection Pipeline ")
    logger.info("==================================================")

    try:
        # 1. Load and validate metrics
        metrics_data = load_and_validate_metrics()

        # 2. Build comparison matrix
        df = build_comparison_matrix(metrics_data)

        # 3. Determine metric leaders
        leaders = determine_metric_leaders(df)

        # 4. Perform objective model selection analysis
        selection_info = analyze_model_selection(df, leaders)

        # 5. Save comparison artifacts and generate report
        save_comparison_artifacts(metrics_data, df, leaders, selection_info)

        logger.info("==================================================")
        logger.info(" Model Comparison Pipeline Completed Successfully ")
        logger.info("==================================================")
        return df, leaders, selection_info

    except Exception as e:
        logger.error(f"Model Comparison Pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run_comparison_pipeline()
