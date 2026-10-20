"""
Model Evaluation and Feature Importance Module.
Performs final evaluation on held-out test data, generates required
regression diagnostic visualizations, and conducts permutation feature importance.

Student: Kavin M (24AD053)
Institution: KPR Institute of Engineering and Technology
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from src.data_loader import load_raw_dataset
from src.preprocessing import clean_air_quality_dataframe, prepare_modeling_data
from src.utils import get_project_root, ensure_directories, calculate_regression_metrics, set_seed

# Plot formatting
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "figure.titlesize": 14,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})


def plot_loss_curves(history: dict, out_path: Path):
    """Plot 1: Training and Validation Loss (MSE) and MAE across epochs."""
    epochs = range(1, len(history["loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Loss curve (MSE)
    ax1.plot(epochs, history["loss"], label="Training Loss (MSE)", color="#0275d8", linewidth=2.0)
    ax1.plot(epochs, history["val_loss"], label="Validation Loss (MSE)", color="#d9534f", linewidth=2.0, linestyle="--")
    ax1.set_xlabel("Epoch Number")
    ax1.set_ylabel("Mean Squared Error (MSE) [(mg/m³)²]")
    ax1.set_title("Training vs Validation Loss", fontweight="bold")
    ax1.legend(loc="upper right")

    # Metric curve (MAE)
    ax2.plot(epochs, history["mae"], label="Training MAE", color="#0275d8", linewidth=2.0)
    ax2.plot(epochs, history["val_mae"], label="Validation MAE", color="#d9534f", linewidth=2.0, linestyle="--")
    ax2.set_xlabel("Epoch Number")
    ax2.set_ylabel("Mean Absolute Error (MAE) [mg/m³]")
    ax2.set_title("Training vs Validation MAE", fontweight="bold")
    ax2.legend(loc="upper right")

    plt.suptitle("Figure 10: Training and Validation Learning Curves Showing Convergence", fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Evaluation] Saved Figure 10 to: {out_path}")


def plot_predicted_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, metrics: dict, out_path: Path):
    """Plot 2: Predicted vs Actual CO(GT) scatter plot with ideal identity line."""
    fig, ax = plt.subplots(figsize=(8.5, 6))

    ax.scatter(y_true, y_pred, alpha=0.35, color="#2b5c8f", s=25, edgecolors="none", label="Test Observations")

    min_val = min(float(y_true.min()), float(y_pred.min()))
    max_val = max(float(y_true.max()), float(y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], color="#d9534f", linestyle="--", linewidth=2.2, label="Ideal Line (y = x)")

    metrics_text = (
        f"Test Samples (N): {metrics['N_samples']}\n"
        f"MAE:  {metrics['MAE']:.3f} mg/m³\n"
        f"RMSE: {metrics['RMSE']:.3f} mg/m³\n"
        f"R²:   {metrics['R2']:.3f}"
    )
    ax.text(0.05, 0.75, metrics_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc"),
            fontsize=10.5, fontweight="bold")

    ax.set_xlabel("Ground-Truth Measured CO(GT) [mg/m³]")
    ax.set_ylabel("MLP Predicted CO(GT) [mg/m³]")
    ax.set_title("Figure 11: Predicted vs Actual Carbon Monoxide Concentration (Held-Out Test Set)", fontweight="bold")
    ax.legend(loc="lower right")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Evaluation] Saved Figure 11 to: {out_path}")


def plot_residual_distribution(residuals: np.ndarray, out_path: Path):
    """Plot 3: Residual distribution histogram with KDE and normality reference."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    sns.histplot(residuals, kde=True, color="#5bc0de", bins=35, edgecolor="black", alpha=0.6, ax=ax)
    mean_res = float(np.mean(residuals))
    std_res = float(np.std(residuals))

    ax.axvline(0, color="#d9534f", linestyle="--", linewidth=2.0, label="Zero Error Line (Residual = 0)")
    ax.axvline(mean_res, color="#2b5c8f", linestyle="-.", linewidth=2.0, label=f"Mean Error: {mean_res:+.3f}")

    stat_text = (
        f"Mean Error: {mean_res:+.3f} mg/m³\n"
        f"Std Dev:    {std_res:.3f} mg/m³\n"
        f"Skewness:   {pd.Series(residuals).skew():.2f}"
    )
    ax.text(0.70, 0.75, stat_text, transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="#cccccc"),
            fontsize=10)

    ax.set_xlabel("Residual Error (Actual - Predicted CO) [mg/m³]")
    ax.set_ylabel("Frequency (Count)")
    ax.set_title("Figure 12: Residual Distribution Checking Error Symmetry and Unbiasedness", fontweight="bold")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Evaluation] Saved Figure 12 to: {out_path}")


def plot_residuals_vs_predicted(y_pred: np.ndarray, residuals: np.ndarray, out_path: Path):
    """Plot 4: Residuals vs Predicted values plot checking homoscedasticity."""
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.scatter(y_pred, residuals, alpha=0.35, color="#6f42c1", s=25, edgecolors="none")
    ax.axhline(0, color="#d9534f", linestyle="--", linewidth=2.2, label="Zero Error Baseline")

    # Add rolling/moving mean of residuals
    sorted_idx = np.argsort(y_pred)
    res_series = pd.Series(residuals[sorted_idx])
    rolling_res = res_series.rolling(window=50, min_periods=20).mean()
    ax.plot(y_pred[sorted_idx], rolling_res, color="#ffc107", linewidth=2.5, label="Rolling Mean Error Trend")

    ax.set_xlabel("Predicted CO(GT) Concentration [mg/m³]")
    ax.set_ylabel("Residual Error (Actual - Predicted) [mg/m³]")
    ax.set_title("Figure 13: Residuals vs Predicted Values Examining Homoscedasticity", fontweight="bold")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Evaluation] Saved Figure 13 to: {out_path}")


def plot_test_chronological_series(test_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray, out_path: Path):
    """Plot 5: Chronological actual vs predicted time series over held-out test window."""
    fig, ax = plt.subplots(figsize=(15, 6))

    dates = test_df["DateTime"].values
    ax.plot(dates, y_true, color="#0b3c5d", alpha=0.75, linewidth=1.5, label="Actual Observed CO(GT)")
    ax.plot(dates, y_pred, color="#e36209", alpha=0.85, linewidth=1.5, linestyle="--", label="MLP Model Prediction")

    ax.set_xlabel("Chronological Test Timeline (Feb 2005 – Apr 2005)")
    ax.set_ylabel("CO(GT) Concentration [mg/m³]")
    ax.set_title("Figure 15: Chronological Test Set Sequence: Actual vs MLP Predicted CO(GT)", fontweight="bold")
    ax.legend(loc="upper right")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Evaluation] Saved Figure 15 to: {out_path}")


def compute_permutation_importance(
    model: tf.keras.Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5
) -> pd.DataFrame:
    """
    Compute permutation feature importance on validation data.
    Measures the increase in MAE when each feature column is randomly shuffled.
    """
    set_seed(42)
    baseline_preds = model.predict(X_val, verbose=0).ravel()
    baseline_mae = float(np.mean(np.abs(y_val - baseline_preds)))

    importance_results = []
    n_features = X_val.shape[1]

    for j in range(n_features):
        feat_name = feature_names[j]
        mae_increases = []

        for _ in range(n_repeats):
            X_val_perm = X_val.copy()
            # Randomly permute column j
            shuffled_col = np.random.permutation(X_val_perm[:, j])
            X_val_perm[:, j] = shuffled_col

            perm_preds = model.predict(X_val_perm, verbose=0).ravel()
            perm_mae = float(np.mean(np.abs(y_val - perm_preds)))
            mae_increases.append(perm_mae - baseline_mae)

        mean_inc = float(np.mean(mae_increases))
        std_inc = float(np.std(mae_increases))

        importance_results.append({
            "Feature": feat_name,
            "Baseline_MAE": round(baseline_mae, 4),
            "Mean_MAE_Increase": round(mean_inc, 4),
            "Std_MAE_Increase": round(std_inc, 4)
        })

    imp_df = pd.DataFrame(importance_results).sort_values("Mean_MAE_Increase", ascending=False).reset_index(drop=True)
    imp_df["Rank"] = imp_df.index + 1
    return imp_df


def plot_feature_importance(imp_df: pd.DataFrame, out_path: Path):
    """Plot 6: Permutation feature importance horizontal bar chart."""
    df_sorted = imp_df.sort_values("Mean_MAE_Increase", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6.5))
    bars = ax.barh(
        df_sorted["Feature"],
        df_sorted["Mean_MAE_Increase"],
        xerr=df_sorted["Std_MAE_Increase"],
        color="#2b5c8f",
        edgecolor="black",
        alpha=0.85,
        capsize=4
    )

    ax.set_xlabel("Validation MAE Degradation (Mean Increase in Error) [mg/m³]")
    ax.set_ylabel("Input Feature")
    ax.set_title("Figure 14: Permutation Feature Importance on Validation Set (Ranked Impact on Model Error)", fontweight="bold")

    for bar in bars:
        w = bar.get_width()
        ax.annotate(f"{w:.3f}", xy=(w + 0.005, bar.get_y() + bar.get_height() / 2),
                    va="center", ha="left", fontsize=9, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"[Evaluation] Saved Figure 14 to: {out_path}")


def run_full_evaluation():
    """Execute complete model evaluation, chart generation, and importance ranking."""
    root = get_project_root()
    dirs = ensure_directories(root)

    # Load data
    raw_df = load_raw_dataset()
    clean_df = clean_air_quality_dataframe(raw_df)
    prep_data = prepare_modeling_data(clean_df)

    X_val, y_val = prep_data["X_val"], prep_data["y_val"]
    X_test, y_test = prep_data["X_test"], prep_data["y_test"]
    feature_names = prep_data["feature_names"]
    test_df = prep_data["test_df"]

    # Load trained best model
    model_path = dirs["models"] / "best_mlp_model.keras"
    if not model_path.exists():
        raise FileNotFoundError(f"Trained model not found at: {model_path}. Please run train_model.py first.")

    model = tf.keras.models.load_model(model_path)
    print(f"[Evaluation] Successfully loaded model from: {model_path}")

    # Load training history
    history_path = dirs["tables"] / "best_model_training_history.json"
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)

    # 1. Plot loss curve
    fig_eval_dir = dirs["figures_eval"]
    plot_loss_curves(history, fig_eval_dir / "fig10_training_loss_curve.png")

    # 2. Evaluate on held-out test data
    y_test_pred = model.predict(X_test).ravel()
    test_metrics = calculate_regression_metrics(y_test, y_test_pred)
    residuals = y_test - y_test_pred

    print("\n=======================================================")
    print("[Evaluation] FINAL TEST SET METRICS (Held-out Test Data):")
    print(f"  Test Observations: {test_metrics['N_samples']}")
    print(f"  MAE:               {test_metrics['MAE']:.4f} mg/m³")
    print(f"  RMSE:              {test_metrics['RMSE']:.4f} mg/m³")
    print(f"  MSE:               {test_metrics['MSE']:.4f} (mg/m³)²")
    print(f"  R² Score:          {test_metrics['R2']:.4f}")
    print("=======================================================\n")

    # Save metrics JSON & CSV
    metrics_path = dirs["root"] / "outputs" / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    pd.DataFrame([test_metrics]).to_csv(dirs["tables"] / "test_metrics.csv", index=False)

    # 3. Diagnostic plots
    plot_predicted_vs_actual(y_test, y_test_pred, test_metrics, fig_eval_dir / "fig11_predicted_vs_actual.png")
    plot_residual_distribution(residuals, fig_eval_dir / "fig12_residual_distribution.png")
    plot_residuals_vs_predicted(y_test_pred, residuals, fig_eval_dir / "fig13_residuals_vs_predicted.png")
    plot_test_chronological_series(test_df, y_test, y_test_pred, fig_eval_dir / "fig15_test_chronological_series.png")

    # 4. Permutation Feature Importance
    print("[Evaluation] Computing Permutation Feature Importance on Validation Data...")
    imp_df = compute_permutation_importance(model, X_val, y_val, feature_names, n_repeats=5)
    print("\n[Evaluation] Feature Importance Ranking:")
    print(imp_df[["Rank", "Feature", "Mean_MAE_Increase", "Std_MAE_Increase"]])

    imp_df.to_csv(dirs["tables"] / "feature_importance.csv", index=False)
    plot_feature_importance(imp_df, fig_eval_dir / "fig14_feature_importance.png")

    # Save full run summary
    run_summary = {
        "student": {
            "name": "Kavin M",
            "roll_number": "24AD053",
            "college": "KPR Institute of Engineering and Technology",
            "department": "Artificial Intelligence and Data Science",
            "course": "U21ADP05 - Exploratory Data Analysis and Visualization",
            "academic_year": "2026-2027",
            "faculty": "Mr. Rushikesh Kadam"
        },
        "dataset": {
            "source": "UCI Air Quality Dataset",
            "doi": "10.24432/C59K5F",
            "raw_samples": prep_data["total_records"],
            "labeled_samples": prep_data["labeled_count"],
            "excluded_missing_target": prep_data["missing_target_count"],
            "train_samples": len(prep_data["train_df"]),
            "val_samples": len(prep_data["val_df"]),
            "test_samples": len(prep_data["test_df"])
        },
        "test_metrics": test_metrics,
        "feature_ranking": imp_df.to_dict(orient="records")
    }
    with open(dirs["root"] / "outputs" / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    print(f"[Evaluation] All evaluation outputs and run summary successfully created.")


if __name__ == "__main__":
    run_full_evaluation()
