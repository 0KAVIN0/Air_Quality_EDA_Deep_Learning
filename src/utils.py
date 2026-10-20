"""
Utility functions for reproducibility, project paths, and metric evaluations.
Student: Kavin M (24AD053)
Institution: KPR Institute of Engineering and Technology
"""

import os
import random
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    return Path(__file__).resolve().parent.parent


def set_seed(seed: int = 42) -> None:
    """
    Set random seeds across Python, NumPy, and TensorFlow for reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"
    tf.random.set_seed(seed)


def ensure_directories(root_dir: Path = None) -> dict:
    """Ensure all required project output directories exist and return their paths."""
    if root_dir is None:
        root_dir = get_project_root()

    paths = {
        "root": root_dir,
        "data": root_dir / "data",
        "notebooks": root_dir / "notebooks",
        "models": root_dir / "models",
        "outputs": root_dir / "outputs",
        "figures_eda": root_dir / "outputs" / "figures" / "eda",
        "figures_eval": root_dir / "outputs" / "figures" / "evaluation",
        "tables": root_dir / "outputs" / "tables",
        "interactive": root_dir / "outputs" / "interactive",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def calculate_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute standard regression metrics on original scale.
    Metrics: MAE, RMSE, MSE, and R^2.
    """
    y_t = np.asarray(y_true).ravel()
    y_p = np.asarray(y_pred).ravel()

    mae = float(mean_absolute_error(y_t, y_p))
    mse = float(mean_squared_error(y_t, y_p))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_t, y_p))

    return {
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2,
        "N_samples": int(len(y_t))
    }
