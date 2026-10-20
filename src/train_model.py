"""
Deep Learning Model Training Module (TensorFlow / Keras).
Implements, trains, and validates competing Multi-Layer Perceptron (MLP)
regression architectures using chronological validation sets.

Student: Kavin M (24AD053)
Institution: KPR Institute of Engineering and Technology
"""

import json
from pathlib import Path
from typing import Dict, Tuple
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

from src.data_loader import load_raw_dataset
from src.preprocessing import clean_air_quality_dataframe, prepare_modeling_data
from src.utils import get_project_root, ensure_directories, set_seed, calculate_regression_metrics


def build_baseline_mlp(input_dim: int, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Architecture 1: Baseline 2-hidden-layer MLP.
    Dense(64) -> Dropout(0.10) -> Dense(32) -> Dense(1)
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,), name="input_features"),
        layers.Dense(64, activation="relu", name="dense_hidden_1"),
        layers.Dropout(0.10, name="dropout_1"),
        layers.Dense(32, activation="relu", name="dense_hidden_2"),
        layers.Dense(1, activation="linear", name="output_regression")
    ], name="Baseline_MLP_2Layer")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def build_deep_mlp(input_dim: int, learning_rate: float = 0.001) -> tf.keras.Model:
    """
    Architecture 2: Deeper 3-hidden-layer Regularized MLP.
    Dense(128) -> Dropout(0.20) -> Dense(64) -> Dropout(0.15) -> Dense(32) -> Dense(1)
    """
    model = models.Sequential([
        layers.Input(shape=(input_dim,), name="input_features"),
        layers.Dense(128, activation="relu", name="dense_hidden_1"),
        layers.Dropout(0.20, name="dropout_1"),
        layers.Dense(64, activation="relu", name="dense_hidden_2"),
        layers.Dropout(0.15, name="dropout_2"),
        layers.Dense(32, activation="relu", name="dense_hidden_3"),
        layers.Dense(1, activation="linear", name="output_regression")
    ], name="Deep_Regularized_MLP_3Layer")

    optimizer = optimizers.Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse", metrics=["mae"])
    return model


def train_and_compare_models(
    batch_size: int = 32,
    max_epochs: int = 120,
    patience: int = 15
) -> Tuple[tf.keras.Model, Dict, pd.DataFrame]:
    """
    Train competing MLP architectures on the training set, evaluate on
    validation set only, and select the top-performing configuration.
    """
    set_seed(42)
    root = get_project_root()
    dirs = ensure_directories(root)

    # Preprocessing
    raw_df = load_raw_dataset()
    clean_df = clean_air_quality_dataframe(raw_df)
    prep_data = prepare_modeling_data(clean_df)

    X_train, y_train = prep_data["X_train"], prep_data["y_train"]
    X_val, y_val = prep_data["X_val"], prep_data["y_val"]
    input_dim = X_train.shape[1]

    early_stopping = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    configs = [
        {
            "name": "Model_A_Baseline_MLP",
            "builder": lambda: build_baseline_mlp(input_dim, learning_rate=0.001),
            "description": "Dense(64, ReLU) -> Dropout(0.10) -> Dense(32, ReLU) -> Linear"
        },
        {
            "name": "Model_B_Deep_Regularized_MLP",
            "builder": lambda: build_deep_mlp(input_dim, learning_rate=0.001),
            "description": "Dense(128, ReLU) -> Dropout(0.20) -> Dense(64, ReLU) -> Dropout(0.15) -> Dense(32, ReLU) -> Linear"
        }
    ]

    results = []
    trained_models = {}
    histories = {}

    for cfg in configs:
        model_name = cfg["name"]
        print(f"\n=======================================================")
        print(f"[Training] Starting training for: {model_name}")
        print(f"[Training] Architecture: {cfg['description']}")
        print(f"=======================================================")

        set_seed(42)  # Reset seed before each model build
        model = cfg["builder"]()

        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=max_epochs,
            batch_size=batch_size,
            callbacks=[early_stopping],
            verbose=1
        )

        val_preds = model.predict(X_val).ravel()
        val_metrics = calculate_regression_metrics(y_val, val_preds)

        train_preds = model.predict(X_train).ravel()
        train_metrics = calculate_regression_metrics(y_train, train_preds)

        results.append({
            "Model": model_name,
            "Architecture": cfg["description"],
            "StoppedEpoch": len(history.history["loss"]),
            "Train_MAE": round(train_metrics["MAE"], 4),
            "Train_RMSE": round(train_metrics["RMSE"], 4),
            "Train_R2": round(train_metrics["R2"], 4),
            "Val_MAE": round(val_metrics["MAE"], 4),
            "Val_RMSE": round(val_metrics["RMSE"], 4),
            "Val_MSE": round(val_metrics["MSE"], 4),
            "Val_R2": round(val_metrics["R2"], 4)
        })

        trained_models[model_name] = model
        histories[model_name] = history.history

    comparison_df = pd.DataFrame(results)
    print("\n[Training] Model Comparison Summary on Validation Data:")
    print(comparison_df[["Model", "StoppedEpoch", "Val_MAE", "Val_RMSE", "Val_R2"]])

    # Save comparison table
    comparison_df.to_csv(dirs["root"] / "outputs" / "model_comparison.csv", index=False)
    comparison_df.to_csv(dirs["tables"] / "model_comparison.csv", index=False)

    # Select best model based strictly on validation MAE
    best_row = comparison_df.sort_values("Val_MAE").iloc[0]
    best_model_name = best_row["Model"]
    best_model = trained_models[best_model_name]
    best_history = histories[best_model_name]

    print(f"\n[Training] Selected Best Model based on Validation MAE: {best_model_name}")
    print(f"[Training] Best Model Val MAE: {best_row['Val_MAE']} | Val RMSE: {best_row['Val_RMSE']} | Val R²: {best_row['Val_R2']}")

    # Save the selected model
    model_save_path = dirs["models"] / "best_mlp_model.keras"
    best_model.save(model_save_path)
    print(f"[Training] Saved best model to: {model_save_path}")

    # Save training history as JSON
    history_path = dirs["tables"] / "best_model_training_history.json"
    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(best_history, f, indent=2)

    return best_model, prep_data, comparison_df


if __name__ == "__main__":
    train_and_compare_models()
