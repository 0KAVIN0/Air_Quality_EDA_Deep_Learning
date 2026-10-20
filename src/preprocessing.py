"""
Preprocessing and Feature Engineering Pipeline.
Implements chronological splitting, target leakage prevention,
cyclical temporal engineering, and train-only imputation & standardization.

Student: Kavin M (24AD053)
Institution: KPR Institute of Engineering and Technology
"""

from typing import Tuple, List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data_loader import load_raw_dataset, SENTINEL_MISSING

# Features to use as inputs (Sensor responses + Meteorological telemetry)
RAW_PREDICTOR_COLS = [
    "PT08.S1(CO)",
    "PT08.S2(NMHC)",
    "PT08.S3(NOx)",
    "PT08.S4(NO2)",
    "PT08.S5(O3)",
    "T",
    "RH",
    "AH"
]

# Ground truth columns to exclude from inputs to prevent target leakage
LEAKAGE_COLS = [
    "CO(GT)",      # Prediction target
    "NMHC(GT)",    # Ground truth reference
    "C6H6(GT)",    # Ground truth reference
    "NOx(GT)",     # Ground truth reference
    "NO2(GT)"      # Ground truth reference
]

TARGET_COL = "CO(GT)"


def clean_air_quality_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Parse timestamp, convert sentinel missing values (-200) to NaN,
    and sort chronologically.
    """
    df = df_raw.copy()

    # Parse DateTime
    df["DateTime"] = pd.to_datetime(
        df["Date"].astype(str) + " " + df["Time"].astype(str),
        format="%d/%m/%Y %H.%M.%S"
    )

    # Sort strictly by timestamp
    df = df.sort_values("DateTime").reset_index(drop=True)

    # Replace sentinel -200 with NaN across numeric fields
    numeric_cols = [c for c in df.columns if c not in ["Date", "Time", "DateTime"]]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].replace(SENTINEL_MISSING, np.nan)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal attributes and cyclical representations (sine/cosine)
    to capture diurnal and seasonal traffic and meteorological cycles.
    """
    df = df.copy()
    hour = df["DateTime"].dt.hour
    month = df["DateTime"].dt.month
    day_of_week = df["DateTime"].dt.dayofweek

    df["hour"] = hour
    df["month"] = month
    df["day_of_week"] = day_of_week
    df["is_weekend"] = (day_of_week >= 5).astype(int)

    # Cyclical hour encoding (24-hour cycle)
    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)

    # Cyclical month encoding (12-month cycle)
    df["month_sin"] = np.sin(2 * np.pi * (month - 1) / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * (month - 1) / 12.0)

    return df


def prepare_modeling_data(
    df: pd.DataFrame,
    train_frac: float = 0.70,
    val_frac: float = 0.15
) -> Dict:
    """
    Execute end-to-end preprocessing pipeline for deep learning regression:
    1. Filter out records where target CO(GT) is missing (no fake target imputation).
    2. Engineer cyclical temporal features.
    3. Chronological train (70%) / val (15%) / test (15%) partition.
    4. Fit median imputer on training predictors only; transform val and test.
    5. Fit StandardScaler on training predictors only; transform val and test.
    """
    total_records = len(df)
    missing_target_count = int(df[TARGET_COL].isnull().sum())

    # Exclude missing target records
    df_labeled = df[df[TARGET_COL].notnull()].copy()
    labeled_count = len(df_labeled)
    print(f"[Preprocessing] Total records: {total_records}")
    print(f"[Preprocessing] Excluded missing target {TARGET_COL} rows: {missing_target_count}")
    print(f"[Preprocessing] Usable labeled observations: {labeled_count}")

    # Feature engineering
    df_labeled = engineer_features(df_labeled)

    # Define model feature columns (strictly avoiding target and ground truth leakage)
    feature_cols = RAW_PREDICTOR_COLS + [
        "hour_sin", "hour_cos",
        "month_sin", "month_cos",
        "day_of_week", "is_weekend"
    ]

    # Chronological partition
    train_end = int(labeled_count * train_frac)
    val_end = int(labeled_count * (train_frac + val_frac))

    train_df = df_labeled.iloc[:train_end].copy()
    val_df = df_labeled.iloc[train_end:val_end].copy()
    test_df = df_labeled.iloc[val_end:].copy()

    print(f"[Preprocessing] Train split: {len(train_df)} rows ({len(train_df)/labeled_count*100:.1f}%) | "
          f"{train_df['DateTime'].min()} -> {train_df['DateTime'].max()}")
    print(f"[Preprocessing] Val split:   {len(val_df)} rows ({len(val_df)/labeled_count*100:.1f}%) | "
          f"{val_df['DateTime'].min()} -> {val_df['DateTime'].max()}")
    print(f"[Preprocessing] Test split:  {len(test_df)} rows ({len(test_df)/labeled_count*100:.1f}%) | "
          f"{test_df['DateTime'].min()} -> {test_df['DateTime'].max()}")

    # Fit Imputer strictly on train set
    imputer = SimpleImputer(strategy="median")
    X_train_raw = train_df[feature_cols].values
    X_val_raw = val_df[feature_cols].values
    X_test_raw = test_df[feature_cols].values

    X_train_imp = imputer.fit_transform(X_train_raw)
    X_val_imp = imputer.transform(X_val_raw)
    X_test_imp = imputer.transform(X_test_raw)

    # Fit Scaler strictly on train set
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_imp)
    X_val_scaled = scaler.transform(X_val_imp)
    X_test_scaled = scaler.transform(X_test_imp)

    y_train = train_df[TARGET_COL].values.astype(np.float32)
    y_val = val_df[TARGET_COL].values.astype(np.float32)
    y_test = test_df[TARGET_COL].values.astype(np.float32)

    return {
        "X_train": X_train_scaled,
        "y_train": y_train,
        "X_val": X_val_scaled,
        "y_val": y_val,
        "X_test": X_test_scaled,
        "y_test": y_test,
        "feature_names": feature_cols,
        "train_df": train_df,
        "val_df": val_df,
        "test_df": test_df,
        "imputer": imputer,
        "scaler": scaler,
        "total_records": total_records,
        "missing_target_count": missing_target_count,
        "labeled_count": labeled_count
    }


if __name__ == "__main__":
    raw_df = load_raw_dataset()
    cleaned_df = clean_air_quality_dataframe(raw_df)
    prep_data = prepare_modeling_data(cleaned_df)
    print(f"[Preprocessing] Features ({len(prep_data['feature_names'])}): {prep_data['feature_names']}")
    print(f"[Preprocessing] X_train shape: {prep_data['X_train'].shape}")
    print(f"[Preprocessing] X_val shape:   {prep_data['X_val'].shape}")
    print(f"[Preprocessing] X_test shape:  {prep_data['X_test'].shape}")
