"""
Data Loader and Data Quality Inspection Module.
Downloads and loads the official UCI Air Quality dataset,
performs initial validation, and exports data quality summaries.

Student: Kavin M (24AD053)
Institution: KPR Institute of Engineering and Technology
"""

import json
import zipfile
import io
from pathlib import Path
import requests
import pandas as pd
import numpy as np

from src.utils import get_project_root, ensure_directories

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/360/air+quality.zip"
SENTINEL_MISSING = -200.0


def download_dataset_if_missing(data_dir: Path) -> Path:
    """
    Download official UCI Air Quality dataset archive if CSV is not already present.
    Returns path to CSV file.
    """
    csv_path = data_dir / "AirQualityUCI.csv"
    if csv_path.exists():
        print(f"[DataLoader] Dataset already exists at: {csv_path}")
        return csv_path

    print(f"[DataLoader] Downloading official dataset from: {UCI_DATASET_URL}")
    data_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(UCI_DATASET_URL, timeout=60)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
        z.extractall(data_dir)
    print(f"[DataLoader] Extracted dataset archive to: {data_dir}")
    return csv_path


def load_raw_dataset(data_path: Path = None) -> pd.DataFrame:
    """
    Load raw CSV file handling European decimal (',') and semicolon (';') delimiters.
    Removes trailing empty columns and trailing unrecorded blank lines.
    """
    if data_path is None:
        root = get_project_root()
        data_path = root / "data" / "AirQualityUCI.csv"

    if not data_path.exists():
        data_path = download_dataset_if_missing(data_path.parent)

    # Read CSV
    df = pd.read_csv(data_path, sep=";", decimal=",", encoding="latin1")

    # Drop unneeded trailing blank columns
    unnamed_cols = [c for c in df.columns if "Unnamed" in c]
    if unnamed_cols:
        df = df.drop(columns=unnamed_cols)

    # Drop trailing empty rows (where Date is null)
    df = df[df["Date"].notnull()].copy()

    return df


def inspect_data_quality(df: pd.DataFrame, output_dir: Path = None) -> dict:
    """
    Compute comprehensive data quality inspection summary and save to outputs/tables.
    """
    if output_dir is None:
        root = get_project_root()
        output_dir = root / "outputs" / "tables"
    output_dir.mkdir(parents=True, exist_ok=True)

    total_records = len(df)
    columns_info = {}

    for col in df.columns:
        if col in ["Date", "Time"]:
            null_count = int(df[col].isnull().sum())
            columns_info[col] = {
                "dtype": str(df[col].dtype),
                "null_count": null_count,
                "sentinel_count": 0,
                "effective_missing_count": null_count,
                "missing_percentage": round((null_count / total_records) * 100, 2),
            }
        else:
            null_count = int(df[col].isnull().sum())
            sentinel_count = int((df[col] == SENTINEL_MISSING).sum())
            total_missing = null_count + sentinel_count
            columns_info[col] = {
                "dtype": str(df[col].dtype),
                "null_count": null_count,
                "sentinel_count": sentinel_count,
                "effective_missing_count": total_missing,
                "missing_percentage": round((total_missing / total_records) * 100, 2),
                "min": float(df[col][df[col] != SENTINEL_MISSING].min()) if total_missing < total_records else None,
                "max": float(df[col][df[col] != SENTINEL_MISSING].max()) if total_missing < total_records else None,
                "mean": float(df[col][df[col] != SENTINEL_MISSING].mean()) if total_missing < total_records else None,
            }

    duplicate_rows = int(df.duplicated(subset=["Date", "Time"]).sum())

    summary = {
        "dataset_name": "UCI Air Quality Dataset",
        "dataset_doi": "10.24432/C59K5F",
        "official_url": UCI_DATASET_URL,
        "total_valid_rows": total_records,
        "total_columns": int(df.shape[1]),
        "duplicate_timestamps": duplicate_rows,
        "target_variable": "CO(GT)",
        "target_missing_count": columns_info["CO(GT)"]["effective_missing_count"],
        "target_valid_count": total_records - columns_info["CO(GT)"]["effective_missing_count"],
        "columns": columns_info
    }

    # Save summary as JSON
    json_path = output_dir / "data_quality_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Save summary as tabular CSV
    summary_rows = []
    for col, info in columns_info.items():
        summary_rows.append({
            "Feature": col,
            "DataType": info["dtype"],
            "EffectiveMissing": info["effective_missing_count"],
            "MissingPct": info["missing_percentage"],
            "Min": info.get("min", "-"),
            "Max": info.get("max", "-"),
            "Mean": round(info.get("mean", 0), 2) if info.get("mean") is not None else "-"
        })
    summary_df = pd.DataFrame(summary_rows)
    csv_path = output_dir / "data_quality_summary.csv"
    summary_df.to_csv(csv_path, index=False)

    print(f"[DataLoader] Data quality summary saved to: {json_path} and {csv_path}")
    return summary


if __name__ == "__main__":
    ensure_directories()
    raw_df = load_raw_dataset()
    print(f"[DataLoader] Successfully loaded dataset. Shape: {raw_df.shape}")
    summary = inspect_data_quality(raw_df)
    print(f"[DataLoader] Total valid observations: {summary['total_valid_rows']}")
    print(f"[DataLoader] Target CO(GT) labelled observations: {summary['target_valid_count']}")
