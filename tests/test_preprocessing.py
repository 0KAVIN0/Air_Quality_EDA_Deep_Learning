"""
Automated unit tests for data loading and preprocessing pipeline.
Verifies no data leakage, strict chronological ordering, correct shapes,
and imputer/scaler train-only fitting.

Student: Kavin M (24AD053)
Institution: KPR Institute of Engineering and Technology
"""

import unittest
import numpy as np
import pandas as pd
from pathlib import Path

from src.data_loader import load_raw_dataset, SENTINEL_MISSING
from src.preprocessing import (
    clean_air_quality_dataframe,
    engineer_features,
    prepare_modeling_data,
    RAW_PREDICTOR_COLS,
    LEAKAGE_COLS,
    TARGET_COL
)


class TestAirQualityPreprocessing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw_df = load_raw_dataset()
        cls.cleaned_df = clean_air_quality_dataframe(cls.raw_df)
        cls.prep_data = prepare_modeling_data(cls.cleaned_df)

    def test_raw_shape_and_records(self):
        """Verify the raw dataset has expected UCI record count (9357 rows)."""
        self.assertEqual(len(self.raw_df), 9357)
        self.assertNotIn("Unnamed: 15", self.raw_df.columns)
        self.assertNotIn("Unnamed: 16", self.raw_df.columns)

    def test_sentinel_replacement(self):
        """Ensure -200 sentinels are converted to NaN in cleaned data."""
        numeric_cols = [c for c in self.cleaned_df.columns if c not in ["Date", "Time", "DateTime"]]
        for col in numeric_cols:
            sentinel_count = (self.cleaned_df[col] == SENTINEL_MISSING).sum()
            self.assertEqual(sentinel_count, 0, f"Found -200 sentinel in cleaned column {col}")

    def test_chronological_ordering(self):
        """Ensure DateTime is strictly monotonic increasing."""
        self.assertTrue(self.cleaned_df["DateTime"].is_monotonic_increasing)

    def test_target_missingness_dropped(self):
        """Ensure rows where CO(GT) is missing are strictly excluded from modeling."""
        self.assertEqual(self.prep_data["missing_target_count"], 1683)
        self.assertEqual(self.prep_data["labeled_count"], 7674)
        self.assertFalse(np.isnan(self.prep_data["y_train"]).any())
        self.assertFalse(np.isnan(self.prep_data["y_val"]).any())
        self.assertFalse(np.isnan(self.prep_data["y_test"]).any())

    def test_target_leakage_prevention(self):
        """Confirm that target and all other ground-truth pollutants are excluded from inputs."""
        features = self.prep_data["feature_names"]
        for leak_col in LEAKAGE_COLS:
            self.assertNotIn(
                leak_col, features,
                f"Data leakage detected! Ground-truth feature '{leak_col}' is present in model inputs."
            )

    def test_cyclical_feature_ranges(self):
        """Check that sine and cosine temporal features are properly bounded within [-1, 1]."""
        engineered = engineer_features(self.cleaned_df)
        for col in ["hour_sin", "hour_cos", "month_sin", "month_cos"]:
            self.assertTrue((engineered[col] >= -1.0 - 1e-5).all())
            self.assertTrue((engineered[col] <= 1.0 + 1e-5).all())

    def test_no_nans_in_processed_splits(self):
        """Check that imputation successfully resolved all missing values in feature matrices."""
        self.assertFalse(np.isnan(self.prep_data["X_train"]).any())
        self.assertFalse(np.isnan(self.prep_data["X_val"]).any())
        self.assertFalse(np.isnan(self.prep_data["X_test"]).any())

    def test_chronological_split_integrity(self):
        """Verify train, validation, and test splits follow strict temporal sequence without overlap."""
        train_max_date = self.prep_data["train_df"]["DateTime"].max()
        val_min_date = self.prep_data["val_df"]["DateTime"].min()
        val_max_date = self.prep_data["val_df"]["DateTime"].max()
        test_min_date = self.prep_data["test_df"]["DateTime"].min()

        self.assertLess(train_max_date, val_min_date)
        self.assertLess(val_max_date, test_min_date)

    def test_standard_scaler_train_only(self):
        """Verify training inputs have approximately zero mean and unit variance."""
        train_means = np.mean(self.prep_data["X_train"], axis=0)
        train_stds = np.std(self.prep_data["X_train"], axis=0)
        np.testing.assert_allclose(train_means, 0.0, atol=1e-4)
        np.testing.assert_allclose(train_stds, 1.0, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
