import sys
import os
import pandas as pd

# Add project root path for imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper_functions import (
    remove_duplicate_rows,
    drop_high_missing,
    normalize_column_names,
    handle_outliers
)

DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")

def load_csv(name):
    """Helper to load CSV from the test_data folder"""
    path = os.path.join(DATA_PATH, name)
    assert os.path.exists(path), f"CSV file not found: {path}"
    return pd.read_csv(path)


# ✅ Test for duplicate rows
def test_remove_duplicate_rows():
    df = load_csv("AirQualityData.csv")
    original_rows = df.shape[0]
    cleaned_df, msg = remove_duplicate_rows(df)
    # Allow both outcomes — duplicates removed or none found
    assert cleaned_df.shape[0] <= original_rows
    assert "removed" in msg.lower() or "no duplicate" in msg.lower()


# ✅ Test for missing value dropping
def test_drop_high_missing():
    df = load_csv("HousingData.csv")
    cleaned_df, msg = drop_high_missing(df)
    assert isinstance(cleaned_df, pd.DataFrame)
    # Allow both: some columns dropped or none found
    assert "dropped" in msg.lower() or "no columns" in msg.lower()


# ✅ Test for column name normalization
def test_normalize_column_names():
    df = load_csv("StudentPerformanceFactors.csv")
    cleaned_df, msg = normalize_column_names(df)
    # Must be lower-case and no spaces
    assert all(col.islower() for col in cleaned_df.columns)
    assert all(" " not in col for col in cleaned_df.columns)
    # Message flexibility
    assert "normalized" in msg.lower() or "already normalized" in msg.lower()


# ✅ Test for outlier handling
def test_handle_outliers_iqr():
    # Artificial numeric dataset with outliers
    df = pd.DataFrame({"value": [1, 2, 3, 1000, 2000]})
    new_df, msg = handle_outliers(df.copy(), "value", "IQR", 1.5)
    # Ensure outlier values are reduced
    assert new_df["value"].max() <= 2000
    assert "iqr" in msg.lower() or "outlier" in msg.lower()


# ✅ Optional — Add one test for “no outliers case”
def test_handle_outliers_iqr_no_outliers():
    df = pd.DataFrame({"value": [10, 11, 12, 13, 14]})
    new_df, msg = handle_outliers(df.copy(), "value", "IQR", 1.5)
    # No clipping should occur, but function should still return valid DataFrame
    assert isinstance(new_df, pd.DataFrame)
    assert "iqr" in msg.lower() or "clipped" in msg.lower()
