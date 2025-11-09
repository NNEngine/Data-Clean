import os
import pandas as pd
import sys

# Add project root (one directory up) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper_functions import (
    remove_duplicate_rows,
    remove_duplicate_columns,
    drop_high_missing,
    normalize_column_names,
    handle_outliers
)

# Get CSV path dynamically
DATA_PATH = os.path.join(os.path.dirname(__file__), "test_data")

def load_csv(name):
    """Helper to load CSV from the test_data folder"""
    path = os.path.join(DATA_PATH, name)
    return pd.read_csv(path)


def test_remove_duplicate_rows():
    df = load_csv("AirQualityData.csv")
    original_rows = df.shape[0]
    cleaned_df, msg = remove_duplicate_rows(df)
    # ✅ Pass if duplicates were removed OR none existed
    assert cleaned_df.shape[0] <= original_rows
    assert "removed" in msg.lower() or "no duplicate" in msg.lower()



def test_drop_high_missing():
    df = load_csv("HousingData.csv")
    cleaned_df, msg = drop_high_missing(df)
    assert isinstance(cleaned_df, pd.DataFrame)
    assert "dropped" in msg.lower() or "no columns" in msg.lower()


def test_normalize_column_names():
    df = load_csv("StudentPerformanceFactors.csv")
    cleaned_df, msg = normalize_column_names(df)
    assert all(col.islower() for col in cleaned_df.columns)
    assert all(" " not in col for col in cleaned_df.columns)
    assert "normalized" in msg.lower()


def test_handle_outliers_iqr():
    df = pd.DataFrame({"value": [1, 2, 3, 1000, 2000]})
    new_df, msg = handle_outliers(df.copy(), "value", "IQR", 1.5)
    assert new_df["value"].max() < 2000
    assert "iqr" in msg.lower()
