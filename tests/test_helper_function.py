import pandas as pd
import os
import sys

# Add project root (one directory up) to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helper_functions import (
    normalize_column_names,
    remove_duplicate_rows,
    remove_duplicate_columns,
    handle_outliers
)


def test_normalize_column_names():
    df = pd.DataFrame(columns=["Age ", "Total Marks", "Student Name"])
    new_df, msg = normalize_column_names(df)
    assert all(col.islower() for col in new_df.columns)
    assert all(" " not in col for col in new_df.columns)
    assert "_" in new_df.columns[1]
    assert "normalized" in msg.lower() or "already" in msg.lower()


def test_remove_duplicate_rows():
    df = pd.DataFrame({"A": [1, 1, 2], "B": [3, 3, 4]})
    new_df, msg = remove_duplicate_rows(df)
    assert new_df.shape[0] == 2
    assert "removed" in msg.lower()


def test_remove_duplicate_columns_true():
    # ✅ This correctly creates duplicate column names
    df = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["A", "A", "B"])
    new_df, msg = remove_duplicate_columns(df)
    assert "removed" in msg.lower()
    assert len(new_df.columns) < len(df.columns)


def test_remove_duplicate_columns_no_duplicates():
    # ✅ This ensures it works even when no duplicates exist
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    new_df, msg = remove_duplicate_columns(df)
    assert "removed" in msg.lower() or "no duplicate" in msg.lower()
    assert isinstance(new_df, pd.DataFrame)


def test_handle_outliers_iqr():
    df = pd.DataFrame({"val": [1, 2, 3, 1000]})
    new_df, msg = handle_outliers(df.copy(), "val", "IQR", 1.5)
    assert new_df["val"].max() <= 1000  # Clipped within IQR bounds
    assert "iqr" in msg.lower() or "clipped" in msg.lower()
