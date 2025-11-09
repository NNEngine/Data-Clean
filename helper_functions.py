
import gradio as gr
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import io
import numpy as np
import tempfile
import os


# ===========================================================
#                     Helper Functions
# ===========================================================

def file_summary(df):
    if df is None:
        return pd.DataFrame(), "⚠️ No data loaded."
    memory_usage = df.memory_usage(deep=True)
    column_types = []
    for col in df.columns:
        dtype = df[col].dtype
        if pd.api.types.is_numeric_dtype(dtype):
            unique_ratio = df[col].nunique() / len(df) if len(df) > 0 else 0
            if unique_ratio < 0.05 or df[col].nunique() < 20:
                column_types.append("Categorical (Numerical)")
            else:
                column_types.append("Continuous")
        elif pd.api.types.is_object_dtype(dtype) or pd.api.types.is_categorical_dtype(dtype):
            column_types.append("Categorical (String/Object)")
        elif pd.api.types.is_bool_dtype(dtype):
            column_types.append("Categorical (Boolean)")
        else:
            column_types.append("Other")

    mem_vals = [round(df[c].memory_usage(deep=True) / 1024, 2) for c in df.columns]
    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values,
        "Column Type": column_types,
        "NULL Values": df.isnull().sum().values,
        "Memory Size (KB)": mem_vals
    })
    return summary_df, f"📊 Summary Generated: {df.shape[1]} columns, {df.shape[0]} rows"


# ===========================================================
#                   Loading CSV + UI helpers
# ===========================================================

def load_csv(file):
    if file is None:
        return None, None, pd.DataFrame(), gr.update(choices=[]), gr.update(choices=[]), "⚠️ Please upload a CSV file."
    try:
        df = pd.read_csv(file.name)
        cols = df.columns.tolist()
        # Detect only encodable columns
        encodable_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
        summary, _ = file_summary(df)
        return df, df.copy(), summary, gr.update(choices=cols), gr.update(choices=encodable_cols), f"✅ File loaded successfully! Shape: {df.shape}"
    except Exception as e:
        return None, None, pd.DataFrame(), gr.update(choices=[]), gr.update(choices=[]), f"❌ Error: {e}"


# ===========================================================
#                Duplicate, Missing & Deletion
# ===========================================================

def check_duplicate_columns(df):
    if df is None:
        return "⚠️ Please load a dataset first."
    dup_cols = df.columns[df.columns.duplicated()]
    if len(dup_cols) == 0:
        return "✅ No duplicate columns found."
    return f"⚠️ Found duplicate columns: {', '.join(dup_cols)}"

def remove_duplicate_columns(df):
    if df is None:
        return df, "⚠️ Please load a dataset first."
    dup_cols = df.columns[df.columns.duplicated()]
    if len(dup_cols) == 0:
        return df, "✅ No duplicate columns to remove."
    df = df.loc[:, ~df.columns.duplicated()]
    return df, f"✅ Removed duplicate columns: {', '.join(dup_cols)}"

def check_duplicate_rows(df):
    if df is None:
        return "⚠️ Please load a dataset first."
    dup_rows = df.duplicated().sum()
    if dup_rows == 0:
        return "✅ No duplicate rows found."
    return f"⚠️ Found {dup_rows} duplicate rows."

def remove_duplicate_rows(df):
    if df is None:
        return df, "⚠️ Please load a dataset first."
    dup_rows = df.duplicated().sum()
    if dup_rows == 0:
        return df, "✅ No duplicate rows to remove."
    df = df.drop_duplicates()
    return df, f"✅ Removed {dup_rows} duplicate rows successfully."

def check_missing_columns(df):
    if df is None:
        return "⚠️ Please load a dataset first."
    missing = df.isnull().sum()
    cols_with_missing = missing[missing > 0]
    if cols_with_missing.empty:
        return "✅ No missing values found."
    return f"⚠️ Columns with missing values: {', '.join(cols_with_missing.index)}"

def drop_high_missing(df):
    if df is None:
        return df, "⚠️ No data loaded."
    missing_pct = df.isnull().mean() * 100
    to_drop = missing_pct[missing_pct > 50].index.tolist()
    if not to_drop:
        return df, "✅ No columns with >50% missing values."
    df = df.drop(columns=to_drop)
    return df, f"✅ Dropped columns with >50% missing values: {', '.join(to_drop)}"

def delete_column(df, col):
    if df is None:
        return df, "⚠️ Please load a dataset first."
    if col not in df.columns:
        return df, f"⚠️ Column '{col}' not found."
    df = df.drop(columns=[col])
    return df, f"✅ Column '{col}' deleted."


# ===========================================================
#     Missing Value Handler (Column-Type Based Logic)
# ===========================================================

def get_missing_columns(df):
    if df is None:
        return gr.update(choices=[]), "⚠️ Please load a dataset first."
    cols = df.columns[df.isnull().any()].tolist()
    if not cols:
        return gr.update(choices=[]), "✅ No columns with missing values."
    return gr.update(choices=cols), f"⚠️ Columns with missing values: {', '.join(cols)}"

def detect_column_type(df, column):
    if df is None or column not in df.columns:
        return "⚠️ Invalid column.", gr.update(choices=[])
    dtype = df[column].dtype
    if pd.api.types.is_numeric_dtype(dtype):
        unique_ratio = df[column].nunique() / len(df)
        if unique_ratio < 0.05 or df[column].nunique() < 20:
            col_type = "Categorical (Numerical)"
            options = ["Mode"]
        else:
            col_type = "Continuous (Numerical)"
            options = ["Mean", "Median", "Mode"]
    else:
        col_type = "Categorical (String/Object)"
        options = ["Mode"]
    return f"🧩 Column Type: {col_type}", gr.update(choices=options, value=options[0])

def apply_missing_value(df, column, method):
    if df is None:
        return df, "⚠️ Please load a dataset first."
    if column not in df.columns:
        return df, f"⚠️ Column '{column}' not found."
    if df[column].isnull().sum() == 0:
        return df, f"✅ Column '{column}' has no missing values."

    if pd.api.types.is_numeric_dtype(df[column]):
        if method == "Mean":
            df[column].fillna(df[column].mean(), inplace=True)
        elif method == "Median":
            df[column].fillna(df[column].median(), inplace=True)
        elif method == "Mode":
            df[column].fillna(df[column].mode().iloc[0], inplace=True)
    else:
        df[column].fillna(df[column].mode().iloc[0], inplace=True)
    return df, f"✅ Missing values in '{column}' filled using {method}."


# ===========================================================
#               Encoding + Download Functions
# ===========================================================

def show_value_counts(df, col, method):
    """Show value counts only if Ordinal Encoding is selected."""
    if df is None or col not in df.columns:
        return gr.DataFrame(value="⚠️ Please select a valid column.")
    if method != "Ordinal Encoding":
        return gr.DataFrame(value="ℹ️ Value counts visible only for Ordinal Encoding.")
    counts = df[col].value_counts(dropna=False).reset_index()
    counts.columns = [col, "Count"]
    return counts

def encode_column(df, col, method, order):
    if df is None:
        return df, "⚠️ Please load a dataset first."
    if col not in df.columns:
        return df, "⚠️ Column not found."

    if method == "Label Encoding":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        return df, f"✅ Label Encoding applied on '{col}'."

    elif method == "Ordinal Encoding":
        if not order:
            return df, "⚠️ Please provide order for Ordinal Encoding."

        # Normalize both the column values and user-provided order for comparison
        df[col] = df[col].astype(str).str.strip()
        user_order = [x.strip() for x in order if x.strip()]
        col_values = sorted(df[col].dropna().unique().tolist())

        # Check if user provided valid categories
        missing_from_col = [x for x in user_order if x not in col_values]
        extra_in_col = [x for x in col_values if x not in user_order]

        if missing_from_col:
            return df, f"❌ Invalid category(s): {missing_from_col}. Please check spelling/case. Existing values: {col_values}"

        if extra_in_col:
            msg = f"⚠️ Warning: Some values in column were not in the provided order and will be encoded as NaN: {extra_in_col}"
        else:
            msg = ""

        try:
            oe = OrdinalEncoder(categories=[user_order])
            df[col] = oe.fit_transform(df[[col]])
            return df, f"✅ Ordinal Encoding applied on '{col}' with order {user_order}. {msg}"
        except Exception as e:
            return df, f"❌ Error during encoding: {e}"

    return df, "⚠️ Invalid encoding method."



# ===========================================================
#         Column Normalization & Renaming Functions
# ===========================================================

def normalize_column_names(df):
    """Convert all column names to lowercase, strip spaces, and replace internal spaces with underscores."""
    if df is None:
        return df, "⚠️ Please load a dataset first."

    original_cols = df.columns.tolist()
    new_cols = [col.strip().lower().replace(" ", "_") for col in original_cols]
    rename_map = {old: new for old, new in zip(original_cols, new_cols) if old != new}
    df.columns = new_cols

    if not rename_map:
        return df, "✅ All column names were already normalized."
    return df, f"✅ Column names normalized: {rename_map}"


def rename_single_column(df, old_col, new_col):
    """Rename one specific column."""
    if df is None:
        return df, "⚠️ Please load a dataset first."
    if old_col not in df.columns:
        return df, f"⚠️ Column '{old_col}' not found."
    if not new_col.strip():
        return df, "⚠️ Please enter a valid new column name."

    df = df.rename(columns={old_col: new_col.strip()})
    return df, f"✅ Column '{old_col}' renamed to '{new_col.strip()}'."


# ===========================================================
#            Data Type Conversion (Numerical Columns)
# ===========================================================

def get_numeric_columns(df):
    """Return a list of numeric columns for dtype conversion."""
    if df is None:
        return gr.update(choices=[]), "⚠️ Please load a dataset first."
    num_cols = df.select_dtypes(include=["int", "float", "complex"]).columns.tolist()
    if not num_cols:
        return gr.update(choices=[]), "✅ No numeric columns available for conversion."
    return gr.update(choices=num_cols), f"🔢 Numeric columns available: {', '.join(num_cols)}"


def show_current_dtype(df, col):
    """Display the current dtype of the selected numeric column."""
    if df is None or col not in df.columns:
        return "⚠️ Please select a valid column."
    dtype = str(df[col].dtype)
    return f"📘 Current Data Type: {dtype}"


def change_column_dtype(df, col, new_dtype):
    """Change the data type of a numeric column using pandas .astype()."""
    if df is None:
        return df, "⚠️ Please load a dataset first."
    if col not in df.columns:
        return df, f"⚠️ Column '{col}' not found."
    if not new_dtype:
        return df, "⚠️ Please select a new data type."

    try:
        df[col] = df[col].astype(new_dtype)
        return df, f"✅ Column '{col}' converted to type '{new_dtype}'."
    except Exception as e:
        return df, f"❌ Conversion failed: {e}"



# ===========================================================
#            Outlier Detection & Handling Functions
# ===========================================================


def get_continuous_columns(df):
    """Detect all numerical columns (int and float) for outlier handling."""
    if df is None:
        return gr.update(choices=[]), "⚠️ Please load a dataset first."
    
    numeric_cols = df.select_dtypes(include=["int", "float"]).columns.tolist()
    
    if not numeric_cols:
        return gr.update(choices=[]), "✅ No numerical columns found."
    
    return gr.update(choices=numeric_cols), f"📊 Numerical columns detected: {', '.join(numeric_cols)}"



def show_column_stats(df, col):
    """Display basic stats for selected continuous column."""
    if df is None or col not in df.columns:
        return "⚠️ Please select a valid column."
    stats = df[col].describe().to_dict()
    return (
        f"📈 Column: {col}\n"
        f"Mean: {stats['mean']:.3f}, Std: {stats['std']:.3f}, Min: {stats['min']:.3f}, Max: {stats['max']:.3f}"
    )


def handle_outliers(df, col, method, threshold):
    """Apply chosen outlier handling technique."""
    if df is None:
        return df, "⚠️ Please load a dataset first."
    if col not in df.columns:
        return df, f"⚠️ Column '{col}' not found."
    if not pd.api.types.is_numeric_dtype(df[col]):
        return df, f"⚠️ Column '{col}' is not numeric."
    if threshold is None or str(threshold).strip() == "":
        return df, "⚠️ Please enter a valid threshold value."

    try:
        threshold = float(threshold)
    except:
        return df, "⚠️ Threshold value must be numeric."

    series = df[col]

    # IQR method
    if method == "IQR":
        Q1, Q3 = series.quantile(0.25), series.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - threshold * IQR
        upper = Q3 + threshold * IQR
        before = series.copy()
        df[col] = np.clip(series, lower, upper)
        return df, f"✅ IQR method applied with threshold={threshold}. Clipped {sum(before != df[col])} outliers."

    # Z-score method
    elif method == "Z-score":
        mean, std = series.mean(), series.std()
        z_scores = (series - mean) / std
        mask = np.abs(z_scores) > threshold
        before = series.copy()
        df.loc[mask, col] = mean  # replace with mean
        return df, f"✅ Z-score method applied (|Z| > {threshold}). Replaced {mask.sum()} outliers with mean."

    # Winsorization
    elif method == "Winsorization":
        lower = series.quantile(threshold / 100)
        upper = series.quantile(1 - threshold / 100)
        before = series.copy()
        df[col] = np.clip(series, lower, upper)
        return df, f"✅ Winsorization applied with {threshold}% tails capped."

    # Min-Max clipping
    elif method == "MinMax":
        min_val = series.min()
        max_val = series.max()
        lower = min_val + threshold * (max_val - min_val)
        upper = max_val - threshold * (max_val - min_val)
        before = series.copy()
        df[col] = np.clip(series, lower, upper)
        return df, f"✅ Min-Max clipping applied with threshold={threshold}. Clipped {sum(before != df[col])} values."

    else:
        return df, "⚠️ Invalid outlier handling method selected."

# ===========================================================
#            Downloading the Cleaned CSV File
# ===========================================================

def make_csv_download(df):
    if df is None or df.empty:
        return None
    # Create a temporary file
    temp_dir = tempfile.gettempdir()
    temp_path = os.path.join(temp_dir, "cleaned_data.csv")
    df.to_csv(temp_path, index=False)
    return temp_path
