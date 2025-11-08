import streamlit as st 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

st.set_page_config(
    page_title = "Clean Data",
    layout = "wide",
)

uploaded_file = st.file_uploader("Choose a CSV file", type = ["csv"])


#-------------------------------------------------------------------------------
# # ensure session keys exist
# if "df" not in st.session_state:
#     st.session_state.df = pd.DataFrame()
# if "file_loaded" not in st.session_state:
#     st.session_state.file_loaded = False
# if "uploaded_filename" not in st.session_state:
#     st.session_state.uploaded_filename = None

# ================= Version Management =====================

# def init_versions():
#     """Initialize version storage if not already created."""
#     if "versions" not in st.session_state:
#         st.session_state.versions = {}
#     if "current_version" not in st.session_state:
#         st.session_state.current_version = None


# def save_version(name, df):
#     """Save a new version of the dataframe."""
#     st.session_state.versions[name] = df.copy()
#     st.session_state.current_version = name
    # st.success(f"✅ Saved version '{name}'")


# def get_current_df():
#     """Return the currently active DataFrame version."""
#     if st.session_state.current_version and st.session_state.current_version in st.session_state.versions:
#         return st.session_state.versions[st.session_state.current_version].copy()
#     elif "df" in st.session_state:
#         return st.session_state.df.copy()
#     else:
#         st.warning("⚠️ No DataFrame loaded yet.")
#         return None


# def list_versions():
#     """List available versions."""
#     return list(st.session_state.versions.keys())


# def switch_version(name):
#     """Switch to a specific saved version."""
#     if name in st.session_state.versions:
#         st.session_state.current_version = name
#         st.success(f"🔄 Switched to version '{name}'")
#     else:
#         st.error(f"❌ Version '{name}' not found.")
#-----------------------------------------------------------------------
        

#------------------Number of Rows to Display----------------------------

def num_of_rows_to_display(df:pd.DataFrame):
    rows_to_display = st.slider(
                "Select number of rows to display:",
                min_value=1,
                max_value=len(df),
                value=5,
                step=1
            )
    st.write(f"### First {rows_to_display} rows of the file:")
    st.dataframe(df.head(rows_to_display))

#-----------------------------File Summary------------------------------

def file_summary(df: pd.DataFrame):
    """Display a detailed, scrollable summary of the dataframe including dtype, nulls, memory, and column classification."""
    
    memory_usage = df.memory_usage(deep=True)
    column_types = []

    for col in df.columns:
        dtype = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(dtype):
            # Determine if numeric column is categorical or continuous
            unique_ratio = df[col].nunique() / len(df)
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

    summary_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.values,
        "Column Type": column_types,
        "NULL Values": df.isnull().sum().values,
        "Memory Size (KB)": (memory_usage[1:] / 1024).round(2)
    })

    # Display scrollable table using Streamlit dataframe
    st.write("### 📊 File Summary")
    st.dataframe(summary_df, use_container_width=True, height=400)

#-------------------------------delete specific column------------------------------
    
def delete_specific_column(df):
    st.subheader("🧹 Delete Specific Column")

    if df.empty:
        st.warning("⚠️ DataFrame is empty. Please load a dataset first.")
        return
    
    df = st.session_state.df
    col_to_delete = st.selectbox("Select the column you want to delete:", df.columns)

    if st.button("🗑️ Delete Selected Column"):
        if col_to_delete in df.columns:
            df.drop(columns=[col_to_delete], inplace=True)
            st.success(f"✅ Column '{col_to_delete}' has been deleted successfully!")
            st.dataframe(df)
        else:
            st.error(f"❌ Column '{col_to_delete}' not found in the DataFrame.")
    return df

#-------------------------Duplicate Columns------------------------------

def duplicate_columns(df: pd.DataFrame):
    """Check for and handle duplicate column names."""
    if "df" not in st.session_state or st.session_state.df.empty:
        st.warning("⚠️ Please load a dataset first.")
        return


    df = st.session_state.df
    duplicate_columns_name = df.columns[df.columns.duplicated()]

    if len(duplicate_columns_name) > 0:
        st.warning("⚠️ Duplicate Columns Found!")
        st.write("### 🔍 Duplicate Columns:")
        st.write(duplicate_columns_name.tolist())

        # Drop duplicate columns (keep the first occurrence)
        df = df.loc[:, ~df.columns.duplicated()]
        st.success("✅ Duplicate columns removed.")
        num_of_rows_to_display(df)
    else:
        st.success("✅ No duplicate columns found!")


#----------------------------Duplicate Rows-----------------------------
        
def duplicate_rows(df:pd.DataFrame):
    """Check for and handle duplicate rows."""
    num_duplicates = df.duplicated().sum()

    if num_duplicates > 0:
        st.warning(f"{num_duplicates} Duplicate Rows Found!")
        st.write("### Example Duplicate Rows: ")
        st.dataframe(df[df.duplicated()].head())

        if st.button("Remove Duplicate Rows"):
            df = df.drop_duplicates()
            st.session_state.df = df
            st.success("Duplicate rows removed!")
            st.experimental_rerun()
    else:
        st.success("✅ No duplicate rows found!")

#-------------------------------Handling Columns with Missing Values-------------------------

def handle_high_missing_columns(df: pd.DataFrame):
    """
    Detect columns with >50% missing values, show them to the user,
    and optionally delete them from the DataFrame.
    """
    df = st.session_state.df
    st.write("### 🧮 Handling Columns with Missing Values")

    # Calculate missing count and percentage for all columns
    missing_summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum(),
        "Percentage (%)": (df.isnull().sum() / len(df) * 100).round(2)
    })

    # Keep only columns that have at least one missing value
    missing_summary = missing_summary[missing_summary["Missing Values"] > 0]

    if missing_summary.empty:
        st.success("✅ No columns with missing values found in the dataset.")
        return

    st.write(f"Total columns with missing values: **{len(missing_summary)}**")
    st.dataframe(missing_summary, use_container_width=True, height=250)


    # Identify columns having more than 50% missing values
    high_missing = missing_summary[missing_summary["Percentage (%)"] > 50]

    if high_missing.empty:
        st.info("ℹ️ No columns have more than 50% missing values.")
        return

    st.warning(f"⚠️ {len(high_missing)} column(s) have more than 50% missing values.")
    st.dataframe(high_missing, use_container_width=True, height=200)

    st.info(
        "It is generally better to delete columns with >50% missing values, "
        "as they provide very little useful information."
    )

    # Let user confirm deletion
    if st.button("🗑️ Delete Columns with >50% Missing Values"):
        cols_to_delete = high_missing["Column"].tolist()
        df.drop(columns=cols_to_delete, inplace=True)
        st.session_state.df = df

        st.success(f"✅ Deleted {len(cols_to_delete)} column(s): {', '.join(cols_to_delete)}")
        st.markdown("### Updated DataFrame Preview:")
        st.dataframe(df)


#-----------------------------------Encoding Values--------------------------------------

def encode_categorical_columns(df: pd.DataFrame):
    """
    Detect and encode categorical columns interactively using LabelEncoder or OrdinalEncoder.
    Updates the original DataFrame and displays the encoded dataset.
    """

    st.write("### 🔤 Encode Categorical Columns")

    # Detect categorical columns (object or category dtype)
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()

    if not cat_cols:
        st.success("✅ No categorical columns found to encode.")
        return

    st.info(f"Found **{len(cat_cols)}** categorical column(s): {', '.join(cat_cols)}")

    # Iterate through each categorical column
    for col in cat_cols:
        st.subheader(f"🧩 Column: `{col}`")

        # Show value counts as a DataFrame (sorted by frequency)
        vc_df = df[col].value_counts(dropna=False).reset_index()
        vc_df.columns = [col, "Count"]
        st.dataframe(vc_df, use_container_width=True, height=200)

        # Ask user to choose encoding type
        encoding_type = st.selectbox(
            f"Choose encoding method for `{col}`:",
            ["Select", "Label Encoding", "Ordinal Encoding"],
            key=f"encoder_{col}"
        )

        # If Label Encoding selected
        if encoding_type == "Label Encoding":
            if st.button(f"⚙️ Apply Label Encoding to `{col}`"):
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                st.session_state.df = df
                st.success(f"✅ Label Encoding applied successfully on `{col}`!")
                st.dataframe(df[[col]].head(), use_container_width=True)

        # If Ordinal Encoding selected
        elif encoding_type == "Ordinal Encoding":
            unique_values = df[col].dropna().unique().tolist()
            st.write("Provide order of categories (drag to reorder):")
            ordered_values = st.multiselect(
                f"Define order for `{col}`:",
                options=unique_values,
                default=unique_values,
                key=f"order_{col}"
            )

            if len(ordered_values) != len(unique_values):
                st.warning(f"⚠️ Please include all unique values for `{col}` before applying encoding.")
            else:
                if st.button(f"⚙️ Apply Ordinal Encoding to `{col}`"):
                    oe = OrdinalEncoder(categories=[ordered_values])
                    df[col] = oe.fit_transform(df[[col]].astype(str))
                    st.session_state.df = df
                    st.success(f"✅ Ordinal Encoding applied successfully on `{col}`!")
                    st.dataframe(df[[col]].head(), use_container_width=True)

                    st.success("🎉 Encoding process complete!")
                    st.write("### Encoded DataFrame Preview:")
                    st.dataframe(df)


#-------------------------------Handling Missing Values----------------------------------


def handle_missing_values(df: pd.DataFrame):
    """Handle missing (NULL) values interactively using Streamlit.
       Updates df in place and displays using num_of_rows_to_display().
    """

    st.write("### 🧩 Handle Missing / NULL Values")

    total_missing = df.isnull().sum().sum()

    if total_missing == 0:
        st.success("✅ No missing values found in the dataset.")
        return

    st.warning(f"⚠️ Dataset contains {total_missing} missing values.")

    # Display missing value summary
    st.write("#### Missing Values Summary")
    missing_summary = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum(),
        "Percentage (%)": (df.isnull().sum() / len(df) * 100).round(2)
    })
    missing_summary = missing_summary[missing_summary["Missing Values"] > 0]
    st.dataframe(missing_summary, use_container_width=True, height=250)

    # User choice: Delete or Fill
    action = st.radio(
        "Choose how to handle missing values:",
        ["Delete rows with missing values", "Fill missing values"],
        horizontal=True
    )

    # Option 1: Delete rows with missing values
    if action == "Delete rows with missing values":
        if st.button("🗑️ Delete Missing Values"):
            df.dropna(inplace=True)
            st.session_state.df = df
            st.success("✅ Rows with missing values deleted successfully.")
            st.markdown("### New DataFrame Preview:")
            num_of_rows_to_display(df)
            return

    # Option 2: Fill missing values
    elif action == "Fill missing values":
        st.info("Select a column and choose a filling method for each.")
        null_columns = df.columns[df.isnull().any()].tolist()

        fill_methods = {}
        for col in null_columns:
            st.write(f"**Column:** {col}")
            method = st.selectbox(
                f"Choose fill method for '{col}':",
                ["Select", "Mean", "Median", "Mode"],
                key=f"fill_method_{col}"
            )
            fill_methods[col] = method

        if st.button("✨ Apply Fill Methods"):
            all_filled = True

            for col, method in fill_methods.items():
                if method == "Select":
                    st.warning(f"⚠️ Please choose a method for column: `{col}`")
                    all_filled = False
                    continue

                if method == "Mean":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].mean(), inplace=True)
                    else:
                        st.warning(f"Cannot use mean for non-numeric column: `{col}`")

                elif method == "Median":
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        st.warning(f"Cannot use median for non-numeric column: `{col}`")

                elif method == "Mode":
                    mode_values = df[col].mode()
                    if not mode_values.empty:
                        df[col].fillna(mode_values.iloc[0], inplace=True)
                    else:
                        st.warning(f"No mode found for column: `{col}` — skipped.")

            remaining_nulls = df.isnull().sum().sum()

            if all_filled and remaining_nulls == 0:
                st.session_state.df = df
                st.success("✅ All missing values handled successfully!")
                st.markdown("### New DataFrame Preview:")
                st.dataframe(df)
            else:
                st.warning("⚠️ Some columns still have missing values. Please handle them.")


# def load_csv():
#     # Check if user uploaded a file
#     if uploaded_file is not None:
#         # Load only once and store in session_state
#         if st.button("Load CSV"):
#             st.session_state.df = pd.read_csv(uploaded_file)
#             st.success("✅ File loaded successfully!")
        
#         # Display once DataFrame is loaded
#         if "df" in st.session_state:
#             df = st.session_state.df

#             st.markdown("---")

#             # Select number of rows to display
#             num_of_rows_to_display(df)

#             # Shape of the Dataset
#             st.warning(f"Shape of the Dataset: {df.shape}")

#             # File Summary
#             st.markdown("---")
#             file_summary(df)

#             # delete_specific_column
#             st.markdown("---")
#             delete_specific_column(df)
#             df = st.session_state.df


#             # Handling Duplicate Columns
#             st.markdown("---")
#             duplicate_columns(df)
#             df = st.session_state.df

#             # Handling duplicate rows
#             duplicate_rows(df)
#             df = st.session_state.df

#             # Handling columns with missing values
#             st.markdown("---")
#             handle_high_missing_columns(df)
#             df = st.session_state.df

#             # Handling Missing Values
#             handle_missing_values(df)
#             df = st.session_state.df

#             # Encoding values
#             st.markdown("---")
#             encode_categorical_columns(df)
#             df = st.session_state.df

# load_csv()


if uploaded_file is not None:
        # Load only once and store in session_state
        if st.button("Load CSV"):
            st.session_state.df = pd.read_csv(uploaded_file)
            st.success("✅ File loaded successfully!")
        
        # Display once DataFrame is loaded
        if "df" in st.session_state:
            df = st.session_state.df

            st.markdown("---")

            # Select number of rows to display
            num_of_rows_to_display(df)

            # Shape of the Dataset
            st.warning(f"Shape of the Dataset: {df.shape}")

            # File Summary
            st.markdown("---")
            file_summary(df)

            # delete_specific_column
            st.markdown("---")
            delete_specific_column(df)
            # df = st.session_state.df


            # Handling Duplicate Columns
            st.markdown("---")
            duplicate_columns(df)
            # df = st.session_state.df

            # Handling duplicate rows
            duplicate_rows(df)
            # df = st.session_state.df

            # Handling columns with missing values
            st.markdown("---")
            handle_high_missing_columns(df)
            # df = st.session_state.df

            # Handling Missing Values
            handle_missing_values(df)
            # df = st.session_state.df

            # Encoding values
            st.markdown("---")
            encode_categorical_columns(df)
            # df = st.session_state.df
                


# ------- load / reload controls (guaranteed visible when a file is uploaded) -------
# if uploaded_file is not None:
#     st.write(f"File ready to load: **{uploaded_file.name}**")

#     # Show Load button and keep its press remembered via file_loaded flag
#     if st.button("📂 Load CSV", key="btn_load_csv"):
#         # load from the uploaded file only when the button is clicked
#         st.session_state.df = pd.read_csv(uploaded_file)
#         st.session_state.file_loaded = True
#         st.session_state.uploaded_filename = uploaded_file.name
#         st.success("✅ File loaded into session_state.df")
#         st.rerun()   # re-run so UI shows loaded state immediately

#     # Always show reload (reset to original upload) only when a file is already loaded
#     if st.session_state.file_loaded and st.session_state.uploaded_filename == uploaded_file.name:
#         if st.button("🔄 Reload Original File", key="btn_reload"):
#             st.session_state.df = pd.read_csv(uploaded_file)
#             st.success("♻️ File reloaded (original upload)")
#             st.rerun()

#     # If file_loaded is True (either just loaded or from earlier), continue to tools
#     if st.session_state.file_loaded and st.session_state.uploaded_filename == uploaded_file.name:
#         df = st.session_state.df  # always the latest
#         st.markdown("---")

#         # call your UI/tool functions here.
#         # IMPORTANT: ensure each function that modifies the DF writes back to st.session_state.df
#         # Examples below assume these functions accept df and update session_state inside them.
#         num_of_rows_to_display_df = df.copy()
#         num_of_rows_to_display(num_of_rows_to_display_df)
#         st.warning(f"Shape of the Dataset: {df.shape}")
#         st.markdown("---")
#         file_summary(num_of_rows_to_display_df)

#         st.markdown("---")
#         # if your delete_specific_column currently expects df param, you can pass df,
#         # but inside that function make sure after dropping column you do:
#         # st.session_state.df = df.copy()
#         # delete_specific_column(df)
#         delete_specific_column_df = delete_specific_column(num_of_rows_to_display_df)


#         st.markdown("---")
#         duplicate_columns(df)   # same rule: persist by writing st.session_state.df inside

#         st.markdown("---")
#         duplicate_rows(df)

#         st.markdown("---")
#         handle_high_missing_columns(df)

#         st.markdown("---")
#         handle_missing_values(df)

#         st.markdown("---")
#         encode_categorical_columns(delete_specific_column_df)

# else:
#     st.info("📤 Please upload a CSV file to begin.")


# def load_csv():
#     init_versions()  # Initialize version tracking

#     # --- File Upload ---
#     st.markdown("## 📁 Upload CSV File")
#     # uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

#     # --- Load Button (always visible when file selected) ---
#     if uploaded_file is not None:
#         if st.button("📂 Load CSV"):
#             df = pd.read_csv(uploaded_file)
#             st.session_state.df = df
#             save_version("original", df)
#             st.success("✅ File loaded successfully! Version 'original' created.")

#     # --- Check if any version exists ---
#     if not list_versions():
#         st.info("ℹ️ Please upload and load a CSV file first.")
#         return

#     # --- Sidebar for version management ---
#     st.sidebar.markdown("## 🕓 DataFrame Versions")
#     selected_version = st.sidebar.selectbox(
#         "Select version to view/edit:",
#         list_versions(),
#         index=list_versions().index(st.session_state.current_version)
#         if st.session_state.current_version in list_versions()
#         else 0
#     )

#     if st.sidebar.button("🔁 Switch to Selected Version"):
#         switch_version(selected_version)

#     # --- Get current working DataFrame ---
#     df = get_current_df()
#     if df is None:
#         return

#     # --- Show active version info ---
#     st.markdown("---")
#     st.write(f"### Active Version: `{st.session_state.current_version}`")
#     st.warning(f"Shape: {df.shape}")

#     # --- Display Preview ---
#     num_of_rows_to_display(df)

#     # --- Sequential Function Calls ---
#     st.markdown("---")
#     file_summary(df)

#     st.markdown("---")
#     delete_specific_column(df.copy())
#     save_version("deleted_columns", df)

#     st.markdown("---")
#     duplicate_columns(df.copy())
#     save_version("no_duplicate_columns", df)

#     st.markdown("---")
#     duplicate_rows(df.copy())
#     save_version("no_duplicate_rows", df)

#     st.markdown("---")
#     handle_high_missing_columns(df.copy())
#     save_version("removed_high_missing", df)

#     st.markdown("---")
#     handle_missing_values(df.copy())
#     save_version("filled_missing", df)

#     st.markdown("---")
#     encode_categorical_columns(df.copy())
#     save_version("encoded", df)

#     st.success("🎉 All transformations complete. Versions updated!")

# load_csv()
