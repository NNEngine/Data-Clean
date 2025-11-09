#==============================================================
#            Deendencies
#===============================================================

import gradio as gr
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import io
import numpy as np
import tempfile
import os


#==================================================================
#        Other Dependencies
#==================================================================

from helper_functions import file_summary, load_csv
from helper_functions import check_duplicate_columns, remove_duplicate_columns, check_duplicate_rows, remove_duplicate_rows, check_missing_columns, drop_high_missing, delete_column
from helper_functions import get_missing_columns, detect_column_type, apply_missing_value
from helper_functions import show_value_counts, encode_column
from helper_functions import normalize_column_names, rename_single_column
from helper_functions import get_numeric_columns, show_current_dtype, change_column_dtype
from helper_functions import get_continuous_columns, show_column_stats, handle_outliers
from helper_functions import make_csv_download

from report_generation import generate_profile_report



# ===========================================================
#                     Gradio Layout
# ===========================================================

with gr.Blocks(theme="soft") as demo:
    gr.Markdown("# <div align = 'center'> **Clean Data Dashboard** </div>") 
    gr.Markdown("<div align = 'center'>In every machine learning workflow, data cleaning is one of the most time-consuming and repetitive tasks. yet, as ML engineers, our true focus should be on building models, crafting architectures, and solving real problems - not spending endless hours handling missing values, formatting inconsistencies and unwanted noise in CSV files.</div>") 
    gr.Markdown("<div align = 'center'> That's exactly why I build this CSV Data Cleaning App. This tool helps you clean your data in few steps. All you need to do is to click on the button the operation you want to apply on the file. After applying all the operations, you can download the final cleaned CSV File.</div>")
    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1, min_width=400):
            gr.HTML("<div style='max-height: 90vh; overflow-y: auto; padding-right: 10px;'>")
        
            gr.Markdown("# ⚙️ Tools Panel")
            
            file_input = gr.File(label="Choose CSV", file_types=[".csv"])
            load_btn = gr.Button("📂 Load CSV")
            status_box = gr.Textbox(label="Status", interactive=False)
            gr.Markdown("---")

            delete_col = gr.Dropdown(label="Select Column to Delete")
            gr.Markdown("Delete Columns which you don't need!")
            delete_btn = gr.Button("🗑️ Delete Column")
            delete_status = gr.Textbox(label="Delete Status", interactive=True)
            gr.Markdown("---")

            dup_col_status = gr.Textbox(label="Duplicate Columns", interactive=False)
            dup_col_check = gr.Button("🔍 Check Duplicate Columns")
            dup_col_btn = gr.Button("🧬 Remove Duplicate Columns")
            gr.Markdown("---")

            dup_row_status = gr.Textbox(label="Duplicate Rows", interactive=False)
            dup_row_check = gr.Button("🔍 Check Duplicate Rows")
            dup_row_btn = gr.Button("📄 Remove Duplicate Rows")
            gr.Markdown("---")

            missing_status = gr.Textbox(label="Missing Columns Check", interactive=False)
            check_missing_btn = gr.Button("🔍 Check Columns with Missing Values")
            drop_high_missing_btn = gr.Button("🧮 Drop Columns with >50% Missing Values")
            gr.Markdown("---")

            gr.Markdown("### 🧩 Handle Missing Values")
            missing_col = gr.Dropdown(label="Select Column with Missing Values")
            detect_type_box = gr.Textbox(label="Column Type", interactive=False)
            fill_method = gr.Dropdown(label="Select Fill Method", choices=[])
            apply_fill_btn = gr.Button("✨ Apply Fill Method")
            fill_status = gr.Textbox(label="Fill Operation Status", interactive=False)
            gr.Markdown("---")

            gr.Markdown("### 🔤 Encoding Section")
            encode_col = gr.Dropdown(label="Select Column to Encode")
            encode_method = gr.Radio(["Label Encoding", "Ordinal Encoding"], label="Encoding Type", value="Label Encoding")
            value_counts_box = gr.Textbox(label="Value Counts (for Ordinal Encoding)", interactive=False, lines=8)
            encode_order = gr.Textbox(label="If Ordinal, Enter Order (comma-separated)")
            encode_status = gr.Textbox(label="Encoding Status", interactive=False)
            encode_btn = gr.Button("⚙️ Apply Encoding")
            gr.Markdown("---")

            gr.Markdown("### 🏷️ Column Name Normalization & Renaming")
            normalize_btn = gr.Button("🔡 Normalize Column Names")
            normalize_status = gr.Textbox(label="Normalization Status", interactive=False)
            rename_col = gr.Dropdown(label="Select Column to Rename")
            new_col_name = gr.Textbox(label="Enter New Column Name")
            rename_btn = gr.Button("✏️ Rename Column")
            rename_status = gr.Textbox(label="Rename Status", interactive=False)

            gr.Markdown("---")
            gr.Markdown("### 🔢 Change Data Type of Columns")
            numeric_detect_btn = gr.Button("🔍 Detect Numeric Columns")
            numeric_detect_status = gr.Textbox(label="Numeric Column Detection", interactive=False)
            dtype_col = gr.Dropdown(label="Select Numeric Column")
            current_dtype_box = gr.Textbox(label="Current Data Type", interactive=False)

            # Target dtype selection
            dtype_choices = [
                "int8", "int16", "int32", "int64",
                "float16", "float32", "float64",
                "complex64", "complex128"
            ]
            new_dtype = gr.Dropdown(label="Select New Data Type", choices=dtype_choices)
            convert_dtype_btn = gr.Button("🔁 Convert Data Type")
            convert_dtype_status = gr.Textbox(label="Data Type Conversion Status", interactive=False)
            gr.Markdown("---")

            gr.Markdown("### 🚨 Outlier Detection & Handling")
            detect_cont_col_btn = gr.Button("🔍 Detect Continuous Columns")
            cont_col_status = gr.Textbox(label="Continuous Columns Detection", interactive=False)
            outlier_col = gr.Dropdown(label="Select Continuous Column")
            col_stats_box = gr.Textbox(label="Column Statistics", interactive=False)

            # Technique + threshold
            outlier_method = gr.Radio(
                ["IQR", "Z-score", "Winsorization", "MinMax"],
                label="Select Outlier Handling Technique",
                value="IQR"
            )
            threshold_value = gr.Textbox(label="Enter Threshold Value (e.g., 1.5 for IQR, 3 for Z-score, etc.)")

            # Apply technique
            apply_outlier_btn = gr.Button("🧮 Apply Technique")
            outlier_status = gr.Textbox(label="Outlier Handling Status", interactive=False)

            gr.Markdown("---")
            reset_btn = gr.Button("♻️ Reset to Original")
            download_trigger = gr.Button("📥 Generate & Download Cleaned CSV")
            download_file = gr.File(label="Your Cleaned CSV File Will Appear Below 👇")
            gr.HTML("</div>")


        with gr.Column(scale=3):
            gr.Markdown("# Data Panel")
            summary_table = gr.DataFrame(label="📊 File Summary", interactive=True, wrap=True)
            gr.Markdown("---")
            gr.Markdown("## 🧾 Data Preview")
            original_df = gr.DataFrame(label="📘 Original Dataset", wrap=True, interactive=False)
            working_df = gr.DataFrame(label="🧪 Working Dataset", wrap=True)

    gr.Markdown("---")
    gr.Markdown("### 🧾 Generate Detailed Data Report")

    generate_report_btn = gr.Button("📈 Create Data Report (It might take time)")
    report_status = gr.HTML(label="Report Status")
    report_file = gr.File(label="Download or View Report")


    # ===========================================================
    #                    Event Bindings
    # ===========================================================

    load_btn.click(load_csv,
        inputs=file_input,
        outputs=[original_df, working_df, summary_table, delete_col, encode_col, status_box]
    )

    delete_btn.click(delete_column, inputs=[working_df, delete_col], outputs=[working_df, delete_status])
    dup_col_check.click(check_duplicate_columns, inputs=working_df, outputs=dup_col_status)
    dup_col_btn.click(remove_duplicate_columns, inputs=working_df, outputs=[working_df, dup_col_status])
    dup_row_check.click(check_duplicate_rows, inputs=working_df, outputs=dup_row_status)
    dup_row_btn.click(remove_duplicate_rows, inputs=working_df, outputs=[working_df, dup_row_status])
    check_missing_btn.click(check_missing_columns, inputs=working_df, outputs=missing_status)
    drop_high_missing_btn.click(drop_high_missing, inputs=working_df, outputs=[working_df, missing_status])

    # Missing values section
    check_missing_btn.click(get_missing_columns, inputs=working_df, outputs=[missing_col, missing_status])
    missing_col.change(detect_column_type, inputs=[working_df, missing_col], outputs=[detect_type_box, fill_method])
    apply_fill_btn.click(apply_missing_value, inputs=[working_df, missing_col, fill_method], outputs=[working_df, fill_status])

    # Encoding section
    encode_col.change(show_value_counts, inputs=[working_df, encode_col, encode_method], outputs=value_counts_box)
    encode_method.change(show_value_counts, inputs=[working_df, encode_col, encode_method], outputs=value_counts_box)
    encode_btn.click(
        lambda df, col, method, order_str: encode_column(df, col, method, [x.strip() for x in order_str.split(",")] if order_str else None),
        inputs=[working_df, encode_col, encode_method, encode_order],
        outputs=[working_df, encode_status]
    )


   # Normalize column names
    def normalize_and_update(df):
        df, msg = normalize_column_names(df)
        if df is None:
            return df, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), msg
        cols = df.columns.tolist()
        return df, gr.update(choices=cols), gr.update(choices=cols), gr.update(choices=cols), msg

    normalize_btn.click(
        normalize_and_update,
        inputs=working_df,
        outputs=[working_df, delete_col, rename_col, encode_col, normalize_status]
    )

    # rename columns
    def rename_and_update(df, old_col, new_col):
        df, msg = rename_single_column(df, old_col, new_col)
        if df is None:
            return df, gr.update(choices=[]), gr.update(choices=[]), gr.update(choices=[]), msg
        cols = df.columns.tolist()
        return df, gr.update(choices=cols), gr.update(choices=cols), gr.update(choices=cols), msg

    rename_btn.click(
        rename_and_update,
        inputs=[working_df, rename_col, new_col_name],
        outputs=[working_df, delete_col, rename_col, encode_col, rename_status]
    )

    # ====================== Data Type Change Section ======================

    # Detect numeric columns
    numeric_detect_btn.click(get_numeric_columns, inputs=working_df, outputs=[dtype_col, numeric_detect_status])

    # Show current dtype when a column is selected
    dtype_col.change(show_current_dtype, inputs=[working_df, dtype_col], outputs=current_dtype_box)

    # Apply dtype change
    convert_dtype_btn.click(change_column_dtype, inputs=[working_df, dtype_col, new_dtype], outputs=[working_df, convert_dtype_status])

    # ===================== Outlier Detection Section =====================

    # Detect continuous columns
    detect_cont_col_btn.click(get_continuous_columns, inputs=working_df, outputs=[outlier_col, cont_col_status])

    # Show stats when a column is selected
    outlier_col.change(show_column_stats, inputs=[working_df, outlier_col], outputs=col_stats_box)

    # Apply selected outlier handling technique
    apply_outlier_btn.click(
        handle_outliers,
        inputs=[working_df, outlier_col, outlier_method, threshold_value],
        outputs=[working_df, outlier_status]
    )


    reset_btn.click(lambda df_orig: (df_orig.copy(), "✅ Reset to original dataset."),
        inputs=original_df,
        outputs=[working_df, status_box]
    )

    download_trigger.click(make_csv_download, inputs=working_df, outputs=download_file)

    generate_report_btn.click(
        generate_profile_report,
        inputs=working_df,
        outputs=[report_file, report_status]
    )

demo.launch()
