
import gradio as gr
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
import io
import numpy as np
import tempfile
import os


# ===========================================================
#        Detailed Data Report using pandas-profiling
# ===========================================================

def generate_profile_report(df):
    """Generate a pandas profiling HTML report and optionally open in a new tab."""
    if df is None or df.empty:
        return None, "⚠️ Please load a valid dataset first."

    try:
        from ydata_profiling import ProfileReport
    except ImportError:
        return None, "❌ Missing dependency: please install it using 'pip install ydata-profiling'."

    try:
        profile = ProfileReport(df, title="📊 Detailed Data Report", explorative=True)
        output_path = "data_profile_report.html"
        profile.to_file(output_path)

        # Create a clickable HTML link that opens in new tab
        html_link = f"""
        ✅ Report generated successfully! Now Download the report (in HTML format) and open it.<br>
        """
        # Return the file + HTML message
        return output_path, html_link
    except Exception as e:
        return None, f"❌ Failed to generate report: {e}"

