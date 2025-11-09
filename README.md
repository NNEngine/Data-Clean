# <div align = "center">Data Clean</div>

![2807765_18140](https://github.com/user-attachments/assets/09ca0184-f359-49b2-89aa-3bccaf0ad43e)

**Problem**

> In every *machine learning workflow*, **data cleaning** is one of the most time-consuming and repetitive tasks. 
  Yet, as ML engineers, our true focus should be on building models, crafting architectures, and solving real problems - not spending endless hours handling missing values, formatting inconsistencies and      unwanted noise in CSV files.    

**Solution**
> That's exactly why I build this **CSV Data Cleaning App**. This tool helps you *clean your data* in few steps. All you need to do is to click on the button the operation you want to apply on the file.   After applying all the operations, you can download the final cleaned CSV File.

---

#  Clean Data Dashboard

An interactive web-based tool built with **Gradio** for data cleaning, preprocessing, and profiling.  
It helps ML engineers and data scientists clean messy datasets quickly — handling missing values, encoding, outliers, and generating detailed reports.

---

## 🚀 Features
✅ Upload and preview CSV datasets  
✅ Delete, rename, normalize columns  
✅ Handle missing values (Mean, Median, Mode)  
✅ Detect and remove duplicates  
✅ Apply Label / Ordinal encoding  
✅ Change column data types  
✅ Outlier detection via IQR, Z-score, Winsorization, Min-Max  
✅ Generate detailed `pandas-profiling` reports  
✅ Download cleaned datasets  

---

## Project Structure
- .devcontainer
- .github/workflows/code-check.yaml
- tests
- helper_functions.py
- report_generation.py
- app.py
- requirements.txt
- README.md
- LICENSE

---

### How to run locally
clone the repo
```python
git clone
```
