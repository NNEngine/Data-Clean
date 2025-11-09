# <div align = "center">Data Clean</div>


https://github.com/user-attachments/assets/d6c59b1a-e31c-4d54-9117-42aaf21313f0

**Problem**

> In every *machine learning workflow*, **data cleaning** is one of the most time-consuming and repetitive tasks. 
  Yet, as ML engineers, our true focus should be on building models, crafting architectures, and solving real problems - not spending endless hours handling missing values, formatting inconsistencies and      unwanted noise in CSV files.    

**Solution**
> That's exactly why I build this **CSV Data Cleaning App**. This tool helps you *clean your data* in few steps. All you need to do is to click on the button the operation you want to apply on the file.   After applying all the operations, you can download the final cleaned CSV File.

---

# 💻 Tech Stack:
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![NumPy](https://img.shields.io/badge/numpy-%23013243.svg?style=for-the-badge&logo=numpy&logoColor=white) ![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white) ![scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white) ![Git](https://img.shields.io/badge/git-%23F05033.svg?style=for-the-badge&logo=git&logoColor=white) ![GitHub](https://img.shields.io/badge/github-%23121011.svg?style=for-the-badge&logo=github&logoColor=white)

---

#  Clean Data Dashboard

![2807765_18140](https://github.com/user-attachments/assets/09ca0184-f359-49b2-89aa-3bccaf0ad43e)

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

## How to run locally
Clone the repo
```python
git clone https://github.com/NNEngine/Data-Clean.git
```

Then, run
```python
pyhthon app.py
```

Then, upload the CSV File

<img width="402" height="297" alt="image" src="https://github.com/user-attachments/assets/9e8f93aa-561a-48b8-9bc3-caa24c122fbd" />

And Perform the Cleaning Operations Listed Above


### Reort Generation

After CLeaning the Data, You can generate the detailed interactive report of the data

<img width="1226" height="395" alt="image" src="https://github.com/user-attachments/assets/be565a20-87a7-4a5a-a204-0155515d1c5e" />
