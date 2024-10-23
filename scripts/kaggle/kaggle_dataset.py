#%%
# Standard library imports
from pathlib import Path
import pickle
import sys

#%%
# Third-party imports
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pyprojroot import here

#%%
# Append the parent directory of "src"
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

#%%
# Local application/library imports
from src.data_frame_column_cleaner import DataFrameColumnCleaner
from src.data_frame_column_editor import DataFrameColumnEditor

#%%
# from export_py2txt import write_current_file_to_txt

# Ensure I am using an interactive backend:
matplotlib.use("Qt5Agg")  #

#%%
# Establish important project paths
dir_base = Path(here())
dir_data = Path(dir_base, "data/kaggle")
dir_results = Path(dir_base, "results/kaggle")
dir_scripts = Path(dir_base, "scripts/kaggle")

#%%
# Load the dataset (CSV format)
file_path = Path(dir_data, "healthcare_dataset.csv")
if not file_path.exists():
    raise FileNotFoundError(
        f"The file '{file_path}' does not exist. Please check the path and try again."
    )
df_healthcare = pd.read_csv(file_path)

#%%
# Step 1: Data Cleaning and Quality Checks

# Remove leading/trailing whitespaces in all columns and standardize text fields
df_healthcare = df_healthcare.applymap(
    lambda x: x.strip() if isinstance(x, str) else x
)

#%%
# Standardize text fields: Name, Gender, Doctor, Hospital
df_healthcare["Name"] = df_healthcare["Name"].str.title()
df_healthcare["Gender"] = df_healthcare["Gender"].str.capitalize()
df_healthcare["Doctor"] = df_healthcare["Doctor"].str.title()
df_healthcare["Hospital"] = df_healthcare["Hospital"].str.title()

#%%
# Convert 'Date of Admission' and 'Discharge Date' to datetime format
df_healthcare["Date of Admission"] = pd.to_datetime(
    df_healthcare["Date of Admission"], errors="coerce"
)
df_healthcare["Discharge Date"] = pd.to_datetime(
    df_healthcare["Discharge Date"], errors="coerce"
)

#%%
# Check for missing values after conversion and cleaning
missing_values_summary = df_healthcare.isnull().sum()

#%%
# Display the cleaned dataset and missing value summary for review
df_healthcare_head_cleaned = df_healthcare.head()
print(missing_values_summary)
print(df_healthcare_head_cleaned)

#%%
# Step 2: Feature Engineering

# Calculate Length of Stay
df_healthcare["Length of Stay"] = (
    df_healthcare["Discharge Date"] - df_healthcare["Date of Admission"]
).dt.days

#%%
# Categorize Length of Stay: Short, Medium, Long
bins = [-1, 3, 7, np.inf]
labels = ["Short", "Medium", "Long"]
df_healthcare["Stay Category"] = pd.cut(df_healthcare["Length of Stay"], bins=bins, labels=labels)

#%%
# Extract Admission Month and Day of the Week
df_healthcare["Admission Month"] = df_healthcare["Date of Admission"].dt.month
df_healthcare["Admission Day"] = df_healthcare["Date of Admission"].dt.day_name()

#%%
# Group Age into categories: Child, Adult, Senior
age_bins = [0, 17, 64, np.inf]
age_labels = ["Child", "Adult", "Senior"]
df_healthcare["Age Group"] = pd.cut(df_healthcare["Age"], bins=age_bins, labels=age_labels)

#%%
# Categorize Medicines: Penicillin as "antibiotic", others as "pain management"
df_healthcare["Medicine Category"] = df_healthcare["Medication"].apply(
    lambda x: "antibiotic" if x.lower() == "penicillin" else "pain management"
)

#%%
# Categorize Room Number into Floors (e.g., 200s = floor 2)
df_healthcare["Floor Number"] = df_healthcare["Room Number"] // 100

#%%
# Display the updated dataset with new features
df_healthcare_head_with_features = df_healthcare.head()
print(df_healthcare_head_with_features)

%##