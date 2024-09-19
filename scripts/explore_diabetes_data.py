# %%
# Standard library imports
from pathlib import Path
import sys
import os
import time

# Third-party imports
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pyprojroot.here import here
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

# Adjust the Python path for local application/library imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local application/library imports
from src.pd_pca_utils import apply_pca, visualize_pca
from src.ordinal_ui import prompt_user_for_recode  # Import the UI module


def auto_encode_columns_with_ui(df):
    """
    Main function to handle data recoding. It prompts the user for
    each non-numeric column and allows them to recode it as ordinal if
    necessary.

    Parameters:
        df (DataFrame): The input dataframe.

    Returns:
        df (DataFrame): The modified dataframe with recoded columns.
        ordinal_mappings (dict): Dictionary of the ordinal mappings for each
        column.
    """
    ordinal_mappings = {}
    for col in df.columns:
        if df[col].dtype == "object":
            unique_vals = df[col].dropna().unique()

            # Show a UI prompt for each non-numeric column
            ordered_values, user_recode = prompt_user_for_recode(
                col, unique_vals
            )

            if (
                user_recode == "yes"
            ):  # If user wants to recode, replace the original column
                ordinal_mappings[col] = {
                    val: i for i, val in enumerate(ordered_values)
                }
                df[col] = df[col].map(ordinal_mappings[col])
                print(
                    f"Replaced original column '{col}' with ordinal mapping."
                )
            else:
                print(f"'{col}' was not recoded as ordinal.")

    return df, ordinal_mappings


# Ensure I am using an interactive backend:
matplotlib.use("Qt5Agg")  # or 'TkAgg'

# Establish important project paths
dir_base = here()  # ./geisinger
dir_data = Path(dir_base, "data/diabetes_130_US_hospitals_1999_2008")
dir_results = Path(dir_base, "results/")
dir_scripts = Path(dir_base, "scripts/")
dir_scr = Path(dir_base, "scr/")

# %%
# Load the dataset (CSV format)
df_data = pd.read_csv(Path(dir_data, "data.csv"))

#
df_cleaned, ordinal_mappings = auto_encode_columns_with_ui(df_data)

# Review the mappings
print("Final Ordinal Mappings:")
for col, mapping in ordinal_mappings.items():
    print(f"{col}: {mapping}")

# Load the key file
df_key = pd.read_csv(Path(dir_data, "key.csv"))

# Create a mask to locate the blank rows
mask = df_key["admission_type_id"].isna()

# Identify the indices of blank rows
blank_indices = mask[mask].index.tolist()

# First table: from the top until the first blank row
id_admission_type = df_key.loc[: blank_indices[0] - 1].reset_index(drop=True)

# Second table: from the first blank row to the second blank row
id_discharge_disposition = df_key.loc[
    blank_indices[0] + 1 : blank_indices[1] - 1
].reset_index(drop=True)
id_discharge_disposition.columns = id_discharge_disposition.iloc[
    0
]  # Set the first row as the header
id_discharge_disposition = id_discharge_disposition.drop(0).reset_index(
    drop=True
)  # Drop the header row

# Third table: from the second blank row to the end of the file
id_admission_source = df_key.loc[blank_indices[1] + 1 :].reset_index(drop=True)
id_admission_source.columns = id_admission_source.iloc[
    0
]  # Set the first row as the header
id_admission_source = id_admission_source.drop(0).reset_index(
    drop=True
)  # Drop the header row

# %%
# Calculate the percentage of missing values in each column
missing_percentage = df_data.isnull().mean() * 100

# Display the missing percentage for all columns
print(missing_percentage)
