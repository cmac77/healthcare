# Standard library imports
from pathlib import Path
import sys
import pickle

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

# Append the parent directory of "src"
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Local application/library imports
from src.pd_pca_utils import apply_pca, visualize_pca
from src.data_frame_column_cleaner import DataFrameColumnCleaner
from src.data_frame_column_editor import DataFrameColumnEditor

# Ensure I am using an interactive backend:
matplotlib.use("Qt5Agg")  # or 'TkAgg'

# Establish important project paths
dir_base = here()  # ./geisinger
dir_data = Path(dir_base, "data/diabetes_130_US_hospitals_1999_2008")
dir_results = Path(dir_base, "results/diabetes_130_US_hospitals_1999_2008")
dir_scripts = Path(dir_base, "scripts/diabetes_130_US_hospitals_1999_2008")
dir_scr = Path(dir_base, "scr/")

# %% Load and clean relevant data
# Load the dataset (CSV format)
df_data = pd.read_csv(Path(dir_data, "data.csv"))
# df_data = df_data[["race", "age", "medical_specialty", "gender"]]

# Path to save the cleaned dataframe
df_cleaned_pickle_path = Path(dir_data, "df_cleaned.pkl")

# %% Check if the cleaned dataframe already exists
if True:  # df_cleaned_pickle_path.exists():
    #     # Load the saved cleaned dataframe from pickle
    #     with open(df_cleaned_pickle_path, "rb") as f:
    #         df_cleaned = pickle.load(f)
    #     print("Loaded previously cleaned dataframe.")
    # else:
    # Using global defaults for all columns
    cleaner = DataFrameColumnCleaner()

    # Clean the DataFrame data without column-specific settings
    # (e.g., removing whitespace, converting to numeric, etc.)
    df_data_clean = cleaner.process_dataframe(df_data.copy())

    # Now step through df columns and make any edits.
    editor = DataFrameColumnEditor(df_data_clean)  # Launch the UI window

    # Access the edited DataFrame after the window is closed
    df_data_clean = editor.get_edited_df()

    # Now, df_cleaned will contain the updated DataFrame
    print("Updated DataFrame:", df_data_clean)

    # # Save the cleaned dataframe for future use (pickle format)
    # with open(df_cleaned_pickle_path, "wb") as f:
    #     pickle.dump(df_data_clean, f)
    # print(f"Saved cleaned dataframe to {df_cleaned_pickle_path}")

# %%
# Load the key file. Key file contains three separate tables.
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
id_discharge_disposition.columns = id_discharge_disposition.iloc[0]

# Set the first row as the header
id_discharge_disposition = id_discharge_disposition.drop(0).reset_index(
    drop=True
)  # Drop the header row

# Third table: from the second blank row to the end of the file
id_admission_source = df_key.loc[blank_indices[1] + 1 :].reset_index(drop=True)
id_admission_source.columns = id_admission_source.iloc[0]

# Set the first row as the header
id_admission_source = id_admission_source.drop(0).reset_index(
    drop=True
)  # Drop the header row

# # %%
# # Calculate the percentage of missing values in each column
# missing_percentage = df_data.isnull().mean() * 100

# # Display the missing percentage for all columns
# print(missing_percentage)

# %%
