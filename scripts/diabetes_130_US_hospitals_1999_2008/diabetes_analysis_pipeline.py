"""
diabetes_analysis_pipeline.py

This module provides a complete data pipeline for cleaning, processing, and modeling a healthcare dataset, including training a neural network using PyTorch and evaluating its performance. The pipeline covers data preprocessing, feature engineering, dimensionality reduction with PCA, and training a neural network to predict hospital readmission.

The pipeline consists of the following major steps:
1. **Data Loading and Cleaning**: Loading CSV files, applying cleaning rules (e.g., whitespace removal, numeric conversion), and saving/loading the cleaned DataFrame using pickle.
2. **Feature Engineering**: Mapping categorical columns, converting numerical columns to categorical, handling missing values, and generating one-hot encodings for categorical variables.
3. **Dimensionality Reduction**: Applying PCA using PyTorch and visualizing the transformed data.
4. **Train-Test Split**: Splitting the data into training and test sets.
5. **Modeling**: Training a neural network in PyTorch with GPU acceleration, using weighted loss to account for class imbalance.
6. **Evaluation**: Testing the model's performance and calculating the accuracy on the test set.

Main Components:
----------------
- **DataFrameColumnCleaner**: Cleans a Pandas DataFrame by removing commas, stripping whitespace, and converting values to numeric types.
- **DataFrameColumnEditor**: Launches a Tkinter UI to allow manual edits to the DataFrame.
- **PCA**: Applies dimensionality reduction using PCA, with the option to accelerate using PyTorch.
- **SimpleNN**: Defines a simple neural network for classifying hospital readmissions based on preprocessed features.

Example Usage:
--------------
```python
# Step 1: Load, clean, and preprocess the dataset
df_cleaned = load_and_clean_data()

# Step 2: One-hot encode categorical features and standardize numerical features
X, y = prepare_data_for_modeling(df_cleaned)

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Initialize and train a neural network using PyTorch
model = SimpleNN(input_size=X_train.shape[1], hidden_size=64, output_size=3).to(device)
train_model(model, X_train, y_train, X_test, y_test)

# Step 5: Evaluate the model on the test set
test_accuracy = evaluate_model(model, X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}%")
"""

# Standard library imports
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path
import pickle
import sys


# Third-party imports
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pyprojroot.here import here
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

# Append the parent directory of "src"
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

# Local application/library imports
from src.data_frame_column_cleaner import DataFrameColumnCleaner
from src.data_frame_column_editor import DataFrameColumnEditor

# from export_py2txt import write_current_file_to_txt

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


# %%
# Path to save the cleaned dataframe
df_cleaned_pickle_path = Path(dir_data, "df_cleaned.pkl")


# %% Check if the cleaned dataframe already exists
if df_cleaned_pickle_path.exists():
    # Load the saved cleaned dataframe from pickle
    with open(df_cleaned_pickle_path, "rb") as f:
        df_cleaned = pickle.load(f)
    print("Loaded previously cleaned dataframe.")
else:
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

    # Save the cleaned dataframe for future use (pickle format)
    with open(df_cleaned_pickle_path, "wb") as f:
        pickle.dump(df_data_clean, f)
    print(f"Saved cleaned dataframe to {df_cleaned_pickle_path}")

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

# %%
# Step 1: Data Preparation
df = df_cleaned.copy()  # Create a copy of the cleaned DataFrame

# Remap values in the 'A1Cresult' column. Set missing data to zero.
df["A1Cresult"] = df["A1Cresult"].map({0: 1, 1: 2, 2: 3})  # Remap values
df["A1Cresult"] = df["A1Cresult"].fillna(0)  # Fill NaNs with 0

# Remap values in the 'max_glu_serum' column. Set missing data to zero.
df["max_glu_serum"] = df["max_glu_serum"].map(
    {"Norm": 1, "High": 2}
)  # Remap values
df["max_glu_serum"] = df["max_glu_serum"].fillna(0)  # Fill NaNs with 0

# Remap values in the 'readmitted' column. Target variable with ordinal mapping.
df["readmitted"] = df["readmitted"].map({"NO": 0, "SHORT": 1, "LONG": 2})

# Drop unnecessary columns or columns with too many missing values (e.g., 'weight')
columns_to_drop = ["weight", "encounter_id", "patient_nbr"]
df.drop(columns_to_drop, axis=1, inplace=True)  # Drop the columns

# Identify numerical columns (integer and float types) and store in columns_numerical list
columns_numerical = df.select_dtypes(
    include=["float64", "int64"]
).columns.to_list()

# Identify categorical columns as those that are not numerical
columns_categorical = [
    col for col in df.columns if col not in columns_numerical
]

# These columns should be reassigned from numerical to categorical
columns_to_reassign = [
    "A1Cresult",
    "max_glu_serum",
    "admission_type_id",
    "discharge_disposition_id",
    "admission_source_id",
]

# Convert specified columns from numerical to categorical
for col in columns_to_reassign:
    if col in df.columns:  # Ensure column exists
        df[col] = df[col].astype("category")

# Remove columns_to_reassign from the numerical column list
columns_numerical = [
    col for col in columns_numerical if col not in columns_to_reassign
]

# Prepend columns_to_reassign to the beginning of columns_categorical
columns_categorical = columns_to_reassign + columns_categorical

# Verify the updated lists for numerical and categorical columns
print(f"Numerical columns: {columns_numerical}")
print(f"Categorical columns: {columns_categorical}")

# Ensure columns_to_reassign appear first in the DataFrame
remaining_columns = [
    col for col in df.columns if col not in columns_to_reassign
]
df = df[columns_to_reassign + remaining_columns]

# %%
### Step 2: One-hot encode categorical variables ###
df_dummies = pd.get_dummies(df[columns_categorical], drop_first=True)

### Step 3: Select numerical columns ###
df_numerical = df[columns_numerical]

### Step 4: Concatenate numerical and one-hot encoded categorical columns ###
df_new = pd.concat([df_dummies, df_numerical], axis=1)

# Ensure 'readmitted' column exists in the new DataFrame before splitting
if "readmitted" in df_new.columns:
    # Separate features (X) and target (y)
    X = df_new.drop(columns=["readmitted"]).values  # Features (as NumPy array)
    y = df_new["readmitted"].values  # Target (as NumPy array)
else:
    print("Error: 'readmitted' column not found in df_new.")


# %%
# Function to perform PCA using PyTorch with GPU acceleration

# Step 1: Convert df_new to a NumPy array (drop target 'readmitted')
X_np = df_new.drop(columns=["readmitted"]).values

# Step 2: Standardize using TensorFlow
X_scaled_tf = standardize_tensorflow(X_np)

# Step 3: Convert TensorFlow tensor to PyTorch tensor and move to GPU
X_tensor = torch.tensor(X_scaled_tf.numpy(), dtype=torch.float32).cuda()

# Apply PCA using PyTorch (the rest of the code remains unchanged)
n_components = 2
X_pca_torch, explained_variance_ratio = pca_torch(
    X_tensor, n_components=n_components
)

# Move PCA results back to CPU for plotting
X_pca_cpu = X_pca_torch.cpu().detach().numpy()

# Step 4: Visualize the PCA-transformed data
y_color = df_new["readmitted"].values  # For coloring

plt.figure(figsize=(8, 6))
scatter = plt.scatter(
    X_pca_cpu[:, 0],
    X_pca_cpu[:, 1],
    c=y_color,
    cmap="plasma",
    edgecolor="k",
    s=40,
)
plt.xlabel(
    f"Principal Component 1 ({explained_variance_ratio[0]:.2f} variance)"
)
plt.ylabel(
    f"Principal Component 2 ({explained_variance_ratio[1]:.2f} variance)"
)
plt.colorbar(scatter, label="Readmitted")
plt.title(f"PCA of Sampled df_new (Colored by 'readmitted')")
plt.show()


# %%
# %%
# Step 3: Plot the PCA-transformed data
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap="viridis", edgecolor="k", s=40)
plt.colorbar(label="Readmitted")
plt.xlabel(
    f"Principal Component 1 ({pca.explained_variance_ratio_[0]:.2f} variance)"
)
plt.ylabel(
    f"Principal Component 2 ({pca.explained_variance_ratio_[1]:.2f} variance)"
)
plt.title("PCA of df_new (Scaled Data, 2D Projection)")
plt.show()


# %%
# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 6: Standardize the data (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_train_clean = np.nan_to_num(X_train, nan=0)  # Ensure no NaNs remain
X_test = scaler.transform(X_test)
X_test_clean = np.nan_to_num(X_test, nan=0)

# Convert the standardized data into PyTorch tensors (needed for model training)
X_train_tensor = torch.tensor(X_train_clean, dtype=torch.float32)
y_train_tensor = torch.tensor(
    y_train, dtype=torch.long
)  # Use long for classification
X_test_tensor = torch.tensor(X_test_clean, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# %%
# Step 7: Create datasets and DataLoader for batch processing
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

# DataLoader for batching during training and testing
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# %%
# Step 8: Check if CUDA is available for GPU training, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# %%
# Step 9: Define a simple neural network with one hidden layer and dropout
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout_prob=0.5):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First hidden layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second hidden layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Output layer
        self.relu = nn.ReLU()  # Activation function
        self.dropout = nn.Dropout(
            dropout_prob
        )  # Dropout layer to prevent overfitting

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout to prevent overfitting

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)  # Apply dropout again after second hidden layer

        x = self.fc3(x)  # Final output layer
        return x


# %%
# Step 10: Initialize the model, loss function (CrossEntropy), and optimizer (Adam)
input_size = X_train.shape[1]  # Number of input features
hidden_size = 64  # Hidden layer size
output_size = 3  # Number of output classes (NO, SHORT, LONG)

model = SimpleNN(input_size, hidden_size, output_size).to(
    device
)  # Move model to GPU if available

# Compute class weights (inverse of class frequency)
class_counts = df["readmitted"].value_counts().sort_index()
class_weights = 1.0 / torch.tensor(class_counts.values, dtype=torch.float32)

# Move class_weights to the device (GPU/CPU)
class_weights = class_weights.to(device)

# Use CrossEntropyLoss with class weights
criterion = nn.CrossEntropyLoss(weight=class_weights)
# criterion = nn.CrossEntropyLoss()  # Loss function for classification

optimizer = optim.Adam(
    model.parameters(), lr=0.001
)  # Adam optimizer with learning rate 0.001

# Print NaN check (if any)
print(f"Number of NaNs in X_train: {np.isnan(X_train).sum()}")
print(f"Number of NaNs in y_train: {np.isnan(y_train).sum()}")

# %%
# Step 11: Training loop for the model
num_epochs = 20  # Number of epochs (iterations over the full dataset)
for epoch in range(num_epochs):
    model.train()  # Set model to training mode
    running_loss = 0.0
    correct = 0
    total = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(
            device
        )  # Move data to GPU if available

        optimizer.zero_grad()  # Clear gradients
        outputs = model(X_batch)  # Forward pass
        loss = criterion(outputs, y_batch)  # Compute loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

    accuracy = 100 * correct / total
    print(
        f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss:.4f}, Accuracy: {accuracy:.2f}%"
    )

# %%
# Step 12: Testing loop (evaluation mode)
model.eval()  # Set model to evaluation mode
correct = 0
total = 0

with torch.no_grad():  # No need to compute gradients during evaluation
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(
            device
        )  # Move data to GPU if available
        outputs = model(X_batch)  # Forward pass
        _, predicted = torch.max(outputs.data, 1)
        total += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

# Compute final test accuracy
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")

# %%
