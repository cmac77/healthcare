"""
synthetic_multivar_data_pipeline.py

This script generates synthetic multivariate data, including numerical, categorical, and interaction features, using the `synthetic_multivar_data_utils` module. It also builds and evaluates a machine learning model (XGBoost) to classify the synthetic data. The pipeline includes data generation, feature engineering, model training, and performance evaluation.

The script follows these main steps:
1. **Numerical Feature Creation**: Generates numerical features by defining target clusters and centroids using specified parameters (e.g., number of features, clusters, separation).
2. **Categorical Feature Addition**: Adds categorical features, with an option to one-hot encode the categories.
3. **Interaction Feature Addition**: Introduces interaction terms between features (multiplicative, additive, and non-linear interactions).
4. **XGBoost Model Training and Evaluation**: Trains an XGBoost classifier on the generated data and evaluates model accuracy.

Main Components:
----------------
- **Numerical Feature Creation**: Uses `add_numerical_features` to generate numerical data based on specified target clusters.
- **Categorical Feature Addition**: Uses `add_categorical_features` to append categorical features to the DataFrame.
- **Interaction Features**: Adds interaction terms between specified features using `add_interaction_features`.
- **XGBoost Classifier**: Trains an XGBoost model on the synthetic dataset and evaluates model performance using accuracy score.

Example Usage:
--------------
```python
# Run the synthetic data generation process and model training
python synthetic_data_pipeline.py
"""

# %% Imports and Setup
import sys
from pathlib import Path

# Get the absolute path to the project root and add the src folder to sys.path
project_root = Path(__file__).resolve().parent.parent.parent
src_path = project_root / "src"
sys.path.append(str(src_path))

# Import the module
import synthetic_multivar_data_utils as smdu

# %% Step 1: Create Numerical Features
# Define the numerical information
numerical_info = smdu.NumericalInfo(
    n_targets=3,  # Number of target clusters
    n_features=5,  # Number of informative features
    n_clusters_per_target=[1, 2, 1],  # Clusters per target
    between_target_sep=3.0,  # Separation between target centroids
    within_target_sep=1.0,  # Separation within clusters
    cluster_dist="normal",  # Distribution for numerical data generation
    cluster_spread=1.0,  # Spread around cluster centers
    mean=0,  # Mean for normal distribution
    std_dev=1,  # Standard deviation for normal distribution
)

# Generate the numerical features
df_numerical, target_labels, target_centroids, cluster_centroids = (
    smdu.add_numerical_features(
        df=None,  # Create a new DataFrame
        numerical_info=numerical_info,
        n_samples=100,  # Number of samples
        target_label=None,  # No correlation with target yet
        is_target_correlated=False,  # No target correlation for now
    )
)

print("Numerical features created:")
print(df_numerical.head())

# %% Step 2: Add Categorical Features
# Define the categorical information
categorical_info = smdu.CategoricalInfo(
    n_categorical=3,  # Number of categorical variables
    categories_per_variable=[
        4,
        3,
        5,
    ],  # Number of categories for each variable
    one_hot_encode=True,  # One-hot encode the categories
    category_distributions=None,  # Default uniform distribution
)

# Add the categorical features to the DataFrame
df_categorical = smdu.add_categorical_features(
    categorical_info=categorical_info,
    df=df_numerical,  # Append to the numerical DataFrame
    target_label=None,  # No correlation with target for now
    is_target_correlated=False,  # No correlation with target for now
)

print("Numerical and categorical features combined:")
print(df_categorical.head())


# %% Step 3: Add Interaction Features
# Update the interaction_info_list to use the correct column names
interaction_info_list = [
    smdu.InteractionInfo(
        features=("Num_0", "Num_1"), interaction_type="multiplicative"
    ),
    smdu.InteractionInfo(
        features=("Num_2", "Num_3"), interaction_type="additive"
    ),
    smdu.InteractionInfo(
        features=("Num_1", "Cat_2_0"),
        interaction_type="non-linear",
    ),
]

# Add interaction features
df_full = smdu.add_interaction_features(
    df=df_categorical,  # Use the DataFrame with numerical and categorical features
    interaction_info_list=interaction_info_list,  # Specify interaction pairs and types
    target_label=None,  # No target correlation
    is_target_correlated=False,  # No correlation with target for now
)

print("DataFrame with interaction features added:")
print(df_full.head())

# %%
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Step 1: Create a synthetic target label if you haven't already
n_samples = df_full.shape[0]
target = np.random.randint(0, 3, size=n_samples)  # 3 clusters/target classes

# Add target to your DataFrame
df_full["target_label"] = target

# Step 2: Split the data into training and test sets
X = df_full.drop(columns=["target_label"])
y = df_full["target_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Step 3: Initialize and train the XGBoost model
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="mlogloss")

model.fit(X_train, y_train)

# Step 4: Make predictions and evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Model accuracy: {accuracy * 100:.2f}%")


# %%
