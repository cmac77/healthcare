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
    n_clusters_per_target=2,  # Clusters per target
    cluster_dist="normal",  # Distribution for numerical data generation
    between_target_sep=3.0,  # Separation between target centroids
    within_target_sep=1.0,  # Separation within clusters
    cluster_spread=1.0,  # Spread around cluster centers
    mean=0,  # Mean for normal distribution
    std_dev=1,  # Standard deviation for normal distribution
)

# Generate the numerical features
df_numerical = smdu.add_numerical_features(
    df=None,  # Create a new DataFrame
    numerical_info=numerical_info,
    n_samples=100,  # Number of samples
    target_label=None,  # No correlation with target yet
    is_target_correlated=False,  # No target correlation for now
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
df_with_interactions = smdu.add_interaction_features(
    df=df_categorical,  # Use the DataFrame with numerical and categorical features
    interaction_info_list=interaction_info_list,  # Specify interaction pairs and types
    target_label=None,  # No target correlation
    is_target_correlated=False,  # No correlation with target for now
)

print("DataFrame with interaction features added:")
print(df_with_interactions.head())

# %%
# Check the columns of the df_categorical
print(df_categorical.columns)
