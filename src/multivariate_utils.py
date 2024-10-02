from dataclasses import dataclass, field
from typing import Union, List, Optional
import numpy as np
import pandas as pd


# Helper function to ensure a parameter is a list
def ensure_list(param, n_items):
    """Convert a single value into a list of length n_items if it's not already a list."""
    if isinstance(param, (int, float)):
        return [param] * n_items
    return param


# Helper function to ensure cluster centers are sufficiently separated
def is_far_enough(new_centroid, existing_centroids, min_distance):
    """Check if the new centroid is far enough from all existing centroids."""
    for centroid in existing_centroids:
        if np.linalg.norm(new_centroid - centroid) < min_distance:
            return False
    return True


# Data class to hold numerical feature generation info
@dataclass
class NumericalInfo:
    n_targets: int  # Number of targets (or clusters per target)
    n_features: int  # Number of informative features
    n_clusters_per_target: Union[int, List[int]] = 1  # Clusters per target
    cluster_dist: str = (
        "normal"  # Distribution type: "normal", "uniform", "lognormal"
    )
    between_target_sep: float = (
        3.0  # Separation between target centroids (in standard deviations)
    )
    within_target_sep: float = (
        1.0  # Separation between clusters within each target
    )
    cluster_spread: float = (
        1.0  # Spread for generating data points around cluster centers
    )

    # Optional distribution-specific parameters
    mean: float = 0  # Mean for normal and log-normal distributions
    std_dev: float = (
        1  # Standard deviation for normal and log-normal distributions
    )
    min_value: float = 0  # Minimum value for uniform distribution
    max_value: float = 1  # Maximum value for uniform distribution
    shape: float = 1  # Shape parameter for log-normal distribution (optional)
    scale: float = 1  # Scale parameter for log-normal distribution (optional)

    def validate_parameters(self):
        """Validate parameters depending on the selected distribution."""
        if self.cluster_dist == "uniform" and self.min_value >= self.max_value:
            raise ValueError(
                "For uniform distribution, min_value must be less than max_value."
            )
        if self.cluster_dist == "lognormal" and (
            self.shape <= 0 or self.scale <= 0
        ):
            raise ValueError(
                "For log-normal distribution, shape and scale must be positive."
            )


# Data class to hold correlation info with target
@dataclass
class TargetCorrelationInfo:
    specific_or_general: str  # Whether the correlation is 'specific' to target or 'general' to all
    magnitude: float  # Correlation magnitude
    correlation_type: (
        str  # Type of correlation ('exp', 'log', 'poly', or 'normal')
    )


@dataclass
class CategoricalInfo:
    n_categorical: int  # Number of categorical variables
    categories_per_variable: List[
        int
    ]  # Number of categories for each variable
    one_hot_encode: bool = True  # Whether to apply one-hot encoding
    category_distributions: Optional[List[List[float]]] = (
        None  # Category distribution, default is uniform
    )
    n_samples: Optional[int] = (
        None  # Optional, used only if no DataFrame is supplied
    )


# Helper functions for correlation (used for categorical, interaction, and numerical)
def apply_specific_correlation(interaction_term, target_label, corr_info):
    """Apply class-specific correlation to the interaction term."""
    if corr_info.correlation_type == "exp":
        interaction_term += corr_info.magnitude * np.exp(target_label)
    elif corr_info.correlation_type == "log":
        interaction_term += corr_info.magnitude * np.log(
            np.abs(target_label) + 1
        )
    elif corr_info.correlation_type == "poly":
        interaction_term += corr_info.magnitude * np.power(target_label, 2)
    else:
        interaction_term += corr_info.magnitude * np.random.normal(
            loc=target_label, scale=0.1
        )
    return interaction_term


def apply_general_correlation(interaction_term, target_label, corr_info):
    """Apply general correlation to the interaction term."""
    if corr_info.correlation_type == "exp":
        interaction_term += corr_info.magnitude * np.exp(target_label)
    elif corr_info.correlation_type == "log":
        interaction_term += corr_info.magnitude * np.log(
            np.abs(target_label) + 1
        )
    elif corr_info.correlation_type == "poly":
        interaction_term += corr_info.magnitude * np.power(target_label, 2)
    else:
        interaction_term += corr_info.magnitude * np.random.normal(
            loc=target_label, scale=0.1
        )
    return interaction_term


def apply_categorical_correlation(categorical_values, target_label, corr_info):
    """
    Apply correlation between categorical values and the target.

    Parameters:
    ----------
    categorical_values : array-like
        The generated categorical values.
    target_label : array-like
        The target labels (outcome).
    corr_info : TargetCorrelationInfo
        Information about how the target and categorical values are correlated.

    Returns:
    -------
    correlated_values : array-like
        Adjusted categorical values correlated with the target.
    """
    correlated_values = categorical_values.copy()

    for i, label in enumerate(np.unique(target_label)):
        idx = target_label == label
        prob = (
            np.random.rand(np.sum(idx)) < corr_info.magnitude
        )  # Magnitude controls probability of correlation
        correlated_values[idx] = np.where(
            prob,
            np.random.choice(
                categorical_values[idx], size=np.sum(idx)
            ),  # Correlate category values
            categorical_values[idx],  # Keep original values
        )

    return correlated_values


# Main function to generate numerical features
def add_numerical_features(
    df: Optional[pd.DataFrame],
    numerical_info: NumericalInfo,
    n_samples: Optional[int] = None,
    target_label: Optional[np.ndarray] = None,
    is_target_correlated: bool = True,
    target_corr_info: Optional[dict] = None,
):
    """Add numerical features to an existing DataFrame or create a new one."""
    numerical_info.validate_parameters()

    if df is None:
        if n_samples is None:
            raise ValueError(
                "You must provide n_samples if no DataFrame is supplied."
            )
        df = pd.DataFrame()  # Create a new DataFrame if none is provided
    else:
        n_samples = df.shape[
            0
        ]  # Infer the number of samples if df is provided

    n_clusters_per_target = ensure_list(
        numerical_info.n_clusters_per_target, numerical_info.n_targets
    )
    cluster_spread = ensure_list(
        numerical_info.cluster_spread, sum(n_clusters_per_target)
    )

    # Generate the centroids for the targets
    target_centroids = []
    for target_idx in range(numerical_info.n_targets):
        while True:
            new_centroid = np.random.normal(
                0,
                numerical_info.between_target_sep,
                size=(numerical_info.n_features,),
            )
            if len(target_centroids) == 0 or is_far_enough(
                new_centroid,
                target_centroids,
                numerical_info.between_target_sep,
            ):
                target_centroids.append(new_centroid)
                break

    target_centroids = np.stack(target_centroids)

    # Generate data points around the target centroids
    cluster_idx = 0
    for target_idx, centroid in enumerate(target_centroids):
        num_clusters = n_clusters_per_target[target_idx]
        for _ in range(num_clusters):
            cluster_center = np.random.normal(
                centroid,
                numerical_info.within_target_sep,
                size=(numerical_info.n_features,),
            )
            samples = n_samples // sum(
                n_clusters_per_target
            )  # Uniformly distribute samples across clusters

            data_points = generate_cluster_points(
                samples,
                numerical_info.n_features,
                cluster_center,
                numerical_info,
            )

            # Apply correlation with target label if specified
            if (
                is_target_correlated
                and target_label is not None
                and target_corr_info is not None
            ):
                corr_info = target_corr_info.get(
                    (target_idx, cluster_idx), None
                )
                if corr_info:
                    if corr_info.specific_or_general == "specific":
                        data_points = apply_specific_correlation(
                            data_points, target_label, corr_info
                        )
                    else:
                        data_points = apply_general_correlation(
                            data_points, target_label, corr_info
                        )

            # Add generated points to the DataFrame
            for i in range(numerical_info.n_features):
                df[f"Feature_{i + cluster_idx}"] = data_points[:, i]

            cluster_idx += 1

    return df


# Main function to add categorical features
def add_categorical_features(
    categorical_info: CategoricalInfo,
    df=None,
    target_label=None,
    is_target_correlated: bool = True,
    target_corr_info: dict = None,
):
    """Add categorical features to a DataFrame or create a new one if no DataFrame is provided."""

    # Determine number of samples
    if df is None:
        if categorical_info.n_samples is None:
            raise ValueError(
                "Must provide 'n_samples' when no DataFrame is supplied."
            )
        n_samples = categorical_info.n_samples
        df = pd.DataFrame()
    else:
        n_samples = df.shape[0]

    # Handle default distributions (uniform) if not provided
    if categorical_info.category_distributions is None:
        category_distributions = [
            [1.0 / categories] * categories
            for categories in categorical_info.categories_per_variable
        ]
    else:
        category_distributions = categorical_info.category_distributions

    # Generate categorical features
    for i in range(categorical_info.n_categorical):
        categories = categorical_info.categories_per_variable[i]
        dist = category_distributions[i]
        categorical_values = np.random.choice(
            categories, size=n_samples, p=dist
        )

        # Correlate with target labels if specified
        if is_target_correlated and target_label is not None:
            corr_info


# Helper function for concatenating category and numerical values for mixed interactions
def concatenate_category_numerical(categorical_feature, numerical_feature):
    """
    Concatenate categorical labels and numerical values to create a new mixed interaction feature.

    Parameters:
    ----------
    categorical_feature : pd.Series
        Categorical feature (e.g., product type).
    numerical_feature : pd.Series
        Numerical feature (e.g., price).

    Returns:
    -------
    pd.Series
        Concatenated interaction feature (e.g., "Product_A_100").
    """
    return (
        categorical_feature.astype(str) + "_" + numerical_feature.astype(str)
    )


# Main function to add interaction features
def add_interaction_features(
    df,
    interaction_info_list,  # List of InteractionInfo objects
    target_label=None,  # Target or class label for correlation purposes
    is_target_correlated=False,  # Whether the interaction is correlated with the target
    target_corr_info=None,  # Dictionary with TargetCorrelationInfo for each interaction
    interaction_only=False,  # Whether to keep only interaction terms
):
    """
    Add interaction features to the DataFrame, including categorical, numerical, and mixed interactions.

    Parameters:
    ----------
    df : DataFrame
        Input data containing the features.
    interaction_info_list : list
        List of InteractionInfo objects specifying feature pairs and interaction types.
    target_label : Series, optional
        Target or class label to use for correlations.
    is_target_correlated : bool, default=False
        Whether the interaction terms are correlated with the target (class/outcome).
    target_corr_info : dict, optional
        Dictionary where keys are feature pairs and values are TargetCorrelationInfo objects.
    interaction_only : bool, default=False
        Whether to return only the interaction terms or to keep the original features as well.

    Returns:
    -------
    df : DataFrame
        DataFrame with interaction features added (or replacing the original features if interaction_only=True).
    """
    interaction_df = pd.DataFrame()

    for interaction_info in interaction_info_list:
        feature1, feature2 = interaction_info.features
        interaction_type = interaction_info.interaction_type

        interaction_name = (
            f"Interaction_{feature1}_{feature2}_{interaction_type}"
        )

        if interaction_type == "multiplicative":
            # Handle both categorical and numerical features
            if (
                df[feature1].dtype == "object"
                or df[feature2].dtype == "object"
            ):
                interaction_term = concatenate_category_numerical(
                    df[feature1], df[feature2]
                )
            else:
                interaction_term = df[feature1] * df[feature2]

        elif interaction_type == "additive":
            interaction_term = df[feature1] + df[feature2]

        elif interaction_type == "non-linear":
            interaction_term = np.power(df[feature1], 2) + np.log(
                np.abs(df[feature2]) + 1
            )

        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")

        # Handle target correlation if applicable
        if (
            is_target_correlated
            and target_label is not None
            and target_corr_info
        ):
            corr_info = target_corr_info.get((feature1, feature2), None)

            if corr_info:
                if corr_info.specific_or_general == "specific":
                    interaction_term = apply_specific_correlation(
                        interaction_term, target_label, corr_info
                    )
                else:
                    interaction_term = apply_general_correlation(
                        interaction_term, target_label, corr_info
                    )

        interaction_df[interaction_name] = interaction_term

    if interaction_only:
        return interaction_df
    else:
        return pd.concat([df, interaction_df], axis=1)
