"""
correlation_features.py

This module handles the generation of correlated synthetic data, allowing for the addition of both numerical and
categorical correlations. It uses copulas to fit and generate data with specified correlation structures.

Classes:
    - CorrelationFeatures: Class to define and apply correlation structures for synthetic data generation.

Functions:
    - add_correlation(data: pd.DataFrame): Adds correlated relationships between variables in the provided dataset.

Example Usage:
    # Initialize CorrelationFeatures with predefined correlations:
    correlation_feature_generator = CorrelationFeatures(correlations=[("var1", "var2", 0.8), ("var3", "var4", -0.6)])
    
    # Create a random DataFrame:
    data = pd.DataFrame({
        'var1': np.random.randn(100),
        'var2': np.random.randn(100),
        'var3': np.random.randn(100),
        'var4': np.random.randn(100)
    })
    
    # Add correlations to the DataFrame:
    correlated_data = correlation_feature_generator.add_correlation(data)
    
    # Output the correlated dataset:
    print(correlated_data.corr())  # Displays the correlation matrix

Requirements:
    pandas, numpy, copulas
"""

# %%
from dataclasses import dataclass
import pandas as pd
import numpy as np
from copulas.multivariate import GaussianMultivariate
from pandas.api.types import is_numeric_dtype
from typing import Dict, Tuple


@dataclass
class CorrelationFeatures:
    """
    A class for handling and applying correlations between features in synthetic data generation.

    Attributes:
        correlations (Dict[Tuple[str, str], Tuple[float, str]]):
            A dictionary where keys are pairs of feature names (e.g., ('feature_1', 'feature_2')), and
            values are tuples of (magnitude, correlation function). The correlation function can be "linear",
            "log", "exp" for numerical correlations, or "categorical" for categorical correlations.

    Methods:
        validate_magnitude(magnitude: float, correlation_type: str):
            Validates the magnitude of correlation depending on whether it is a numerical-numerical
            or a categorical correlation.

        apply_all_linear_correlations(data: pd.DataFrame) -> pd.DataFrame:
            Applies linear correlations between numerical features in the dataset by adjusting the
            covariance matrix.

        apply_all_nonlinear_correlations(data: pd.DataFrame) -> pd.DataFrame:
            Applies nonlinear (logarithmic or exponential) correlations between numerical features
            using a copula transformation.

        apply_categorical_correlation(data: pd.DataFrame, col_1: str, col_2: str, magnitude: float) -> pd.DataFrame:
            Applies a correlation between two categorical features by adjusting their contingency table.

        apply_categorical_numerical_correlation(data: pd.DataFrame, col_1: str, col_2: str, magnitude: float) -> pd.DataFrame:
            Applies a correlation between a categorical feature and a numerical feature by adjusting
            the numerical values based on the categorical groups.

        apply_correlations(data: pd.DataFrame) -> pd.DataFrame:
            Applies all specified correlations (both numerical and categorical) to the given DataFrame.
            Numerical-numerical correlations are applied first (linear and nonlinear), followed by
            categorical-categorical and categorical-numerical correlations in sequence.

    Raises:
        ValueError: If the correlation magnitude is out of bounds based on the type of correlation
                    or if invalid data is provided for applying correlations.
    """

    correlations: Dict[
        Tuple[str, str], Tuple[float, str]
    ]  # (magnitude, correlation function)

    # Validate correlation magnitude
    def validate_magnitude(self, magnitude: float, correlation_type: str):
        if correlation_type == "numerical-numerical":
            if not -1 <= magnitude <= 1:
                raise ValueError(
                    f"Magnitude {magnitude} out of range [-1, 1] for numerical-numerical correlation"
                )
        else:
            if not 0 <= magnitude <= 1:
                raise ValueError(
                    f"Magnitude {magnitude} out of range [0, 1] for categorical correlation"
                )

    def apply_all_linear_correlations(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        involved_columns = set()
        for (col_1, col_2), (magnitude, function) in self.correlations.items():
            if function == "linear":
                involved_columns.update([col_1, col_2])

        # Extract numerical columns involved
        data_subset = data[list(involved_columns)]

        # Compute covariance matrix and adjust it based on specified correlations
        cov_matrix = np.cov(data_subset.values, rowvar=False)
        column_list = list(
            involved_columns
        )  # Track the order of columns in the subset

        for (col_1, col_2), (magnitude, function) in self.correlations.items():
            if function == "linear":
                index_1 = column_list.index(
                    col_1
                )  # Get index relative to the subset
                index_2 = column_list.index(
                    col_2
                )  # Get index relative to the subset
                std_1 = np.sqrt(cov_matrix[index_1, index_1])
                std_2 = np.sqrt(cov_matrix[index_2, index_2])
                cov_matrix[index_1, index_2] = magnitude * std_1 * std_2
                cov_matrix[index_2, index_1] = cov_matrix[index_1, index_2]

        # Cholesky decomposition and transformation
        L = np.linalg.cholesky(cov_matrix)
        correlated_data = data_subset.values @ L.T
        data[list(involved_columns)] = correlated_data
        return data

    def apply_all_nonlinear_correlations(
        self, data: pd.DataFrame
    ) -> pd.DataFrame:
        involved_columns = set()
        for (col_1, col_2), (magnitude, function) in self.correlations.items():
            if function in ["log", "exp"]:
                involved_columns.update([col_1, col_2])

        if not involved_columns:
            # No columns involved in nonlinear correlations, skip this step
            return data

        # Extract numerical columns involved
        data_subset = data[list(involved_columns)]

        if data_subset.empty:
            raise ValueError("No data to apply nonlinear correlations on.")

        # Apply transformations (log or exp) if specified
        for col in involved_columns:
            if self.correlations[(col, col)][1] == "log":
                data[col] = np.log(data[col] - data[col].min() + 1)
            elif self.correlations[(col, col)][1] == "exp":
                data[col] = np.exp(data[col])

        # Copula fit and transformation
        copula = GaussianMultivariate()
        copula.fit(data_subset)

        params = copula.to_dict()
        corr_matrix = np.array(params["correlation"])

        # Adjust the correlation matrix based on specified correlations
        for (col_1, col_2), (magnitude, function) in self.correlations.items():
            if function in ["log", "exp"]:
                index_1 = list(involved_columns).index(col_1)
                index_2 = list(involved_columns).index(col_2)
                corr_matrix[index_1, index_2] = magnitude
                corr_matrix[index_2, index_1] = magnitude

        params["correlation"] = corr_matrix.tolist()
        copula = GaussianMultivariate.from_dict(params)
        sampled_data = copula.sample(len(data_subset))

        # Back-transform data if needed
        for i, col in enumerate(involved_columns):
            data[col] = sampled_data.iloc[:, i]

        return data

    # Apply Categorical-Categorical correlation (sequential)
    def apply_categorical_correlation(
        self, data: pd.DataFrame, col_1: str, col_2: str, magnitude: float
    ) -> pd.DataFrame:
        contingency_table = pd.crosstab(data[col_1], data[col_2])
        scaling_factor = magnitude
        adjusted_table = contingency_table * scaling_factor

        for cat_1 in adjusted_table.index:
            for cat_2 in adjusted_table.columns:
                mask = (data[col_1] == cat_1) & (data[col_2] == cat_2)
                n_to_adjust = int(adjusted_table.loc[cat_1, cat_2])

                if np.sum(mask) > n_to_adjust:
                    indices_to_adjust = mask[mask].index
                    num_adjust = np.sum(mask) - n_to_adjust
                    sample_indices = np.random.choice(
                        indices_to_adjust, size=num_adjust, replace=False
                    )

                    data.loc[sample_indices, col_2] = np.random.choice(
                        adjusted_table.columns, num_adjust
                    )

        return data

    # Apply Categorical-Numerical correlation (sequential)
    def apply_categorical_numerical_correlation(
        self, data: pd.DataFrame, col_1: str, col_2: str, magnitude: float
    ) -> pd.DataFrame:
        for category in data[col_1].unique():
            mask = data[col_1] == category
            category_numeric = pd.Categorical(data[col_1]).codes[mask][0]
            new_mean = np.mean(data.loc[mask, col_2]) + (
                magnitude * category_numeric
            )
            data.loc[mask, col_2] = np.random.normal(new_mean, 1, np.sum(mask))
        return data

    # Main method to apply correlations based on column types
    def apply_correlations(self, data: pd.DataFrame) -> pd.DataFrame:
        # Apply all linear numerical-numerical correlations at once
        data = self.apply_all_linear_correlations(data)

        # Apply all nonlinear numerical-numerical correlations at once (copula)
        data = self.apply_all_nonlinear_correlations(data)

        # Apply categorical-categorical and categorical-numerical sequentially
        for (col_1, col_2), (magnitude, function) in self.correlations.items():
            if isinstance(
                data[col_1].dtype, pd.CategoricalDtype
            ) and isinstance(data[col_2].dtype, pd.CategoricalDtype):
                data = self.apply_categorical_correlation(
                    data, col_1, col_2, magnitude
                )
            elif isinstance(
                data[col_1].dtype, pd.CategoricalDtype
            ) and is_numeric_dtype(data[col_2]):
                data = self.apply_categorical_numerical_correlation(
                    data, col_1, col_2, magnitude
                )

        return data
