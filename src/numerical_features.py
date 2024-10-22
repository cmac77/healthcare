"""
numerical_features.py

This module generates synthetic numerical features using hierarchical structures based on predefined levels, centroids,
and cluster distributions. It allows customization of feature generation by specifying how clusters and distances between
centroids are arranged.

Classes:
    - NumericalFeatures: Class responsible for generating synthetic numerical features.
    - ClustersInfo: Holds information about cluster configurations.
    - DistanceInfo: Stores distance-related details between centroids.

Functions:
    - add_numerical_features(n_samples: int): Generates numerical features for a synthetic dataset.

Example Usage:
    # Initialize numerical feature generator:
    numerical_feature_generator = NumericalFeatures(levels=[2, 3], distance_info=DistanceInfo([0.5, 1.0]))
    
    # Generate synthetic numerical features:
    df = numerical_feature_generator.add_numerical_features(n_samples=100)
    
    # Example Output:
    print(df.head())
    
    #     feature1  feature2  feature3
    # 0      1.23      0.45      2.12
    # 1      2.34      1.56      3.45
    # ...

Requirements:
    pandas, numpy
"""

# %%
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, List
from hierarchical_simplex import HierarchicalSimplex


# %%
@dataclass
class ClustersInfo:
    """
    A class that defines cluster information for synthetic data generation, including distributions
    and parameters for each cluster at different hierarchical levels.

    Attributes:
        distributions (Union[str, Dict[int, Union[str, List[str]]]]): Specifies the type of distribution
            used for each level or cluster. It can be a single string (e.g., "normal") applied across all levels,
            or a dictionary specifying different distributions for different levels or clusters.

        parameters (Union[dict, Dict[int, Union[dict, List[dict]]]]): Specifies the parameters for the distributions.
            Default parameters are used if not provided. It can be a dictionary mapping levels to parameters.

    Constants:
        VALID_PARAMETERS (Dict[str, List[str]]): Specifies the valid parameters for each type of distribution.
        DEFAULT_PARAMETERS (Dict[str, Dict]): Default parameters for each distribution type.
    """

    distributions: Union[str, Dict[int, Union[str, List[str]]]] = "normal"
    parameters: Union[dict, Dict[int, Union[dict, List[dict]]]] = field(
        default_factory=lambda: {"mean": 0, "std_dev": 1}
    )

    # Dictionary defining the required parameters for each distribution
    VALID_PARAMETERS = {
        "normal": ["mean", "std_dev"],
        "uniform": ["min_value", "max_value"],
        "lognormal": ["mean", "std_dev"],
        "exponential": ["scale"],
        "beta": ["alpha", "beta"],
    }

    # Default parameters for each distribution
    DEFAULT_PARAMETERS = {
        "normal": {"mean": 0, "std_dev": 1},
        "uniform": {"min_value": 0, "max_value": 1},
        "lognormal": {"mean": 1, "std_dev": 0.5},
        "exponential": {"scale": 1.0},
        "beta": {"alpha": 2.0, "beta": 5.0},
    }

    def get_distribution_for_level(
        self, level_idx: int, cluster_idx: Optional[int] = None
    ):
        """
        Get the distribution type and corresponding parameters for a given level or cluster.

        Args:
            level_idx (int): The hierarchical level for which the distribution is requested.
            cluster_idx (Optional[int]): The index of the cluster within the level (if applicable).

        Returns:
            tuple: A tuple containing the distribution type (str) and its parameters (dict).

        Raises:
            ValueError: If the cluster index is required but not provided, or if an invalid cluster
                        index is specified.
        """
        # Determine the distribution type for the specified level or cluster
        if isinstance(self.distributions, str):
            # Single distribution for all levels
            distribution = self.distributions
            parameters = (
                self.parameters
                if isinstance(self.parameters, dict)
                else self.DEFAULT_PARAMETERS[distribution]
            )
        elif isinstance(self.distributions, dict):
            # Level-specific distributions
            level_dist = self.distributions.get(
                level_idx, self.distributions.get(0, "normal")
            )

            # Expand single distribution to all clusters if necessary
            if isinstance(level_dist, list) and len(level_dist) == 1:
                level_dist = level_dist * self._get_cluster_count(level_idx)

            # Handling when level_dist is a list of distributions
            if isinstance(level_dist, list) and cluster_idx is not None:
                cluster_idx = (
                    int(cluster_idx.split("-")[-1])
                    if isinstance(cluster_idx, str)
                    else cluster_idx
                )
                if cluster_idx >= len(level_dist):
                    # Adjust cluster_idx to be within valid range by repeating parameter if necessary
                    cluster_idx %= len(level_dist)

                distribution = level_dist[cluster_idx]
                level_params = self.parameters.get(
                    level_idx, [{}] * len(level_dist)
                )

                if isinstance(level_params, list):
                    # Expand single parameter set to all clusters if necessary
                    if len(level_params) == 1:
                        level_params = level_params * len(level_dist)

                    if cluster_idx >= len(level_params):
                        cluster_idx %= len(level_params)
                    parameters = level_params[cluster_idx]
                else:
                    parameters = level_params

            elif isinstance(level_dist, list):
                # If a list is provided but cluster_idx is None, raise an error
                raise ValueError(
                    f"Cluster index must be provided for level {level_idx} as it has multiple clusters."
                )

            else:
                # If a single distribution is provided for the level
                distribution = level_dist
                level_params = self.parameters.get(
                    level_idx, self.DEFAULT_PARAMETERS.get(level_dist, {})
                )
                parameters = level_params

        # Ensure parameters are appropriate for the selected distribution
        parameters = self.validate_parameters(distribution, parameters)

        return distribution, parameters

    def _get_cluster_count(self, level_idx: int) -> int:
        """
        Helper method to determine the number of clusters at a given level.

        Args:
            level_idx (int): The hierarchical level for which the cluster count is needed.

        Returns:
            int: The number of clusters for the specified level.
        """
        if isinstance(self.parameters, dict) and level_idx in self.parameters:
            level_params = self.parameters[level_idx]
            if isinstance(level_params, list):
                return len(level_params)
            else:
                return 1
        return 1

    def validate_parameters(self, distribution: str, parameters: dict) -> dict:
        """
        Validate and adjust the parameters for the specified distribution.

        Args:
            distribution (str): The distribution type (e.g., "normal", "uniform").
            parameters (dict): The parameters for the given distribution.

        Returns:
            dict: A dictionary with validated parameters, using defaults where necessary.

        Warns:
            Prints a warning if any irrelevant parameters are provided and subsequently ignored.
        """
        valid_keys = self.VALID_PARAMETERS.get(distribution, [])
        default_params = self.DEFAULT_PARAMETERS.get(distribution, {})

        # Use default parameters for any missing key and ignore irrelevant keys
        validated_params = {}
        for key in valid_keys:
            validated_params[key] = parameters.get(
                key, default_params.get(key)
            )

        # Optionally, warn if irrelevant parameters are provided
        irrelevant_keys = set(parameters.keys()) - set(valid_keys)
        if irrelevant_keys:
            print(
                f"Warning: Irrelevant parameters {irrelevant_keys} ignored for distribution '{distribution}'."
            )

        return validated_params


# %%
@dataclass
class DistanceInfo:
    """
    A class to encapsulate and validate distance information for centroids.

    Attributes:
        distance (List[float]): List of distances for each level.
        distance_type (List[str]): List of distance types for each level (e.g., 'edge', 'origin').

    Methods:
        validate(): Ensures that the length of distance and distance_type match the number of levels.
    """

    distance: List[float]
    distance_type: List[str]

    def validate(self, num_levels: int):
        """
        Validate that the number of distance and distance_type entries match the number of levels.

        Args:
            num_levels (int): The number of hierarchical levels to validate against.

        Raises:
            ValueError: If the lengths of the distance or distance_type lists do not match num_levels.
        """
        # Validate the distance list
        if len(self.distance) == 1:
            self.distance = self.distance * num_levels
        elif len(self.distance) != num_levels:
            raise ValueError(
                f"Expected {num_levels} distance values, but got {len(self.distance)}."
            )

        # Validate the distance_type list
        if len(self.distance_type) == 1:
            self.distance_type = self.distance_type * num_levels
        elif len(self.distance_type) != num_levels:
            raise ValueError(
                f"Expected {num_levels} distance_type values, but got {len(self.distance_type)}."
            )


@dataclass
class NumericalFeatures:
    """
    A class representing numerical information for hierarchical synthetic data generation.

    Attributes:
        levels (List[int]): List of hierarchical levels, starting with the number of targets.
        distance_info (DistanceInfo): Information about distances between centroids.
        clusters_info (ClustersInfo): An instance of ClustersInfo containing distribution data.
        n_features (Optional[int]): Number of numerical features (dimensions). If not provided,
            it defaults to the highest level plus one.
        active_dimensions (Optional[Union[int, List[int]]]): Dimensions that are allowed to vary
            among the generated features.
        centroids (Optional[Dict[str, np.ndarray]]): Stores the centroids generated for each level.
        distance_matrix_global (Optional[dict]): Stores the global distance matrix from HierarchicalSimplex.
        distance_matrix_levels (Optional[dict]): Stores the distance matrices specific to each level.
    """

    levels: List[int]  # Hierarchical levels, starting with number of targets
    distance_info: DistanceInfo  # Now using the DistanceInfo class
    clusters_info: ClustersInfo  # Information about the distribution for generating points around centroids
    n_features: Optional[int] = None
    active_dimensions: Optional[Union[int, List[int]]] = None
    centroids: Optional[Dict[str, np.ndarray]] = None
    distance_matrix_global: Optional[dict] = None
    distance_matrix_levels: Optional[dict] = None

    def __post_init__(self):
        """
        Post-initialization to ensure parameters are set correctly and centroids are generated.
        """
        # Validate and set the distances using the DistanceInfo class
        self.distance_info.validate(len(self.levels))

        # Ensure n_features is set appropriately
        if self.n_features is None:
            self.n_features = max(self.levels) + 1

        # Set active dimensions
        if self.active_dimensions is None:
            self.active_dimensions = list(range(self.n_features))
        elif isinstance(self.active_dimensions, int):
            if self.active_dimensions > self.n_features:
                raise ValueError(
                    "active_dimensions cannot be greater than n_features"
                )
            self.active_dimensions = list(range(self.active_dimensions))

        if isinstance(self.active_dimensions, list):
            if max(self.active_dimensions) >= self.n_features:
                raise ValueError(
                    "Indices in active_dimensions cannot be greater than or equal to n_features"
                )

        # Generate initial centroids with HierarchicalSimplex
        hierarchical_simplex = HierarchicalSimplex(
            self.levels, self.distance_info.__dict__
        )

        # Check if the vertices_hierarchy generated is compatible with n_features
        if not hasattr(hierarchical_simplex, "vertices_hierarchy"):
            raise ValueError(
                "HierarchicalSimplex did not generate vertices_hierarchy."
            )

        self.centroids = hierarchical_simplex.vertices_hierarchy

        # Expand centroids to n_features dimensions, with additional safety checks
        self._expand_to_n_features()

        # Store the global distance matrix from HierarchicalSimplex
        self.distance_matrix_global = (
            hierarchical_simplex.distance_matrix_global
        )

        # Store the level-specific distance matrices from HierarchicalSimplex
        self.distance_matrix_levels = (
            hierarchical_simplex.distance_matrix_levels
        )

    def add_numerical_features(
        self, n_samples: int, assignments: str = "equal"
    ) -> pd.DataFrame:
        """
        Generate synthetic numerical features based on cluster and centroid information stored in the instance.

        Args:
            n_samples (int): Number of data points to generate.
            assignments (str): Strategy for assigning samples to clusters.
                            - "equal": Assign samples equally among all clusters.
                            - "proportional": Assign samples proportionally according to defined weights.
                            - "per_cluster": User-defined specific number of samples for each cluster (not implemented).

        Returns:
            pd.DataFrame: A DataFrame containing generated synthetic numerical features with appropriate cluster structure.

        Raises:
            ValueError: If an invalid assignment strategy is provided, or if distribution parameters are incompatible
                        with the dimensionality of the data.
            NotImplementedError: If "per_cluster" assignment strategy is requested (currently not implemented).
        """
        data_points = []

        n_clusters = len(self.centroids)  # Use self to access centroids

        if assignments == "equal":
            base_samples_count = n_samples // n_clusters
            remainder = n_samples % n_clusters
            samples_per_cluster = [base_samples_count] * n_clusters

            for i in range(remainder):
                samples_per_cluster[i % n_clusters] += 1

        elif assignments == "proportional":
            raise NotImplementedError(
                "The 'proportional' assignment strategy is not yet implemented."
            )

        elif assignments == "per_cluster":
            raise NotImplementedError(
                "The 'per_cluster' assignment strategy is not yet implemented."
            )

        else:
            raise ValueError(f"Invalid assignment strategy: {assignments}")

        # Generate data points around each centroid
        for idx, (centroid_key, centroid) in enumerate(
            self.centroids.items()
        ):  # Use self for centroids
            level_idx = len(centroid_key.split("-")) - 1
            cluster_idx = int(centroid_key.split("-")[-1])

            # Get distribution and parameters for the cluster
            distribution, parameters = (
                self.clusters_info.get_distribution_for_level(
                    level_idx, cluster_idx
                )
            )

            # Get number of samples for this cluster
            samples_count = samples_per_cluster[idx]

            # Generate samples based on the distribution type
            if distribution == "normal":
                if isinstance(parameters["mean"], (int, float)):
                    mean_vector = [
                        parameters["mean"]
                    ] * self.n_features  # Use self for n_features
                elif (
                    isinstance(parameters["mean"], list)
                    and len(parameters["mean"]) == self.n_features
                ):
                    mean_vector = parameters["mean"]
                else:
                    raise ValueError(
                        f"Invalid mean: {parameters['mean']} for cluster {centroid_key}"
                    )

                samples = np.random.multivariate_normal(
                    mean_vector,
                    np.eye(self.n_features)
                    * parameters["std_dev"] ** 2,  # Use self for n_features
                    size=samples_count,
                )

            elif distribution == "uniform":
                if isinstance(
                    parameters["min_value"], (int, float)
                ) and isinstance(parameters["max_value"], (int, float)):
                    min_vector = [parameters["min_value"]] * self.n_features
                    max_vector = [parameters["max_value"]] * self.n_features
                elif (
                    isinstance(parameters["min_value"], list)
                    and len(parameters["min_value"]) == self.n_features
                    and isinstance(parameters["max_value"], list)
                    and len(parameters["max_value"]) == self.n_features
                ):
                    min_vector = parameters["min_value"]
                    max_vector = parameters["max_value"]
                else:
                    raise ValueError(
                        f"Invalid min/max values for cluster {centroid_key}"
                    )

                samples = np.random.uniform(
                    min_vector,
                    max_vector,
                    size=(samples_count, self.n_features),
                )

            else:
                raise NotImplementedError(
                    f"Distribution type '{distribution}' is not implemented."
                )

            # Offset samples by the centroid to ensure clusters are centered around the calculated centroids
            adjusted_samples = samples + centroid

            # Use the active dimensions to decide which features should have generated variability
            final_samples = adjusted_samples.copy()
            for i in range(len(centroid)):
                if (
                    i not in self.active_dimensions
                ):  # Use self for active_dimensions
                    final_samples[:, i] = centroid[i]

            data_points.extend(final_samples)

        # Convert data points into a pandas DataFrame
        df = pd.DataFrame(
            data_points,
            columns=[f"feature_{i}" for i in range(self.n_features)],
        )  # Use self for n_features
        return df

    def _expand_to_n_features(self):
        """
        Expand each centroid in the vertices_hierarchy to match the specified number of features (n_features).

        - If a centroid has fewer dimensions, it is padded with zeros.
        - If a centroid exceeds n_features, an exception is raised.

        Raises:
            ValueError: If the dimensionality of any centroid exceeds the specified n_features.
        """
        expanded_centroids = {}
        for key, centroid in self.centroids.items():
            # If centroid has fewer dimensions than n_features, expand it
            if len(centroid) < self.n_features:
                expanded_centroid = np.zeros(self.n_features)
                expanded_centroid[: len(centroid)] = (
                    centroid  # Keep original values
                )
            elif len(centroid) > self.n_features:
                raise ValueError(
                    f"Centroid dimensionality {len(centroid)} exceeds n_features {self.n_features}. Please adjust n_features accordingly."
                )
            else:
                expanded_centroid = centroid

            expanded_centroids[key] = expanded_centroid

        # Update centroids with the expanded versions
        self.centroids = expanded_centroids
