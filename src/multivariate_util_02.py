import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
import matplotlib

matplotlib.use("tkagg")  # Set the backend to TkAgg for interactive plotting


# %% Create Class Centers with Centroids
def create_class_centers_with_centroid(
    n_classes,
    n_clusters_per_class,
    n_informative,
    between_class_sep,
    within_class_sep,
    use_gpu=False,
    random_state=None,
):
    """
    Generate class centroids and cluster centers for each class.

    Parameters:
    ----------
    n_classes : int
        Number of distinct classes to generate.
    n_clusters_per_class : list or int
        Number of clusters to generate for each class. Can be a list specifying clusters per class or an int applied to all classes.
    n_informative : int
        Number of informative features (dimensions) in the dataset.
    between_class_sep : float
        Minimum separation between the class centroids (Euclidean distance).
    within_class_sep : float or list
        Spread of cluster centers around the class centroids. Can be a float applied to all classes or a list for each class.
    use_gpu : bool, default=False
        Whether to use GPU (CUDA) for computation.
    random_state : int, optional
        Random seed for reproducibility.

    Returns:
    -------
    centers : torch.Tensor
        Tensor containing the cluster centers.
    cluster_labels : list
        List of class labels for each cluster center.
    class_centroids : torch.Tensor
        Tensor containing the centroids for each class.
    """
    device = torch.device(
        "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    )
    rng = (
        torch.Generator(device=device).manual_seed(random_state)
        if random_state
        else None
    )

    centers, cluster_labels, class_centroids = [], [], []

    # Ensure n_clusters_per_class and within_class_sep are lists
    if isinstance(n_clusters_per_class, int):
        n_clusters_per_class = [n_clusters_per_class] * n_classes
    if isinstance(within_class_sep, (int, float)):
        within_class_sep = [within_class_sep] * n_classes

    def is_far_enough(new_centroid, existing_centroids, min_distance):
        """Check if the new centroid is far enough from all existing centroids."""
        for centroid in existing_centroids:
            if torch.norm(new_centroid - centroid) < min_distance:
                return False
        return True

    # Step 1: Generate centroids for each class
    for class_idx in range(n_classes):
        while True:
            new_centroid = torch.normal(
                0, 1, (n_informative,), generator=rng, device=device
            )
            if len(class_centroids) == 0 or is_far_enough(
                new_centroid, class_centroids, between_class_sep
            ):
                class_centroids.append(new_centroid)
                break

    class_centroids = torch.stack(class_centroids)

    # Step 2: Generate clusters around the centroids
    for class_idx, centroid in enumerate(class_centroids):
        num_clusters = n_clusters_per_class[class_idx]
        spread = within_class_sep[class_idx]
        for cluster_idx in range(num_clusters):
            std_tensor = torch.full_like(centroid, spread)
            cluster_center = torch.normal(centroid, std_tensor, generator=rng)
            centers.append(cluster_center)
            cluster_labels.append(class_idx)

    centers = torch.stack(centers)

    return centers, cluster_labels, class_centroids


# %% Plot Cluster Centers and Centroids
def plot_centers_and_centroids(centers, cluster_labels, class_centroids):
    """
    Plot cluster centers and class centroids in 2D, 3D, or PCA-reduced space.

    Parameters:
    ----------
    centers : torch.Tensor
        Tensor containing the cluster centers.
    cluster_labels : list
        Class labels for each cluster center.
    class_centroids : torch.Tensor
        Tensor containing the centroids for each class.
    """
    centers_np = centers.cpu().numpy()
    centroids_np = class_centroids.cpu().numpy()
    n_features = centers_np.shape[1]

    if n_features == 2:
        plt.figure(figsize=(8, 6))
        for i, center in enumerate(centers_np):
            plt.scatter(
                center[0],
                center[1],
                label=f"Class {cluster_labels[i]}",
                alpha=0.6,
            )
        for i, centroid in enumerate(centroids_np):
            plt.scatter(
                centroid[0],
                centroid[1],
                color="black",
                marker="X",
                s=200,
                label=f"Class {i} Centroid",
                edgecolor="black",
            )
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("Cluster Centers and Class Centroids in 2D")
        plt.grid(True)
        plt.legend()
        plt.show()

    elif n_features == 3:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for i, center in enumerate(centers_np):
            ax.scatter(
                center[0],
                center[1],
                center[2],
                label=f"Class {cluster_labels[i]}",
                alpha=0.6,
            )
        for i, centroid in enumerate(centroids_np):
            ax.scatter(
                centroid[0],
                centroid[1],
                centroid[2],
                color="black",
                marker="X",
                s=200,
                label=f"Class {i} Centroid",
                edgecolor="black",
            )
        ax.set_xlabel("Feature 1")
        ax.set_ylabel("Feature 2")
        ax.set_zlabel("Feature 3")
        plt.title("Cluster Centers and Class Centroids in 3D")
        plt.legend()
        plt.show()

    else:
        pca = PCA(n_components=3)
        centers_pca = pca.fit_transform(centers_np)
        centroids_pca = pca.transform(centroids_np)

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        for i, center in enumerate(centers_pca):
            ax.scatter(
                center[0],
                center[1],
                center[2],
                label=f"Class {cluster_labels[i]}",
                alpha=0.6,
            )
        for i, centroid in enumerate(centroids_pca):
            ax.scatter(
                centroid[0],
                centroid[1],
                centroid[2],
                color="black",
                marker="X",
                s=200,
                label=f"Class {i} Centroid",
                edgecolor="black",
            )
        ax.set_xlabel("PC 1")
        ax.set_ylabel("PC 2")
        ax.set_zlabel("PC 3")
        plt.title("Cluster Centers and Class Centroids in PCA 3D Space")
        plt.legend()
        plt.show()


# %% Add Interaction Terms
def add_interaction_terms(
    df,
    interaction_pairs,
    interaction_types,
    interaction_informativeness,
    interaction_only=False,
    class_label=None,
    correlation_magnitude=1.0,
    non_linear_correlation=None,
    class_specific_correlation=None,
):
    """
    Add interaction terms to a DataFrame with optional correlation with the class label.

    Parameters:
    ----------
    df : DataFrame
        Input data containing the features.
    interaction_pairs : list of tuples
        List of tuples specifying which features to interact.
    interaction_types : dict
        A dictionary specifying the type of interaction for each pair.
    interaction_informativeness : dict
        A dictionary specifying whether each interaction is informative or uninformative.
    interaction_only : bool, default=False
        If True, only the interaction terms are kept. If False, both the original features and interaction terms are kept.
    class_label : Series or ndarray, optional
        The target variable or class label to be used for informative interactions.
    correlation_magnitude : float or dict, default=1.0
        If a float, it applies to all informative interaction terms. If a dict, it specifies the correlation magnitude for each pair.
    non_linear_correlation : str, optional
        Type of non-linear transformation for the class label ('exp', 'log', 'poly').
    class_specific_correlation : dict, optional
        Specify different correlation strategies for each class.

    Returns:
    -------
    df : DataFrame
        DataFrame with the interaction terms added (or replacing the original features if interaction_only=True).
    """
    interaction_df = pd.DataFrame()

    for pair in interaction_pairs:
        feature1, feature2 = pair
        interaction_type = interaction_types.get(pair, "multiplicative")
        informative = interaction_informativeness.get(pair, False)
        interaction_name = (
            f"Interaction_{feature1}_{feature2}_{interaction_type}"
        )

        if interaction_type == "multiplicative":
            interaction_term = df.iloc[:, feature1] * df.iloc[:, feature2]
        elif interaction_type == "additive":
            interaction_term = df.iloc[:, feature1] + df.iloc[:, feature2]
        elif interaction_type == "non-linear":
            interaction_term = np.power(df.iloc[:, feature1], 2) + np.log(
                np.abs(df.iloc[:, feature2]) + 1
            )
        else:
            raise ValueError(f"Unknown interaction type: {interaction_type}")

        # Determine correlation magnitude
        magnitude = (
            correlation_magnitude.get(pair, 1.0)
            if isinstance(correlation_magnitude, dict)
            else correlation_magnitude
        )

        # Apply correlation with class label if informative
        if informative and class_label is not None:
            if (
                class_specific_correlation
                and class_label in class_specific_correlation
            ):
                correlation_type = class_specific_correlation[class_label]
                if correlation_type == "exp":
                    interaction_term += magnitude * np.exp(class_label)
                elif correlation_type == "log":
                    interaction_term += magnitude * np.log(
                        np.abs(class_label) + 1
                    )
                elif correlation_type == "poly":
                    interaction_term += magnitude * np.power(class_label, 2)
                else:
                    interaction_term += magnitude * np.random.normal(
                        loc=class_label, scale=0.1
                    )
            else:
                if non_linear_correlation == "exp":
                    interaction_term += magnitude * np.exp(class_label)
                elif non_linear_correlation == "log":
                    interaction_term += magnitude * np.log(
                        np.abs(class_label) + 1
                    )
                elif non_linear_correlation == "poly":
                    interaction_term += magnitude * np.power(class_label, 2)
                else:
                    interaction_term += magnitude * np.random.normal(
                        loc=class_label, scale=0.1
                    )

        interaction_df[interaction_name] = interaction_term

    return (
        interaction_df
        if interaction_only
        else pd.concat([df, interaction_df], axis=1)
    )


# %% Example Usage: Creating synthetic data and adding interaction terms

# Parameters for generating synthetic data
n_classes = 3
n_clusters_per_class = [
    1,
    1,
    3,
]  # Class 0 has 1 cluster, Class 1 has 1 cluster, Class 2 has 3 clusters
n_informative = 5  # Number of informative features
between_class_sep = 3.0  # Minimum separation between class centroids
within_class_sep = [0.2, 0.5, 1.0]  # Different spread for each class
use_gpu = True  # Use GPU if available
random_state = 42  # For reproducibility

# Step 1: Generate class centers and centroids using create_class_centers_with_centroid
centers, cluster_labels, class_centroids = create_class_centers_with_centroid(
    n_classes=n_classes,
    n_clusters_per_class=n_clusters_per_class,
    n_informative=n_informative,
    between_class_sep=between_class_sep,
    within_class_sep=within_class_sep,
    use_gpu=use_gpu,
    random_state=random_state,
)

# Step 2: Convert the generated centers and labels into a Pandas DataFrame
# Assuming the generated centers represent the features in the synthetic dataset
df_synthetic = pd.DataFrame(
    centers.cpu().numpy(),
    columns=[f"Feature_{i}" for i in range(n_informative)],
)
df_synthetic["ClassLabel"] = cluster_labels  # Add the class labels

print("First 5 rows of synthetic data:")
print(df_synthetic.head())

# Step 3: Define interaction pairs and types
interaction_pairs = [
    (0, 1),
    (2, 3),
]  # Interactions between Feature_0 & Feature_1, and Feature_2 & Feature_3
interaction_types = {
    (0, 1): "multiplicative",
    (2, 3): "non-linear",
}  # Interaction types
interaction_informativeness = {
    (0, 1): True,
    (2, 3): True,
}  # Mark both as informative

# Define correlation magnitudes for specific interactions
correlation_magnitude = {
    (
        0,
        1,
    ): 2.0,  # Interaction between Feature_0 and Feature_1 has stronger correlation
    (
        2,
        3,
    ): 0.5,  # Interaction between Feature_2 and Feature_3 has weaker correlation
}

# Step 4: Add interaction terms to the synthetic dataset
df_with_interactions = add_interaction_terms(
    df=df_synthetic.drop(
        columns="ClassLabel"
    ),  # Drop the class label before adding interactions
    interaction_pairs=interaction_pairs,
    interaction_types=interaction_types,
    interaction_informativeness=interaction_informativeness,
    interaction_only=False,  # Keep original features along with interaction terms
    class_label=df_synthetic[
        "ClassLabel"
    ],  # Use the class labels to correlate interaction terms
    correlation_magnitude=correlation_magnitude,  # Specify different correlation magnitudes for interactions
    non_linear_correlation=None,  # No non-linear transformation applied globally
    class_specific_correlation=None,  # No class-specific correlation logic applied
)

# Step 5: Display the first few rows of the DataFrame with interaction terms
print("\nFirst 5 rows of synthetic data with interaction terms:")
print(df_with_interactions.head())

# Step 6: Plot the generated cluster centers and class centroids
plot_centers_and_centroids(centers, cluster_labels, class_centroids)

# %%
