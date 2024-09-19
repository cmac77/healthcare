import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


def apply_pca(df, n_components=2, handle_missing="drop"):
    """
    Apply PCA to a DataFrame and return the principal components.

    Parameters:
    - df: pd.DataFrame - The input data to be transformed using PCA
    - n_components: int - The number of principal components to retain
    - handle_missing: str - How to handle missing data: 'drop' to drop rows, 'fill_mean' to fill with the mean

    Returns:
    - df_pca: pd.DataFrame - A DataFrame containing the PCA components
    - pca: PCA - The PCA object (useful for further analysis)
    """

    # Handle missing data
    if handle_missing == "drop":
        df_cleaned = df.dropna()
    elif handle_missing == "fill_mean":
        df_cleaned = df.fillna(
            df.mean(numeric_only=True)
        )  # Fill missing numerical values with column means
    else:
        raise ValueError(f"Unknown option for handling missing data: {handle_missing}")

    # Select numerical columns
    numerical_cols = df_cleaned.select_dtypes(include=["int64", "float64"]).columns

    # Standardize the numerical data
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df_cleaned[numerical_cols])

    # Apply PCA
    pca = PCA(n_components=n_components)
    pca_components = pca.fit_transform(df_scaled)

    # Create a DataFrame with the PCA components
    df_pca = pd.DataFrame(
        data=pca_components, columns=[f"PC{i+1}" for i in range(n_components)]
    )

    return df_pca, pca


def visualize_pca(df_pca, hue_column=None):
    """
    Visualize the principal components of a PCA-reduced DataFrame.

    Parameters:
    - df_pca: pd.DataFrame - The DataFrame containing PCA components (PC1, PC2, PC3, etc.)
    - hue_column: str or None - The column to color the points by (optional)
    """
    if "PC3" in df_pca.columns:
        # 3D visualization if PC3 exists
        fig = plt.figure(figsize=(5, 4))
        ax = fig.add_subplot(111, projection="3d")

        # Scatter plot for 3 PCs
        scatter = ax.scatter(
            df_pca["PC1"],
            df_pca["PC2"],
            df_pca["PC3"],
            c=df_pca[hue_column] if hue_column else None,
            cmap="coolwarm",
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_zlabel("PC3")
        plt.title("3D PCA of Data")
        if hue_column:
            fig.colorbar(scatter, ax=ax, label=hue_column)
        plt.show()

    else:
        # 2D visualization (default)
        plt.figure(figsize=(10, 6))
        sns.scatterplot(
            x="PC1", y="PC2", hue=hue_column, data=df_pca, palette="coolwarm"
        )
        plt.title("PCA of Data")
        plt.show()
