"""
PLS_Regression_model.py

This module implements Partial Least Squares (PLS) regression for synthetic data analysis. PLS is used to model linear
relationships between dependent and independent variables, while reducing the feature space. It is useful in cases where
the number of features is high relative to the number of observations.

Functions:
    - fit_pls_model(X: np.ndarray, y: np.ndarray, n_components: int): Fits a PLS model to the provided data.
    - plot_data(X: np.ndarray, y: np.ndarray): Visualizes the original and transformed data.
    - reconstruct_features(X: np.ndarray): Reconstructs original features from PLS-transformed components.

Example Usage:
    # Fit a PLS model with 2 components:
    pls_model = fit_pls_model(X, y, n_components=2)
    
    # Plot the original and PLS-transformed data:
    plot_data(X, y)
    
    # Reconstruct original features from PLS components:
    reconstructed_features = reconstruct_features(pls_model, X_transformed)

Requirements:
    pandas, numpy, scikit-learn, matplotlib
"""

# %%
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn.datasets import make_regression
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd

# Generate synthetic data with correlations between features
np.random.seed(42)
X, y = make_regression(n_samples=100, n_features=2, noise=5.0, random_state=42)

# Fit a PLS Regression model with 1 component
pls = PLSRegression(n_components=1)
X_pls = pls.fit_transform(X, y)[0]

# Plotting the original 3D data (X1, X2, y)
fig = plt.figure(figsize=(16, 8))
ax = fig.add_subplot(121, projection="3d")
ax.scatter(X[:, 0], X[:, 1], y, c=y, cmap="viridis", alpha=0.7)
ax.set_xlabel("Feature X1")
ax.set_ylabel("Feature X2")
ax.set_zlabel("Target y")
ax.set_title("Original 3D Data: Features X1, X2 and Target y")

# Plotting the PLS-transformed 2D data
ax2 = fig.add_subplot(122)
ax2.scatter(X_pls[:, 0], y, c=y, cmap="viridis", alpha=0.7)
ax2.set_xlabel("PLS Component 1")
ax2.set_ylabel("Target y")
ax2.set_title("PLS Projection: Latent Component vs Target")

plt.tight_layout()
plt.show()

# Display transformed data
pls_df = pd.DataFrame(X_pls, columns=["PLS Component 1"])
print(pls_df.head())

# Interpretability: Extracting the weights of the components (loadings)
# The weights (or loadings) show how much each original feature contributes to each PLS component
weights = pls.x_weights_
weights_df = pd.DataFrame(
    weights, index=["Feature X1", "Feature X2"], columns=["PLS Component 1"]
)
print("\nPLS Component Weights (Loadings):")
print(weights_df)

# Back-calculating the original feature contributions (interpreting the components back to the original space)
# This step helps understand the marginal effect of each original feature on the target
X_reconstructed = pls.inverse_transform(X_pls)
reconstructed_df = pd.DataFrame(
    X_reconstructed, columns=["Reconstructed X1", "Reconstructed X2"]
)
print("\nReconstructed Features from PLS Components:")
print(reconstructed_df.head())

# Compare reconstructed features to the original features
comparison_df = pd.DataFrame(
    {
        "Original X1": X[:, 0],
        "Original X2": X[:, 1],
        "Reconstructed X1": X_reconstructed[:, 0],
        "Reconstructed X2": X_reconstructed[:, 1],
    }
)
print("\nComparison of Original and Reconstructed Features:")
print(comparison_df.head())

# Calculate and display the residuals between original and reconstructed features
residuals = X - X_reconstructed
residuals_df = pd.DataFrame(residuals, columns=["Residual X1", "Residual X2"])
print("\nResiduals (Original - Reconstructed Features):")
print(residuals_df.head())
# %%
