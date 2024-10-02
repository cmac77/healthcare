# %%
import shap
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Simulate a large dataset with interaction between features
X, y = make_classification(
    n_samples=100000,
    n_features=1000,
    n_informative=50,
    n_redundant=50,
    random_state=42,
)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Convert to DMatrix for XGBoost (GPU-optimized structure)
feature_names = [
    f"Feature_{i}" for i in range(X_train.shape[1])
]  # Adjust to your dataset
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names)


# Parameters for XGBoost with GPU acceleration
params = {
    "objective": "binary:logistic",
    "max_depth": 10,  # Increased depth for capturing interactions
    "eta": 0.05,  # Learning rate
    "tree_method": "hist",  # GPU-accelerated histogram-based method
    "device": "cuda",  # Use GPU
    "subsample": 0.8,  # Row subsampling (80% of rows)
    "colsample_bytree": 0.8,  # Feature subsampling (80% of features)
    "eval_metric": "auc",
}

# Train the model on a large dataset using GPU
bst = xgb.train(params, dtrain, num_boost_round=100, evals=[(dtest, "test")])

# Evaluate predictions
y_pred = bst.predict(dtest)
accuracy = sum((y_pred > 0.5) == y_test) / len(y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

import shap
import pandas as pd

# Explain the model's predictions using SHAP values
explainer = shap.TreeExplainer(bst)

# Convert test data to DataFrame with feature names for SHAP
X_test_df = pd.DataFrame(X_test, columns=feature_names)

# Generate SHAP values
shap_values = explainer.shap_values(dtest)

# Visualize feature importance using SHAP
shap.summary_plot(shap_values, X_test_df, plot_type="bar")

# Plot feature interactions
shap.dependence_plot(
    "Feature_1", shap_values, X_test_df, interaction_index="Feature_2"
)
# Generate SHAP interaction values
shap_interaction_values = explainer.shap_interaction_values(dtest)
# %%
