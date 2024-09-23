Initialization


```python
# Standard library imports
from pathlib import Path
import sys

# Third-party imports
from pyprojroot.here import here
import pandas as pd
import matplotlib

# matplotlib.use("Qt5Agg")  # or 'TkAgg'

# Adjust the Python path for local application/library imports
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Local application/library imports
from src.pd_pca_utils import apply_pca, visualize_pca

# Establish important project paths
dir_base = here()  # ./geisinger
dir_data = Path(dir_base, "data/diabetes_130_US_hospitals_1999_2008")
dir_results = Path(dir_base, "results/")
dir_scripts = Path(dir_base, "scripts/")
dir_scr = Path(dir_base, "scr/")
```

Load data and remove rows with missing data.


```python
# Load the dataset (assuming it's in CSV format)
df_data = pd.read_csv(Path(dir_data, "data.csv"))
df_key = pd.read_csv(Path(dir_data, "key.csv"))

# Drop missing data in both df_data and readmitted column before PCA
df_cleaned = df_data.dropna()
```

Let's explore the data wrt the readmitted variable.


```python

# Convert 'readmitted' values to numerical codes for visualization
df_cleaned.loc[:, "readmitted_numeric"] = df_cleaned["readmitted"].map(
    {"NO": 0, "<30": 1, ">30": 2}
)

# Apply PCA to the cleaned dataset
df_pca, pca_model = apply_pca(df_cleaned, n_components=3, handle_missing="drop")

# Ensure the 'readmitted_numeric' column is added to df_pca for coloring in the plot
df_pca["readmitted_numeric"] = df_cleaned["readmitted_numeric"].values

# Visualize the PCA with the numeric readmitted column
visualize_pca(df_pca, hue_column="readmitted_numeric")
```
