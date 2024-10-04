# Healthcare Data Analysis Repository

## Overview
This repository contains scripts, data, and utilities for performing machine learning and statistical analysis on healthcare datasets, specifically the **Diabetes 130-US hospitals dataset (1999-2008)**. The purpose of this project is to analyze patient readmission trends, apply feature engineering, and build predictive models using various techniques, including neural networks and decision trees.

The repository is structured to support data cleaning, exploratory data analysis, feature engineering, and model development, with a focus on classification problems in healthcare.

## Repository Structure
The repository is organized into the following directories:

### 1. `data/`
This directory contains the raw and cleaned datasets used in the analysis.
- `diabetes_130_US_hospitals_1999_2008/`:
  - **`data.csv`**: The raw dataset.
  - **`key.csv`**: A key for interpreting categorical values in the dataset.
  - **`df_cleaned.pkl`**: A cleaned version of the dataset, saved in pickle format.
  - **`data_summary.txt`**: Summary statistics or notes about the dataset.
  - **`README.md`**: Documentation for the data folder.

### 2. `docs/`
Contains documentation for the project.
- **`README.md`**: Describes documentation and guidelines for using the project.

### 3. `results/`
Stores the results of experiments and analysis performed on the dataset.
- **`diabetes_130_US_hospitals_1999_2008/`**: Contains results related to the diabetes dataset.
- **`README.md`**: Provides details about the results stored in this directory.

### 4. `scripts/`
Contains Python scripts used to perform various analyses on the diabetes dataset.
- **`diabetes_analysis_pipeline.py`**: The main analysis pipeline for data cleaning, preprocessing, and model training.
- **`synthetic_multivar_data_pipeline.py`**: Generates synthetic multivariate data for testing models.
- **`sandbox01.py`** & **`sandbox02.py`**: Experimental or sandbox scripts for quick tests.
- **`README.md`**: Documentation for the scripts directory.

### 5. `src/`
Contains utility modules used across the project.
- **`data_frame_column_cleaner.py`**: A utility for cleaning DataFrame columns.
- **`data_frame_column_editor.py`**: A Tkinter-based UI for editing DataFrame columns.
- **`synthetic_multivar_data_utils.py`**: Utilities for generating synthetic multivariate data.
- **`random_forest_classifier.py`**: Script for building and training a Random Forest classifier.
- **`__init__.py`**: Initialization file for the `src/` directory.

### 6. `tests/`
Contains test scripts or test data for verifying the functionality of the code.
- **`README.md`**: Documentation for the tests directory.

## Key Features
- **Data Cleaning**: Utilities to clean and preprocess healthcare data, including whitespace stripping, numeric conversion, and handling missing values.
- **Feature Engineering**: Tools for adding interaction features, one-hot encoding categorical variables, and transforming features.
- **Machine Learning Models**: Includes implementations for Random Forest classifiers and neural networks (with PyTorch) to predict patient readmission based on the dataset.
- **Synthetic Data Generation**: Functions to generate synthetic multivariate datasets for testing and experimentation.

## Setup Instructions
### Prerequisites
- Python 3.10 or later
- `pip` or `conda` for package management

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/geisinger.git
    cd geisinger
    ```

2. Install the required Python packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run any of the Python scripts in the `scripts/` directory:
    ```bash
    python scripts/diabetes_analysis_pipeline.py
    ```

## Usage
- **Data Cleaning**: Use the `data_frame_column_cleaner.py` and `data_frame_column_editor.py` utilities to clean and preprocess the dataset.
- **Model Training**: The main analysis pipeline (`diabetes_analysis_pipeline.py`) performs model training, including data preparation, PCA, and training a neural network using PyTorch.
- **Synthetic Data**: Use the `synthetic_multivar_data_pipeline.py` script to generate synthetic datasets for testing machine learning models.

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

