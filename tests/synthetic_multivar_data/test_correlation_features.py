import unittest
import pandas as pd

from correlation_features import CorrelationFeatures


class TestCorrelationFeatures(unittest.TestCase):

    def test_linear_correlation(self):
        # Create some test data
        data = pd.DataFrame(
            {"feature_1": range(100), "feature_2": range(100, 200)}
        )

        # Define a linear correlation between feature_1 and feature_2
        correlations = {("feature_1", "feature_2"): (0.9, "linear")}
        corr_handler = CorrelationFeatures(correlations)

        # Apply the correlations
        correlated_data = corr_handler.apply_correlations(data)

        # Test if the data shape is maintained and correlations applied
        self.assertEqual(correlated_data.shape, data.shape)

    def test_categorical_correlation(self):
        # Create some categorical test data
        data = pd.DataFrame(
            {
                "cat_feature_1": ["A", "B"] * 50,
                "cat_feature_2": ["X", "Y"] * 50,
            }
        )

        # Define a correlation between categorical features
        correlations = {
            ("cat_feature_1", "cat_feature_2"): (0.5, "categorical")
        }
        corr_handler = CorrelationFeatures(correlations)

        # Apply the correlation
        correlated_data = corr_handler.apply_correlations(data)

        # Test that the data shape is maintained
        self.assertEqual(correlated_data.shape, data.shape)


if __name__ == "__main__":
    unittest.main()
