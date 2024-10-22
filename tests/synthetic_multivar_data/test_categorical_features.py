import unittest

from categorical_features import CategoricalFeatures


class TestCategoricalFeatures(unittest.TestCase):

    def test_default_feature_generation(self):
        # Test default generation of categorical features (binary)
        categorical = CategoricalFeatures(n_features=2)
        df = categorical.add_categorical_features(n_samples=100)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(df.shape[1], 2)
        self.assertIn("categorical_feature_1", df.columns)

    def test_custom_levels_feature_generation(self):
        # Test generation with custom levels (3 levels for first feature, 2 for second)
        categorical = CategoricalFeatures(n_features=2, levels=[3, 2])
        df = categorical.add_categorical_features(n_samples=50)
        self.assertEqual(df.shape, (50, 2))
        self.assertEqual(len(df["categorical_feature_1"].unique()), 3)
        self.assertEqual(len(df["categorical_feature_2"].unique()), 2)

    def test_invalid_levels(self):
        # Test with invalid levels (negative values)
        with self.assertRaises(ValueError):
            CategoricalFeatures(n_features=2, levels=[3, -2])

    def test_distribution_weights(self):
        # Test categorical features with distribution weights
        categorical = CategoricalFeatures(
            n_features=1, levels=[3], distribution_weights={0: [0.7, 0.2, 0.1]}
        )
        df = categorical.add_categorical_features(n_samples=100)
        self.assertEqual(df.shape[0], 100)
        self.assertEqual(len(df["categorical_feature_1"].unique()), 3)


if __name__ == "__main__":
    unittest.main()
