import unittest


from numerical_features import (
    NumericalFeatures,
    DistanceInfo,
    ClustersInfo,
)


class TestNumericalFeatures(unittest.TestCase):

    def test_numerical_feature_generation(self):
        # Define levels, distance info, and cluster info
        levels = [2, 3]
        distance_info = DistanceInfo(
            distance=[1, 2], distance_type=["edge", "edge"]
        )
        clusters_info = ClustersInfo()

        # Initialize NumericalFeatures with the given inputs
        numerical = NumericalFeatures(levels, distance_info, clusters_info)

        # Generate synthetic features
        df = numerical.add_numerical_features(n_samples=50)

        # Test that 50 samples were generated
        self.assertEqual(df.shape[0], 50)
        self.assertGreater(df.shape[1], 0)

    def test_invalid_active_dimensions(self):
        # Test with invalid active dimensions
        levels = [2, 2]
        distance_info = DistanceInfo(
            distance=[1, 2], distance_type=["edge", "edge"]
        )
        clusters_info = ClustersInfo()

        # active_dimensions > n_features should raise an error
        with self.assertRaises(ValueError):
            NumericalFeatures(
                levels, distance_info, clusters_info, active_dimensions=10
            )


if __name__ == "__main__":
    unittest.main()
