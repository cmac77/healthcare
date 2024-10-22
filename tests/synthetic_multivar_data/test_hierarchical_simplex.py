import unittest


from hierarchical_simplex import HierarchicalSimplex


class TestHierarchicalSimplex(unittest.TestCase):

    def test_simplex_generation(self):
        # Test basic simplex generation with defined levels and distance info
        levels = [2, 3]
        distance_info = {"distance": [1, 2], "distance_type": ["edge", "edge"]}
        simplex = HierarchicalSimplex(levels, distance_info)

        # Ensure vertices hierarchy is not empty
        self.assertGreater(len(simplex.vertices_hierarchy), 0)

    def test_distance_matrix(self):
        # Test if distance matrix generation works correctly
        levels = [2, 2, 3]
        distance_info = {"distance": [1, 2, 3], "distance_type": ["edge"]}
        simplex = HierarchicalSimplex(levels, distance_info)

        # Ensure distance matrix global is correctly generated
        self.assertIn("distance_matrix", simplex.distance_matrix_global)
        self.assertGreater(
            len(simplex.distance_matrix_global["distance_matrix"]), 0
        )


if __name__ == "__main__":
    unittest.main()
