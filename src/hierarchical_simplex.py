"""
hierarchical_simplex.py

This module defines the logic for generating centroids in hierarchical synthetic data generation processes using the
mathematical concept of a simplex. It helps to place centroids in N-dimensional space in a structured and deterministic
way across multiple hierarchical levels.

Classes:
    - HierarchicalSimplex: Class to handle the generation of centroids for different levels of a hierarchy.

Functions:
    - generate_centroids(levels: List[int], dimensions: int): Generates centroids based on a hierarchical simplex structure.

Example Usage:
    # Initialize the HierarchicalSimplex object:
    simplex_generator = HierarchicalSimplex(levels=[2, 3, 4], distance_info={'distance': [1.0, 0.5]})
    
    # Generate centroids:
    centroids = simplex_generator.generate_centroids(dimensions=3)
    
    # Example Centroids Output:
    print(centroids)
    # Output: array of centroids in 3D space

Requirements:
    numpy
"""

# %%
import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt


class HierarchicalSimplex:
    def __init__(self, levels, distance_info, center=None):
        """
        Initialize the HierarchicalSimplex class.

        Parameters:
        - levels (list): A list defining the number of vertices at each level.
        - distance_info (dict): A dictionary containing 'distance' and 'distance_type' for each level.
        - center (list): Center coordinates for the root level. If not provided, the origin will be used.
        """
        self.levels = levels
        self.center = (
            center
            if center is not None
            else (np.zeros(levels[0] - 1) if levels[0] > 1 else [])
        )

        # Ensure distance_info has the appropriate number of values for each level.
        self.distance_info = self._prepare_distance_info(distance_info)

        # Initialize attributes that will be populated later
        self.vertices_hierarchy = {}  # Initialize as an empty dictionary
        self.parent_child_relationships = (
            {}
        )  # Initialize as an empty dictionary
        self.centroids_level = {}  # Initialize as an empty dictionary
        self.level_metadata = {}  # Initialize as an empty dictionary
        self.distance_matrix_levels = {}  # Initialize as an empty dictionary
        self.distance_matrix_global = {}  # Initialize as an empty dictionary

        # Populate all other attributes on initialization
        self.vertices_hierarchy = self.build_hierarchy()
        self.distance_matrix_levels, self.distance_matrix_global = (
            self.compute_distances()
        )
        self.centroids_level = self.compute_level_centroids()
        self.level_metadata = self.compute_level_metadata()
        self.parent_child_relationships = (
            self.compute_parent_child_relationships()
        )

    def _prepare_distance_info(self, distance_info):
        """
        Prepare the distance_info dictionary, ensuring each level has a corresponding distance and distance_type.

        Parameters:
        - distance_info (dict): A dictionary containing 'distance' and 'distance_type' for each level.

        Returns:
        - dict: A dictionary with 'distance' and 'distance_type' lists with appropriate length.
        """
        num_levels = len(self.levels)

        # Handle 'distance' values
        if len(distance_info["distance"]) == 1:
            distance_info["distance"] = distance_info["distance"] * num_levels
        elif len(distance_info["distance"]) < num_levels:
            raise ValueError(
                "Not enough 'distance' values provided for all levels."
            )

        # Handle 'distance_type' values
        if len(distance_info["distance_type"]) == 1:
            distance_info["distance_type"] = (
                distance_info["distance_type"] * num_levels
            )
        elif len(distance_info["distance_type"]) < num_levels:
            raise ValueError(
                "Not enough 'distance_type' values provided for all levels."
            )

        return distance_info

    def generate_simplex_coordinates(
        self, n_dims, distance=1, center=None, distance_type="edge"
    ):
        """
        Generate the coordinates of the vertices of a simplex in M dimensions.

        Parameters:
        - n_dims (int): The number of spatial dimensions in which to embed the simplex.
        - distance (float, optional): The scaling factor based on the desired distance_type. Default is 1 (no scaling).
        - center (array_like, optional): The coordinates to center the simplex on.
        - distance_type (str, optional): Specifies whether to distance based on 'edge' length or 'origin' distance.

        Returns:
        - numpy.ndarray: An ((n_dims+1) x n_dims) array containing the coordinates of the vertices of the simplex,
          where each row corresponds to a vertex and each column corresponds to a dimension.
        """
        if center is not None:
            center = np.array(center)
            if len(center) != n_dims and n_dims > 0:
                raise ValueError(
                    f"Center dimensionality ({len(center)}) must match n_dims ({n_dims})."
                )
        else:
            center = np.zeros(n_dims)

        if n_dims == 0:
            return np.array([center])

        n_vertices = n_dims + 1
        vertices = np.zeros([n_vertices, n_dims])

        for i in range(n_dims):
            sum_squares = np.sum(vertices[i, :i] ** 2)
            vertices[i, i] = np.sqrt(1.0 - sum_squares)

            for j in range(i + 1, n_vertices):
                dot_product = np.dot(vertices[i, :i], vertices[j, :i])
                vertices[j, i] = (-1.0 / n_dims - dot_product) / vertices[i, i]

        if distance_type == "edge":
            current_length = np.sqrt(2 * (1 + 1 / n_dims))
            scale_factor = distance / current_length
        elif distance_type == "origin":
            scale_factor = distance

        vertices *= scale_factor
        vertices += np.array(center)

        return vertices

    def build_hierarchy(self):
        """
        Recursively generate hierarchical simplex vertices for multiple levels and store in a dictionary.
        """
        max_dims = max(self.levels) - 1
        self._build_level_vertices(0, "0", max_dims, self.center)
        return self.vertices_hierarchy

    def _build_level_vertices(
        self, level, parent_key, max_dims, parent_vertex=None
    ):
        """
        Helper method to generate simplex vertices recursively for a given level.

        Parameters:
        - level (int): Current level in the hierarchy.
        - parent_key (str): The key of the parent vertex.
        - max_dims (int): Maximum dimensionality for embedding the simplex.
        - parent_vertex (numpy.ndarray, optional): Coordinates of the parent vertex.
        """
        if level >= len(self.levels):
            return

        n_vertices = self.levels[level]
        n_dims = n_vertices - 1

        full_center = self._compute_inherited_dims(parent_vertex, max_dims)

        distance = self.distance_info["distance"][level]
        distance_type = self.distance_info["distance_type"][level]

        vertices = self.generate_simplex_coordinates(
            n_dims=n_dims,
            distance=distance,
            center=full_center[:n_dims],
            distance_type=distance_type,
        )

        embedded_vertices = self._embed_vertices(
            vertices, full_center, n_dims, max_dims
        )

        for i, vertex in enumerate(embedded_vertices):
            # Update naming convention for clarity
            child_key = f"{parent_key}-{i}" if level > 0 else f"{i}"
            self.vertices_hierarchy[child_key] = vertex
            self._update_relationships(parent_key, child_key)

            self._build_level_vertices(level + 1, child_key, max_dims, vertex)

    def _compute_inherited_dims(self, parent_vertex, max_dims):
        """
        Compute the inherited dimensions for a new level based on the parent's vertex.

        Parameters:
        - parent_vertex (numpy.ndarray, optional): The parent vertex coordinates.
        - max_dims (int): The maximum dimensionality for embedding the simplex.

        Returns:
        - numpy.ndarray: The inherited center coordinates.
        """
        if parent_vertex is not None:
            full_center = np.zeros(max_dims)
            full_center[: len(parent_vertex)] = parent_vertex
        else:
            full_center = np.zeros(max_dims)
        return full_center

    def _embed_vertices(self, vertices, full_center, n_dims, max_dims):
        embedded_vertices = np.zeros((vertices.shape[0], max_dims))
        embedded_vertices[:, :n_dims] = vertices
        embedded_vertices[:, n_dims:] = full_center[n_dims:]
        return embedded_vertices

    def _update_relationships(self, parent_key, child_key):
        """
        Update the parent-child relationships for the given vertices.

        Parameters:
        - parent_key (str): The key of the parent vertex.
        - child_key (str): The key of the child vertex.
        """
        if parent_key not in self.parent_child_relationships:
            self.parent_child_relationships[parent_key] = []
        self.parent_child_relationships[parent_key].append(child_key)

        # # Debug: Print updated relationships
        # print(
        #     f"Updated relationship: Parent: {parent_key}, Child: {child_key}"
        # )

    def compute_distances(self):
        """
        Compute the distance matrices for each group of sibling vertices at each level
        and also compute a global distance matrix for all vertices, while storing keys
        corresponding to rows and columns.

        Returns:
        - dict: A dictionary with parent keys as keys and a data structure containing the
                distance matrix and keys for each level.
        - dict: A dictionary containing the global distance matrix and keys for all vertices.
        """
        # Compute distance matrices for vertices at each level (existing functionality)
        self.distance_matrix_levels = {}
        processed_keys = (
            set()
        )  # To avoid reprocessing the same group of vertices

        for key in self.vertices_hierarchy.keys():
            # Get the parent key to group the siblings together
            parent_key = "-".join(key.split("-")[:-1])

            # Skip if already processed this group or if there's no parent (root level)
            if parent_key in processed_keys or parent_key == "":
                continue

            # Gather all vertices with the same parent key (i.e., at the same level) and sort them
            group_keys = sorted(
                [
                    k
                    for k in self.vertices_hierarchy.keys()
                    if "-".join(k.split("-")[:-1]) == parent_key
                ]
            )
            group_vertices = np.array(
                [self.vertices_hierarchy[k] for k in group_keys]
            )

            # Ensure group_vertices is a 2D array
            if group_vertices.ndim == 1:
                group_vertices = group_vertices.reshape(1, -1)

            # Add the parent key to processed keys to avoid duplicate validation
            processed_keys.add(parent_key)

            # Calculate the distance matrix for vertices in this group
            if len(group_vertices) > 1:
                dist_matrix = distance_matrix(group_vertices, group_vertices)
                self.distance_matrix_levels[parent_key] = {
                    "distance_matrix": dist_matrix,
                    "keys": group_keys,
                }

        # Compute the global distance matrix for all vertices
        all_vertices_keys = list(self.vertices_hierarchy.keys())
        all_vertices = np.array(
            [self.vertices_hierarchy[key] for key in all_vertices_keys]
        )

        # Ensure all_vertices is a 2D array
        if all_vertices.ndim == 1:
            all_vertices = all_vertices.reshape(1, -1)

        self.distance_matrix_global = {
            "distance_matrix": distance_matrix(all_vertices, all_vertices),
            "keys": all_vertices_keys,
        }

        return self.distance_matrix_levels, self.distance_matrix_global

    def verify_centers(self):
        """
        Verify that child vertices are properly centered around their parent vertices,
        and store the centroids for each group of children.

        Returns:
        - dict: A dictionary containing verification results for each level,
                with a message indicating whether centroids align with parent vertices.
        """
        centers = {}
        self.centroids_level = (
            {}
        )  # Dictionary to store the centroids of each group of children
        processed_parents = (
            set()
        )  # Track processed parents to avoid redundant checks

        print("\nVerifying Centers of Vertices in the Hierarchy:")

        for (
            parent_key,
            children_keys,
        ) in self.parent_child_relationships.items():
            # Skip the top-level keys as they have no parent to compare against
            if parent_key == "0":
                continue

            if parent_key in processed_parents:
                continue

            # Gather all child vertices
            child_vertices = np.array(
                [
                    self.vertices_hierarchy[child_key]
                    for child_key in children_keys
                ]
            )

            # Calculate the centroid of the child vertices
            centroid = np.mean(child_vertices, axis=0)
            self.centroids_level[parent_key] = (
                centroid  # Store the centroid for this group
            )
            parent_vertex = self.vertices_hierarchy.get(parent_key)

            if parent_vertex is None:
                continue  # Skip if the parent vertex doesn't exist

            # Verification comparison
            if np.allclose(centroid, parent_vertex, atol=1e-2):
                centers[parent_key] = (
                    "Centroid correctly aligns with its parent."
                )
                print(
                    f"Parent Key: {parent_key} - Centroid is correctly aligned."
                )
            else:
                centers[parent_key] = (
                    "WARNING: Centroid does not align properly with the parent vertex."
                )
                print(
                    f"Parent Key: {parent_key} - WARNING: Centroid is not correctly aligned."
                )

            # Mark the parent as processed
            processed_parents.add(parent_key)

        return centers

    def plot_vertices(self):
        """
        Plot the hierarchical structure of vertices with connections and distances.

        Returns:
        - None
        """
        # Determine the dimensionality of the vertices
        sample_vertex = next(iter(self.vertices_hierarchy.values()))
        dimensions = sample_vertex.shape[0]
        is_3d = dimensions == 3

        # Set up the figure
        fig = plt.figure(figsize=(26, 20))
        ax = fig.add_subplot(111, projection="3d" if is_3d else None)
        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        if is_3d:
            ax.set_zlabel("Z-axis")

        # Plot vertices level by level
        for key, vertices in self.vertices_hierarchy.items():
            parent_key = "-".join(key.split("-")[:-1])

            # Plot vertices
            if is_3d:
                ax.scatter(
                    vertices[0], vertices[1], vertices[2], color="blue", s=50
                )
                ax.text(
                    vertices[0],
                    vertices[1],
                    vertices[2],
                    key,
                    fontsize=8,
                    ha="right",
                    color="blue",
                )
            else:
                ax.scatter(vertices[0], vertices[1], color="blue", s=50)
                ax.text(
                    vertices[0],
                    vertices[1],
                    key,
                    fontsize=8,
                    ha="right",
                    color="blue",
                )

            # Draw lines between sibling vertices
            if parent_key != "":
                sibling_keys = [
                    k
                    for k in self.vertices_hierarchy.keys()
                    if "-".join(k.split("-")[:-1]) == parent_key
                ]
                sibling_vertices = np.array(
                    [
                        self.vertices_hierarchy[sibling_key]
                        for sibling_key in sibling_keys
                    ]
                )
                if len(sibling_vertices) > 1:
                    dist_matrix = distance_matrix(
                        sibling_vertices, sibling_vertices
                    )
                    for i in range(len(sibling_keys)):
                        for j in range(i + 1, len(sibling_keys)):
                            # Black lines between siblings
                            if is_3d:
                                ax.plot(
                                    [
                                        sibling_vertices[i, 0],
                                        sibling_vertices[j, 0],
                                    ],
                                    [
                                        sibling_vertices[i, 1],
                                        sibling_vertices[j, 1],
                                    ],
                                    [
                                        sibling_vertices[i, 2],
                                        sibling_vertices[j, 2],
                                    ],
                                    color="black",
                                )
                                # Midpoint for labeling
                                mid_x = (
                                    sibling_vertices[i, 0]
                                    + sibling_vertices[j, 0]
                                ) / 2
                                mid_y = (
                                    sibling_vertices[i, 1]
                                    + sibling_vertices[j, 1]
                                ) / 2
                                mid_z = (
                                    sibling_vertices[i, 2]
                                    + sibling_vertices[j, 2]
                                ) / 2
                                ax.text(
                                    mid_x,
                                    mid_y,
                                    mid_z,
                                    f"{dist_matrix[i, j]:.2f}",
                                    fontsize=8,
                                    color="black",
                                )
                            else:
                                ax.plot(
                                    [
                                        sibling_vertices[i, 0],
                                        sibling_vertices[j, 0],
                                    ],
                                    [
                                        sibling_vertices[i, 1],
                                        sibling_vertices[j, 1],
                                    ],
                                    color="black",
                                )
                                # Midpoint for labeling
                                mid_x = (
                                    sibling_vertices[i, 0]
                                    + sibling_vertices[j, 0]
                                ) / 2
                                mid_y = (
                                    sibling_vertices[i, 1]
                                    + sibling_vertices[j, 1]
                                ) / 2
                                ax.text(
                                    mid_x,
                                    mid_y,
                                    f"{dist_matrix[i, j]:.2f}",
                                    fontsize=8,
                                    color="black",
                                )

            # Draw red dotted lines from the parent to the child vertices
            if parent_key in self.vertices_hierarchy:
                parent_vertex = self.vertices_hierarchy[parent_key]
                if is_3d:
                    ax.plot(
                        [parent_vertex[0], vertices[0]],
                        [parent_vertex[1], vertices[1]],
                        [parent_vertex[2], vertices[2]],
                        "r--",
                    )
                    # Midpoint for labeling
                    mid_x = (parent_vertex[0] + vertices[0]) / 2
                    mid_y = (parent_vertex[1] + vertices[1]) / 2
                    mid_z = (parent_vertex[2] + vertices[2]) / 2
                    ax.text(
                        mid_x,
                        mid_y,
                        mid_z,
                        f"{np.linalg.norm(vertices - parent_vertex):.2f}",
                        fontsize=8,
                        color="red",
                    )
                else:
                    ax.plot(
                        [parent_vertex[0], vertices[0]],
                        [parent_vertex[1], vertices[1]],
                        "r--",
                    )
                    # Midpoint for labeling
                    mid_x = (parent_vertex[0] + vertices[0]) / 2
                    mid_y = (parent_vertex[1] + vertices[1]) / 2
                    ax.text(
                        mid_x,
                        mid_y,
                        f"{np.linalg.norm(vertices - parent_vertex):.2f}",
                        fontsize=8,
                        color="red",
                    )

        # Set plot limits and scaling
        max_range = np.max(
            [np.abs(v).max() for v in self.vertices_hierarchy.values()]
        )
        if is_3d:
            ax.set_xlim([-1.5 * max_range, 1.5 * max_range])
            ax.set_ylim([-1.5 * max_range, 1.5 * max_range])
            ax.set_zlim([-1.5 * max_range, 1.5 * max_range])
        else:
            ax.set_xlim([-1.5 * max_range, 1.5 * max_range])
            ax.set_ylim([-1.5 * max_range, 1.5 * max_range])
            ax.set_aspect("equal", "box")

        plt.show()

    def compute_level_centroids(self):
        """
        Compute the centroid of each parent based on its child vertices.

        Returns:
        - dict: A dictionary with parent keys as keys and centroids as values.
        """
        for (
            parent_key,
            children_keys,
        ) in self.parent_child_relationships.items():
            # Get the child vertices associated with the parent_key
            child_vertices = np.array(
                [
                    self.vertices_hierarchy[child_key]
                    for child_key in children_keys
                ]
            )

            # Calculate the centroid of the child vertices
            centroid = np.mean(child_vertices, axis=0)
            self.centroids_level[parent_key] = centroid

        return self.centroids_level

    def compute_level_metadata(self):
        """
        Compute the metadata for each vertex indicating the depth level in the hierarchy.

        Returns:
        - dict: A dictionary with vertex keys as keys and depth level as values.
        """
        for key in self.vertices_hierarchy.keys():
            level = key.count("-")
            self.level_metadata[key] = level

        return self.level_metadata

    def compute_parent_child_relationships(self):
        """
        Compute the parent-child relationships in the hierarchical simplex.

        Returns:
        - dict: A dictionary where each key represents a parent vertex and each value is a list of its children.
        """
        self.parent_child_relationships = {}

        # Iterate over each vertex in the hierarchy
        for key in self.vertices_hierarchy.keys():
            # Determine the parent key by removing the last segment of the key
            parent_key = "-".join(key.split("-")[:-1])

            # If there's a valid parent key, add the child to the parent's list
            if parent_key != "":
                if parent_key not in self.parent_child_relationships:
                    self.parent_child_relationships[parent_key] = []
                self.parent_child_relationships[parent_key].append(key)

        return self.parent_child_relationships

    def get_vertices(self, keys, store=False):
        """
        Retrieve the coordinates of specific vertices based on their keys.

        Parameters:
        ----------
        keys : list
            A list of vertex keys to retrieve, e.g., ["1-0", "2-1-1"].
        store : bool
            If True, stores the selected vertices internally as _selected_vertices.

        Returns:
        -------
        dict
            A dictionary containing:
            - "vertices_select": The selected vertices and their coordinates.
            - "distance_matrix_select": The distances between the selected vertices from the global distance matrix.
        """
        # Retrieve coordinates of selected vertices
        vertices_select = {
            key: self.vertices_hierarchy[key]
            for key in keys
            if key in self.vertices_hierarchy
        }

        # Store the selected vertices if requested
        if store:
            self._selected_vertices = vertices_select

        # Get the indices for the selected vertices in the global distance matrix
        global_keys = self.distance_matrix_global["keys"]
        selected_indices = [
            global_keys.index(key) for key in keys if key in global_keys
        ]

        # Extract the submatrix of distances between selected vertices
        distance_matrix_select = self.distance_matrix_global[
            "distance_matrix"
        ][np.ix_(selected_indices, selected_indices)]

        # Return the selected vertices and distances
        return {
            "vertices_select": vertices_select,
            "distance_matrix_select": distance_matrix_select,
        }
