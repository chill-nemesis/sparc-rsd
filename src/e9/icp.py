"""Code template for exercise 9 - Iterative Closest Point (ICP) algorithm."""

from time import sleep
from copy import deepcopy

import open3d as o3d
import numpy as np


class ICPSolver:
    def __init__(
        self,
        max_iterations=100,
        tolerance=1e-5,
    ):
        self.max_iterations = max_iterations
        self.tolerance = tolerance

        self._do_next_iteration = False

    def fit(self, source, target):
        """
        Fit the source point cloud to the target point cloud using the ICP algorithm.

        Parameters:
        - source: The source point cloud to be aligned.
        - target: The target point cloud to align against.

        Returns:
        - The transformed source point cloud.
        """

        # Make a copy of the source point cloud to avoid modifying the original
        local_source = deepcopy(source)
        local_source.paint_uniform_color([1, 0, 0])  # Color the source red

        target.paint_uniform_color([0, 1, 0])  # Color the target green

        target_tree = o3d.geometry.KDTreeFlann(target)

        current_transformation = np.eye(4)

        # Setup visualization
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.register_key_callback(ord(" "), lambda vis: setattr(self, "_do_next_iteration", True))
        vis.create_window()
        vis.add_geometry(local_source)
        vis.add_geometry(target)

        for iteration in range(self.max_iterations):

            self._do_next_iteration = False

            # Find nearest neighbors in the target for each point in the source
            source_matched, target_matched = self._find_nearest_neighbors(
                local_source,
                target,
                target_tree,
            )

            # Compute the transformation matrix from matched points
            delta_transformation = self._compute_delta_transformation(source_matched, target_matched)

            # Update the current transformation
            current_transformation = np.dot(delta_transformation, current_transformation)

            # Update the source point cloud with the new transformation
            local_source.transform(delta_transformation)

            # Update the visualization
            vis.update_geometry(local_source)
            while not self._do_next_iteration:
                vis.poll_events()
                vis.update_renderer()
                sleep(1.0 / 30)  # run at 30 FPS

            # Check for convergence
            error = self._compute_error(local_source, target)
            print(f"Iteration {iteration + 1}, Error: {error:.6f}")
            if error < self.tolerance:
                break

        print(f"Finished ICP after {iteration + 1} iterations.")
        print("Press q to close the visualization window.")
        vis.run()
        vis.destroy_window()

        return current_transformation

    def _compute_error(self, source, target) -> float:
        """Compute the error between the source and target point clouds.

        Args:
            source (o3d.pointcloud): The source point cloud (i.e., the one being transformed).
            target (o3d.pointcloud): The target point cloud (i.e., the one to match against).

        Returns:
            float: The mean squared error between the source and target point clouds.
        """
        return 100  # Placeholder for error computation

    def _find_nearest_neighbors(
        self,
        source,
        target,
        target_tree,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Calculate the nearest neighbors for all points in the source point cloud to the target point cloud and return two lists of matched points.
        Let's call the resulting lists `source_matched` and `target_matched`.
        source_matched[i] is the point in the source point cloud that corresponds to target_matched[i] in the target point cloud (i.e., the nearest neighbor in the target point cloud).

        Args:
            source (o3d.pointcloud): The source point cloud (i.e., the one being transformed).
            target (o3d.pointcloud): The target point cloud (i.e., the one to match against).
            target_tree (o3d.KDTreeFlann): A KDTree for the target point cloud to speed up nearest neighbor search.

        Returns:
            tuple[np.ndarray, np.ndarray]: A tuple containing two lists of matched points.
        """
        pass

    def _compute_delta_transformation(
        self,
        source_matched: np.ndarray,
        target_matched: np.ndarray,
    ) -> np.ndarray:
        """Compute the delta transformation matrix for the current iteration of ICP that aligns the source point cloud to the target point cloud based on the matched points.

        Args:
            source_matched (np.ndarray): The matched points from the source point cloud.
            target_matched (np.ndarray): The matched points from the target point cloud.

        Returns:
            np.ndarray: The transformation matrix that aligns the source point cloud to the target point cloud.
        """
        pass
