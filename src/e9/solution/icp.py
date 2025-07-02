import numpy as np

from e9.icp import ICPSolver as ICP_STUDENT


class ICPSolver(ICP_STUDENT):
    def _compute_delta_transformation(self, source_matched, target_matched):
        # Compute centroids of the matched points
        source_centroid = np.mean(source_matched, axis=0)
        target_centroid = np.mean(target_matched, axis=0)

        # Center the points around the origin
        source_centered = source_matched - source_centroid
        target_centered = target_matched - target_centroid

        # Cacluate the covariance matrix
        covariance = np.dot(source_centered.T, target_centered)

        # Singular Value Decomposition (SVD)
        U, _, Vt = np.linalg.svd(covariance)

        # Compute the rotation matrix
        R = np.dot(Vt.T, U.T)

        if np.linalg.det(R) < 0:
            # If the determinant is negative, we need to correct the rotation matrix
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)

        # Compute the translation vector
        t = target_centroid - np.dot(R, source_centroid)

        # Create the transformation matrix
        transformation_matrix = np.eye(4)
        transformation_matrix[:3, :3] = R
        transformation_matrix[:3, 3] = t

        return transformation_matrix

    def _find_nearest_neighbors(self, source, target, target_tree):

        # For each point in the source, find the nearest point in the target
        source_matched = np.zeros((len(source.points), 3))
        target_matched = np.zeros((len(source.points), 3))

        for s_idx, p in enumerate(source.points):
            [_, idx, _] = target_tree.search_knn_vector_3d(p, 1)

            source_matched[s_idx] = p
            target_matched[s_idx] = target.points[idx[0]]

        return source_matched, target_matched

    def _compute_error(self, source, target):
        # Compute the mean squared error between the source and target point clouds
        source_points = np.asarray(source.points)
        target_points = np.asarray(target.points)

        if len(source_points) != len(target_points):
            raise ValueError("Source and target point clouds must have the same number of points.")

        error = np.mean(np.linalg.norm(source_points - target_points, axis=1) ** 2)
        return error
