from copy import deepcopy

import open3d as o3d
import numpy as np

from e9._module_loader import ICPSolver


def _main():
    # Load a point cloud
    source = o3d.io.read_point_cloud("data/pointcloud/utah_teapot.ply")

    if source.is_empty():
        raise ValueError("Source point cloud is empty. Please check the file path.")

    # Create a target point cloud by applying a random transformation to the source
    target = deepcopy(source)
    transform = np.eye(4)
    if False: # use random
        transform[:3, :3] = np.linalg.qr(np.random.rand(3, 3))[0]  # Random rotation
        transform[:3, 3] = (np.random.rand(3) - 0.5) * 4  # Random translation
    else:  # use fixed transformation
        transform[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz((0.1, 0.2, 0.3))  # Random rotation
        transform[:3, 3] = [0.5, -0.5, 0.0]  # Random translation


    target.transform(transform)

    # Create an instance of the ICP solver
    icp_solver = ICPSolver(max_iterations=50, tolerance=1e-5)
    # Fit the source point cloud to the target point cloud
    transformation = icp_solver.fit(source, target)

    print(f"Final transformation matrix:\\{transformation}")


if __name__ == "__main__":
    _main()
