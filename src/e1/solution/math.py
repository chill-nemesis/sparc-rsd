"""Implements the solution code for exercises 1.

Author: Steffen Peikert, steffen.peikert@fau.de
Version & Changelog:
- 2.0 (2026-04-20): Moved math to a dedicated file.
"""

import numpy as np

from e1.math import Transformation


def rotation_matrix_from_axis_angle(axis, angle_rad):
    """Create a 3D rotation matrix from an axis-angle representation.

    Args:
        axis: Rotation axis. Does not need to be normalized.
        angle_rad: Rotation angle in radians.

    Returns:
        np.ndarray: 3x3 rotation matrix.
    """
    axis = np.asarray(axis, dtype=float)
    axis_norm = np.linalg.norm(axis)

    if axis_norm == 0:
        raise ValueError("Rotation axis must not be the zero vector.")

    axis = axis / axis_norm
    kx, ky, kz = axis
    skew_matrix = np.array(
        [
            [0, -kz, ky],
            [kz, 0, -kx],
            [-ky, kx, 0],
        ]
    )

    return np.eye(3) + np.sin(angle_rad) * skew_matrix + (1 - np.cos(angle_rad)) * np.dot(skew_matrix, skew_matrix)


def homogeneous_transform(rotation_matrix=None, translation_vector=None):
    """Create a homogeneous transform from rotation and translation parts.

    Args:
        rotation_matrix: Optional 3x3 rotation matrix. Defaults to the identity rotation.
        translation_vector: Optional translation vector with three values. Defaults to zero translation.

    Returns:
        Transformation: Homogeneous transformation with the given rotation and translation.
    """
    transform = np.eye(4)

    if rotation_matrix is not None:
        rotation_matrix = np.asarray(rotation_matrix, dtype=float)
        if rotation_matrix.shape != (3, 3):
            raise ValueError("Rotation matrices must have shape (3, 3).")
        transform[:3, :3] = rotation_matrix

    if translation_vector is not None:
        translation_vector = np.asarray(translation_vector, dtype=float)
        if translation_vector.shape != (3,):
            raise ValueError("Translation vectors must contain exactly three values.")
        transform[:3, 3] = translation_vector

    return Transformation(transform)
