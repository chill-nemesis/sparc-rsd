from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Transformation:
    """Representation of a homogeneous 4x4 transformation matrix.

    The class mainly exists so callers can explicitly access the rotation and
    translation parts of a homogeneous transform without repeatedly slicing
    NumPy arrays throughout the exercises.
    """

    matrix: np.ndarray

    def __post_init__(self):
        matrix = np.asarray(self.matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError("Transformation matrices must have shape (4, 4).")
        object.__setattr__(self, "matrix", matrix)

    @property
    def rotation(self) -> np.ndarray:
        """Return the 3x3 rotation block."""
        return self.matrix[:3, :3]

    @property
    def position(self) -> np.ndarray:
        """Return the 3D translation part."""
        return self.matrix[:3, 3]

    def __matmul__(self, other):
        """Compose two transformations using the ``@`` operator."""
        if isinstance(other, Transformation):
            return Transformation(np.dot(self.matrix, other.matrix))
        if hasattr(other, "matrix"):
            return Transformation(np.dot(self.matrix, np.asarray(other.matrix, dtype=float)))
        return Transformation(np.dot(self.matrix, np.asarray(other, dtype=float)))

    @classmethod
    def identity(cls):
        """Create the identity transformation."""
        return cls(np.eye(4))


def rotation_matrix_from_axis_angle(axis, angle_rad) -> np.ndarray:
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

    # Implement Here
    raise NotImplementedError("Implement this function to create a rotation matrix from an axis-angle representation.")


def homogeneous_transform(rotation_matrix=None, translation_vector=None) -> Transformation:
    """Create a homogeneous transform from rotation and translation parts.

    Args:
        rotation_matrix: Optional 3x3 rotation matrix. Defaults to the identity rotation.
        translation_vector: Optional translation vector with three values. Defaults to zero translation.

    Returns:
        Transformation: Homogeneous transformation with the given rotation and translation.
    """

    # Implement Here
    raise NotImplementedError(
        "Implement this function to create a homogeneous transformation matrix from rotation and translation components."
    )
