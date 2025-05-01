import numpy as np


from abc import abstractmethod


class Joint3D:
    """Abstract base class for 3D joints."""

    def __init__(self, axis_of_rotation, length_mm, parent=None):
        """
        Initialize a 3D joint.

        :param axis: Local axis of rotation/translation (must be a unit vector) for this joint. When initially created, this joint uses the global coordinate system. Only when attached to a parent, the coordinate system becomes local.
        :param length: Fixed length of the link
        :param parent: Parent joint (None for base)
        """
        self.axis_of_rotation = np.array(axis_of_rotation) / np.linalg.norm(axis_of_rotation)  # Normalize
        self.length_mm = length_mm
        self.parent = parent

    @abstractmethod
    def get_transformation_matrix(self, joint_configuration):
        """
        Compute the local transformation matrix for this joint.

        :param joint_configuration: The joint configuration
        :return: 4x4 transformation matrix
        """
        raise NotImplementedError()

    @abstractmethod
    def get_global_position(self, joint_configurations):
        """
        Compute the global (x, y, z) position of this joint.

        :param joint_configurations: List of joint configurations (one per joint). Index 0 is the base joint, ... until index -1 describes the joint configuration of this joint.
        :return: (x, y, z) coordinates
        """
        raise NotImplementedError()

    @abstractmethod
    def get_cumulative_transformation(self, joint_configurations):
        """
        Compute the cumulative homogeneous transformation matrix from base to this joint.

        :param joint_configurations: List of joint configurations (one per joint). Index 0 is the base joint, ... until index -1 describes the joint configuration of this joint.
        :return: 4x4 homogenous transformation matrix
        """
        raise NotImplementedError()


class RevoluteJoint3D(Joint3D):

    def get_transformation_matrix(self, theta_rad: float):
        """Caclulate the transformation matrix for a revolute joint.
        This matrix represents a rotation around the joint's axis of rotation (see self.axis_of_rotation).

        Args:
            theta_rad (float): The angle of rotation in radians.
        """
        # Implement here

        pass


class PrismaticJoint3D(Joint3D):

    def get_transformation_matrix(self, d: float):
        """Caclulate the transformation matrix for a prismatic joint.
        This matrix represents a translation along the joint's axis of rotation (see self.axis_of_rotation).

        Args:
            d (float): The distance of translation.
        """
        # Implement here

        pass
