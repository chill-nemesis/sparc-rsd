from abc import abstractmethod

import numpy as np

from e1 import math


class Joint3D:
    """Abstract base class for 3D joints."""

    def __init__(
        self,
        actuation_axis,
        link_offset_mm=None,
        parent=None,
    ):
        """
        Initialize a 3D joint.

        Args:
            actuation_axis: Local axis of rotation/translation (must be a non-zero vector) for this joint.
                This axis is expressed in the current joint parent frame.
            link_offset_mm: Optional local translation from the current joint parent frame to the child frame.
            parent: Parent joint (None for the base joint).
        """
        actuation_axis = np.asarray(actuation_axis, dtype=float)
        axis_norm = np.linalg.norm(actuation_axis)
        if axis_norm == 0:
            raise ValueError("Joint axis must not be the zero vector.")

        self.actuation_axis = actuation_axis / axis_norm
        self.parent = parent

        if link_offset_mm is None:
            link_offset_mm = np.zeros(3)

        link_offset_mm = np.asarray(link_offset_mm, dtype=float)
        if link_offset_mm.shape != (3,):
            raise ValueError("link_offset_mm must contain exactly three values.")

        self.link_offset_mm = link_offset_mm

    @property
    def num_joints(self):
        """
        Get the number of joints in the kinematic chain.

        Returns:
            int: Number of joints
        """
        return 1 + (self.parent.num_joints if self.parent else 0)

    @property
    def kinematic_chain(self):
        """
        Get the kinematic chain of this joint.
        Index 0 is the base joint, index -1 is this joint.

        Returns:
            list: List of joints in the kinematic chain
        """
        if self.parent is None:
            return [self]

        return self.parent.kinematic_chain + [self]

    @property
    def local_actuation_axis(self):
        """
        Get the local axis of rotation/translation for this joint.

        Returns:
            np.ndarray: Local axis of rotation/translation (unit vector)
        """
        return self.actuation_axis

    def get_global_parent_from_base(self, joint_configurations):
        """Compute the global transform from the robot base frame to the current joint parent frame."""
        if self.parent is None:
            return math.Transformation.identity()

        # The parent child frame is exactly this joint's parent frame.
        return self.parent.get_global_child_from_base(joint_configurations[:-1])

    @abstractmethod
    def get_local_motion(self, joint_configuration):
        """Compute the motion transform for the current joint.

        Args:
            joint_configuration: Rotation angle in radians for revolute joints or
                translation distance in mm for prismatic joints.

        Returns:
            Transformation: Local transform describing only the joint motion.
        """
        raise NotImplementedError()

    def get_local_child_from_parent(self, joint_configuration) -> math.Transformation:
        """Compute the full local transform from the joint parent frame to the child frame.

        The local transform first applies the configurable joint motion and then the
        fixed transform that places the child frame relative to the joint parent frame.

        Args:
            joint_configuration: The joint configuration.

        Returns:
            Transformation: Local transform from the current joint parent frame to the child frame.
        """
        # Implement here
        raise NotImplementedError(
            "Implement the complete local transformation from the joint parent frame to the child frame."
        )

    def get_global_child_from_base(self, joint_configurations) -> math.Transformation:
        """Compute the global transform from the robot base to this joint's child frame.

        Args:
            joint_configurations: List of joint configurations (one per joint). Index 0 is the first joint,
                ... until index -1 describes the configuration of this joint.

        Returns:
            Transformation: Global transform from the robot base frame to the child frame.
        """
        # Implement here
        raise NotImplementedError()


class RevoluteJoint3D(Joint3D):
    def get_local_motion(self, theta_rad: float):
        """Caclulate the transformation matrix for a revolute joint.
        This matrix represents a rotation around the joint's axis of rotation (see self.actuation_axis).

        Args:
            theta_rad (float): The angle of rotation in radians.
        """
        # Implement here
        raise NotImplementedError()


class PrismaticJoint3D(Joint3D):
    def get_local_motion(self, d: float):
        """Caclulate the transformation matrix for a prismatic joint.
        This matrix represents a translation along the joint's axis of rotation (see self.actuation_axis).

        Args:
            d (float): The distance of translation.
        """
        # Implement here
        raise NotImplementedError()
