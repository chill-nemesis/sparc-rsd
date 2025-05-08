import numpy as np

from abc import abstractmethod

from _internal.rebase_base_class import REBASE_BASE_CLASS

from e1.solution.joint import (
    Joint3DSharedImpl as E1_Joint3D,
    RevoluteJoint3D as E1_RevoluteJoint3D,
    PrismaticJoint3D as E1_PrismaticJoint3D,
)


class Joint3D(E1_Joint3D):
    """
    Abstract base class for 3D joints.
    """

    def __init__(
        self,
        *args,
        axis_forward: np.ndarray = None,
        **kwargs,
    ):
        """
        Initialize a 3D joint.

        Args:
            axis_forward (np.ndarray, optional): Forward axis. Defaults to None.
        """
        super().__init__(*args, **kwargs)
        self.forward_axis = axis_forward if axis_forward is not None else self.axis_of_rotation

        self.forward_axis /= np.linalg.norm(self.forward_axis)

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
    def local_axis_of_rotation(self):
        """
        Get the local axis of rotation for this joint.

        Returns:
            np.ndarray: Local axis of rotation (unit vector)
        """
        return self.axis_of_rotation

    @property
    def local_forward_axis(self):
        """
        Get the local forward axis for this joint.

        Returns:
            np.ndarray: Local forward axis (unit vector)
        """
        return self.forward_axis

    def global_forward_axis(self, joint_configurations):
        """
        Get the global forward axis for this joint.

        Returns:
            np.ndarray: Global forward axis (unit vector)
        """
        return np.dot(self.get_cumulative_transformation(joint_configurations)[:3, :3], self.forward_axis)

    def global_axis_of_rotation(self, joint_configurations):
        """
        Get the global axis of rotation for this joint.

        Args:
            joint_configurations (list): Joint configurations up to this joint.

        Returns:
            np.ndarray: Global axis of rotation (unit vector)
        """
        return np.dot(self.get_cumulative_transformation(joint_configurations)[:3, :3], self.axis_of_rotation)

    def get_global_pose(self, joint_configurations):
        """
        Compute the global pose of this joint.

        Args:
            joint_configurations (list): Joint configurations up to this joint.

        Returns:
            np.ndarray: 4x4 global transformation matrix describing the joint pose
        """
        return np.dot(
            self.get_cumulative_transformation(joint_configurations),
            self.get_transformation_matrix(joint_configurations[-1]),
        )


class RevoluteJoint3D(REBASE_BASE_CLASS(E1_RevoluteJoint3D, Joint3D)):
    pass

class PrismaticJoint3D(REBASE_BASE_CLASS(E1_PrismaticJoint3D, Joint3D)):
    pass