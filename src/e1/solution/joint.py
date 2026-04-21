"""Implements the solution code for exercises 1.

Author: Steffen Peikert, steffen.peikert@fau.de
Version & Changelog:
- 1.0 (2025-04-28)
- 1.1 (2025-05-01): Reorderd imports
- 2.0 (2026-04-20): Refactored to use the new module loader. Joints now have a proper frame hierarchy.
"""

from e1.joint import Joint3D
from e1.solution import math


class Joint3DSharedImpl(Joint3D):
    """Intermediate class that provides shared functionality for revolute and prismatic joints."""

    def get_local_child_from_parent(self, joint_configuration):
        """Compute the full local transform from the current joint parent frame to the child frame."""
        return self.get_local_motion(joint_configuration) @ math.homogeneous_transform(
            translation_vector=self.link_offset_mm
        )

    def get_global_child_from_base(self, joint_configurations):
        """Compute the global transform from the robot base to this joint's child frame."""
        joint_parent = self.get_global_parent_from_base(joint_configurations)
        joint_to_child = self.get_local_child_from_parent(joint_configurations[-1])

        return joint_parent @ joint_to_child


class RevoluteJoint3D(Joint3DSharedImpl):
    """Reference revolute joint implementation for exercise 1."""

    def get_local_motion(self, theta_rad: float):
        """Caclulate the transformation matrix for a revolute joint.
        This matrix represents a rotation around the joint's axis of rotation (see self.actuation_axis).

        Args:
            theta_rad (float): The angle of rotation in radians.
        """
        return math.homogeneous_transform(
            rotation_matrix=math.rotation_matrix_from_axis_angle(
                self.actuation_axis,
                theta_rad,
            )
        )


class PrismaticJoint3D(Joint3DSharedImpl):
    """Reference prismatic joint implementation for exercise 1."""

    def get_local_motion(self, d: float):
        """Caclulate the transformation matrix for a prismatic joint.
        This matrix represents a translation along the joint's axis of rotation (see self.actuation_axis).

        Args:
            d (float): The distance of translation.
        """
        return math.homogeneous_transform(translation_vector=d * self.actuation_axis)
