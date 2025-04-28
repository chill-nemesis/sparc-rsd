import numpy as np

from e1.solution.joint import Joint3D


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
