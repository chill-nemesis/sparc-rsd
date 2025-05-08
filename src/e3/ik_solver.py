from abc import abstractmethod
import numpy as np

from e2.solution.joint import Joint3D, RevoluteJoint3D, PrismaticJoint3D


def _rotation_matrix_to_axis_angle(R: np.ndarray) -> np.ndarray:
    """
    Convert a rotation matrix to its axis-angle (rotation vector) representation.
    This is also often called log mapping of a rotation matrix.

    Args:
        R (np.ndarray): Rotation matrix (3x3)

    Returns:
        np.ndarray: Rotation vector (3,) whose direction is the rotation axis
                    and magnitude is the rotation angle in radians.
    """
    assert R.shape == (3, 3), "Rotation matrix must be 3x3."

    theta = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
    if np.isclose(theta, 0):
        return np.zeros(3)

    return (
        theta
        / (2 * np.sin(theta))
        * np.array(
            [
                R[2, 1] - R[1, 2],
                R[0, 2] - R[2, 0],
                R[1, 0] - R[0, 1],
            ]
        )
    )


class BaseConstraint:
    def __init__(self, target: np.ndarray) -> None:

        self.target = np.array(target, dtype=float)

    @abstractmethod
    def compute_jacobian(self, joint: Joint3D, joint_configurations: list | np.ndarray) -> (np.ndarray, np.ndarray):
        """Compute the Jacobian for this constraint.

        Args:
            joint (Joint3D): The joint to compute the Jacobian for.
            joint_configurations (list | np.ndarray): Joint configurations.

        Returns:
            tuple(np.ndarray, np.ndarray): The jacobian matrix for the constraint and joint, as well as the error of the last joint (i.e., the joint that is passed in as an argument).
        """


class PositionConstraint(BaseConstraint):

    def compute_jacobian(
        self,
        end_effector: Joint3D,
        joint_configurations: list | np.ndarray,
    ) -> (np.ndarray, np.ndarray):

        jacobian = np.zeros((3, end_effector.num_joints))

        for joint_idx, joint in enumerate(end_effector.kinematic_chain):
            joint_global_axis_of_rotation = joint.global_axis_of_rotation(joint_configurations[: joint_idx + 1])
            joint_global_position = joint.get_global_position(joint_configurations[: joint_idx + 1])

            error = self.target - joint_global_position

            if isinstance(joint, RevoluteJoint3D):
                jacobian[:, joint_idx] = np.cross(joint_global_axis_of_rotation, error)
            elif isinstance(joint, PrismaticJoint3D):
                jacobian[:, joint_idx] = joint_global_axis_of_rotation
            else:
                raise ValueError(f"Unsupported joint type: {joint.__class__.__name__}")

        return jacobian, error


class RCMConstraint(BaseConstraint):

    def compute_jacobian(
        self,
        end_effector: Joint3D,
        joint_configurations: list | np.ndarray,
    ) -> (np.ndarray, np.ndarray):

        jacobian = np.zeros((3, end_effector.num_joints))

        ee_position = end_effector.get_global_position(joint_configurations)
        ee_direction = end_effector.global_forward_axis(joint_configurations)

        error = np.cross((ee_position - self.target), ee_direction)

        for joint_idx, joint in enumerate(end_effector.kinematic_chain):
            joint_global_axis_of_rotation = joint.global_axis_of_rotation(joint_configurations[: joint_idx + 1])
            joint_global_position = joint.get_global_position(joint_configurations[: joint_idx + 1])

            if isinstance(joint, RevoluteJoint3D):
                Jv_i = np.cross(joint_global_axis_of_rotation, (ee_position - joint_global_position))
            elif isinstance(joint, PrismaticJoint3D):
                Jv_i = joint_global_axis_of_rotation
            else:
                raise ValueError(f"Unsupported joint type: {joint.__class__.__name__}")

            jacobian[:, joint_idx] = np.cross(Jv_i, ee_direction)

        return jacobian, error


class RotationConstraint(BaseConstraint):

    def _rotation_matrix_to_axis_angle(self, R: np.ndarray) -> np.ndarray:
        """
        Convert a rotation matrix to its axis-angle (rotation vector) representation.
        This is also often called log mapping of a rotation matrix.

        Args:
            R (np.ndarray): Rotation matrix (3x3)

        Returns:
            np.ndarray: Rotation vector (3,) whose direction is the rotation axis
                        and magnitude is the rotation angle in radians.
        """
        assert R.shape == (3, 3), "Rotation matrix must be 3x3."

        theta = np.arccos(np.clip((np.trace(R) - 1) / 2.0, -1.0, 1.0))
        if np.isclose(theta, 0):
            return np.zeros(3)

        return (
            theta
            / (2 * np.sin(theta))
            * np.array(
                [
                    R[2, 1] - R[1, 2],
                    R[0, 2] - R[2, 0],
                    R[1, 0] - R[0, 1],
                ]
            )
        )

    def compute_jacobian(
        self,
        end_effector: Joint3D,
        joint_configurations: list | np.ndarray,
    ) -> (np.ndarray, np.ndarray):

        jacobian = np.zeros((3, end_effector.num_joints))

        for joint_idx, joint in enumerate(end_effector.kinematic_chain):
            joint_global_axis_of_rotation = joint.global_axis_of_rotation(joint_configurations[: joint_idx + 1])
            error = _rotation_matrix_to_axis_angle(
                np.dot(
                    self.target[:3, :3],
                    joint.get_global_pose(joint_configurations[: joint_idx + 1])[:3, :3].T,
                )
            )

            if isinstance(joint, RevoluteJoint3D):
                jacobian[:, joint_idx] = joint_global_axis_of_rotation
            elif isinstance(joint, PrismaticJoint3D):
                jacobian[:, joint_idx] = np.zeros(3)
            else:
                raise ValueError(f"Unsupported joint type: {joint.__class__.__name__}")

        return jacobian, error


class DeltaThetaUpdate:
    @abstractmethod
    def apply(self, jacobian, error) -> np.ndarray:
        """_summary_

        Args:
            jacobian (_type_): _description_
            error (_type_): _description_

        Returns:
            _type_: _description_
        """


class PlainJacobianUpdate(DeltaThetaUpdate):
    def apply(self, jacobian, error):
        assert jacobian.shape[0] == error.shape[0], "Jacobian and pose error must have the same number of rows."

        return np.dot(jacobian.T, error)


class DLSJacobianUpdate(DeltaThetaUpdate):
    def __init__(self, damping_factor: float = 0.01):
        self.damping_factor = damping_factor

    def apply(self, jacobian, error):
        assert jacobian.shape[0] == error.shape[0], "Jacobian and pose error must have the same number of rows."

        jjt = np.dot(jacobian, jacobian.T)
        damping = self.damping_factor**2 * np.eye(jacobian.shape[0])
        return np.dot(jacobian.T, np.linalg.solve(jjt + damping, error))


class JacobianIKSolver:
    def __init__(
        self,
        max_iters: int = 100,
        alpha: float = 0.1,
        error_tolerance: float = 1e-3,
        delta_theta_update: DeltaThetaUpdate = PlainJacobianUpdate(),
    ):
        self.max_iters = max_iters
        self.alpha = alpha
        self.tolerance = error_tolerance
        self._delta_theta_update = delta_theta_update

    def solve(
        self,
        constraints: list[BaseConstraint],
        end_effector: Joint3D,
        initial_config: list | np.ndarray,
    ) -> np.ndarray:
        if len(constraints) == 0:
            raise ValueError("No constraints provided. At least one constraint is required.")

        config = np.array(initial_config, dtype=float)
        intermediate_configs = [config.copy()]

        for _ in range(self.max_iters):
            errors = []
            jacobian_rows = []

            for constraint in constraints:
                J, end_effector_error = constraint.compute_jacobian(end_effector, config)
                errors.append(end_effector_error)
                jacobian_rows.append(J)

            total_error = np.concatenate(errors)
            if np.linalg.norm(total_error) < self.tolerance:
                break

            jacobian = np.vstack(jacobian_rows)

            delta_theta = self._delta_theta_update.apply(jacobian, total_error)

            config += self.alpha * delta_theta
            intermediate_configs.append(config.copy())

        return config, intermediate_configs
