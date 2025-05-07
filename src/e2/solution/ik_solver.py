from abc import abstractmethod
import numpy as np

from e2.solution.joint import Joint3D, RevoluteJoint3D, PrismaticJoint3D


class BaseConstraint:
    def __init__(self, target: np.ndarray) -> None:

        self.target = np.array(target, dtype=float)

    @abstractmethod
    def compute_jacobian(
        self,
        joint: Joint3D,
        joint_configurations: list | np.ndarray,
    ) -> np.ndarray:
        """Compute the Jacobian for this constraint.

        Args:
            joint (Joint3D): The joint to compute the Jacobian for.
            joint_configurations (list | np.ndarray): Joint configurations.

        Returns:
            np.ndarray: The Jacobian matrix.
        """

    @abstractmethod
    def compute_error(
        self,
        end_effector: Joint3D,
        joint_configurations: list | np.ndarray,
    ) -> np.ndarray:
        """Compute the error for this constraint.

        Args:
            end_effector (Joint3D): The end effector joint.
            joint_configurations (list | np.ndarray): Joint configurations.

        Returns:
            np.ndarray: The error vector.
        """


class PositionConstraint(BaseConstraint):

    def compute_error(
        self,
        end_effector: Joint3D,
        joint_configurations: list | np.ndarray,
    ) -> np.ndarray:
        ee_pos = end_effector.get_global_position(joint_configurations)
        return self.target - ee_pos

    def compute_jacobian(
        self,
        end_effector: Joint3D,
        joint_configurations: list | np.ndarray,
    ) -> np.ndarray:

        jacobian = np.zeros((3, end_effector.num_joints))

        ee_pos = end_effector.get_global_position(joint_configurations)

        for joint_idx, joint in enumerate(end_effector.kinematic_chain):
            joint_global_axis_of_rotation = joint.global_axis_of_rotation(joint_configurations[: joint_idx + 1])
            joint_global_position = joint.get_global_position(joint_configurations[: joint_idx + 1])

            error = ee_pos - joint_global_position

            if isinstance(joint, RevoluteJoint3D):
                jacobian[:, joint_idx] = np.cross(joint_global_axis_of_rotation, error)
            elif isinstance(joint, PrismaticJoint3D):
                jacobian[:, joint_idx] = joint_global_axis_of_rotation
            else:
                raise ValueError(f"Unsupported joint type: {joint.__class__.__name__}")

        return jacobian


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
        """Create a Jacobian-based inverse kinematics solver.

        Args:
            max_iters (int, optional): The maximum number of iterations to be performed. Defaults to 100.
            alpha (float, optional): The scaling factor for the delta-joint update. Defaults to 0.1.
            error_tolerance (float, optional): The maximum allowed error tolerance. Defaults to 1e-3.
            delta_theta_update (DeltaThetaUpdate, optional): The update-method used for the delta-joint update.
                Defaults to PlainJacobianUpdate().
        """
        self.max_iters = max_iters
        self.alpha = alpha
        self.tolerance = error_tolerance
        self._delta_theta_update = delta_theta_update

    def solve(
        self,
        constraints: list[BaseConstraint],
        end_effector: Joint3D,
        initial_config: list | np.ndarray,
    ) -> tuple[np.ndarray, list[np.ndarray]]:
        """Solves the inverse kinematics problem using the Jacobian method for a given end effector and set of constraints.

        Args:
            constraints (list[BaseConstraint]): The set of constraints to be satisfied.
            end_effector (Joint3D): The end effector joint of the kinematic chain.
            initial_config (list | np.ndarray): The initial joint configuration from which to start the optimization.

        Raises:
            ValueError: At least one constraint must be provided.

        Returns:
            np.ndarray: The final joint configuration that satisfies the constraints and the intermediate joint configurations.
        """
        if len(constraints) == 0:
            raise ValueError("No constraints provided. At least one constraint is required.")

        config = np.array(initial_config, dtype=float)
        intermediate_configs = [config.copy()]

        for _ in range(self.max_iters):
            errors = []
            jacobian_rows = []

            for constraint in constraints:
                j_constraint = constraint.compute_jacobian(end_effector, config)
                error_constraint = constraint.compute_error(end_effector, config)
                jacobian_rows.append(j_constraint)
                errors.append(error_constraint)

            total_error = np.concatenate(errors)
            if np.linalg.norm(total_error) < self.tolerance:
                break

            jacobian = np.vstack(jacobian_rows)

            delta_theta = self._delta_theta_update.apply(jacobian, total_error)

            config += self.alpha * delta_theta
            intermediate_configs.append(config.copy())

        return config, intermediate_configs
