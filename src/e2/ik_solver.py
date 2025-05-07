from abc import abstractmethod
import numpy as np

from e2._module_loader import Joint3D, RevoluteJoint3D, PrismaticJoint3D


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
    """Provides a constraint for the end effector to reach a target position."""

    def compute_jacobian(
        self,
        end_effector: Joint3D,
        joint_configurations: list | np.ndarray,
    ) -> np.ndarray:

        # Take special care of how each joint type influences the error calculation!
        raise NotImplementedError()

    def compute_error(
        self,
        end_effector: Joint3D,
        joint_configurations: list | np.ndarray,
    ) -> np.ndarray:
        # This needs to compute the error between the end effector position and the target position.
        raise NotImplementedError()


class DeltaThetaUpdate:
    @abstractmethod
    def apply(self, jacobian: np.ndarray, error: np.ndarray) -> np.ndarray:
        """Calculates the delta theta update for a joint configuration using the Jacobian and error.

        Args:
            jacobian (np.ndarray): The Jacobian matrix.
            error (np.ndarray): The corresponding error vector.

        Returns:
            np.ndarray: The delta theta update to the joint configuration.
        """


class PlainJacobianUpdate(DeltaThetaUpdate):
    """Provides the basic Jacobian update without any damping or regularization."""

    def apply(self, jacobian, error):
        assert jacobian.shape[0] == error.shape[0], "Jacobian and pose error must have the same number of rows."

        raise NotImplementedError()


class DLSJacobianUpdate(DeltaThetaUpdate):
    """Provides a damped least-squares Jacobian update."""

    def __init__(self, damping_factor: float = 0.01):
        self.damping_factor = damping_factor

    def apply(self, jacobian, error):
        assert jacobian.shape[0] == error.shape[0], "Jacobian and pose error must have the same number of rows."

        # Optional: Try to implement a damped-least-squares update (also called Levenberg-Marquardt DLS)
        raise NotImplementedError()


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

        # Implement here
        raise NotImplementedError()

        return config, intermediate_configs
