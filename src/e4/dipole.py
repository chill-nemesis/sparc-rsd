import numpy as np

DENSITY_IRON = 7.874e3  # kg/m^3
DENSITY_COPPER = 8.96e3  # kg/m^3


VISCOSITY_WATER = 0.001  # Pa.s


def _rodrigues(axis, angle):
    # Rodrigues' rotation formula
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)


class MagneticDipole:
    def __init__(
        self,
        radius=1e-6,
        magnetization=[2e3, 0, 0],
        dipole_density=DENSITY_IRON,
    ) -> None:
        self.radius = radius
        self.volume = (4 / 3) * np.pi * self.radius**3  # Volume of a sphere
        self.material_density = dipole_density
        self.magnetization = np.asarray(magnetization)

        self.reset()

    @property
    def global_magnetization(self):
        raise NotImplementedError()

    def magnetic_torque(self, B):
        raise NotImplementedError()

    def magnetic_force(self, GradB):
        raise NotImplementedError()

    def update(
        self,
        local_B,
        grad_B,
        dt,
        fluid_viscosity=VISCOSITY_WATER,
    ):
        # Compute torque and force
        T = self.magnetic_torque(local_B)
        F = self.magnetic_force(grad_B)

        angular_velocity = T / (8 * np.pi * self.radius**3 * fluid_viscosity)
        velocity = F / (6 * np.pi * self.radius * fluid_viscosity)

        # Integrate velocity to update position and orientation
        self.position += velocity * dt
        self._update_orientation(angular_velocity, dt)

        return T, F

    def reset(self):
        self.position = np.zeros(3)
        self.orientation = np.eye(3)

    def _update_orientation(
        self,
        angular_velocity,
        dt,
    ):
        # Simple integration to update orientation using angular velocity
        angle = np.linalg.norm(angular_velocity * dt)
        if angle < 1e-3:
            return

        axis = angular_velocity / np.linalg.norm(angular_velocity)
        self.orientation = np.dot(_rodrigues(axis, angle), self.orientation)
