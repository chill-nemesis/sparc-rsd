from abc import abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class Joint3D:
    def __init__(self, axis_of_rotation, length_mm, parent=None):
        """
        Initialize a 3D joint.

        :param axis: Local axis of rotation/translation (must be a unit vector)
        :param length: Fixed length of the link
        :param parent: Parent joint (None for base)
        """
        self.axis_of_rotation = np.array(axis_of_rotation) / np.linalg.norm(axis_of_rotation)  # Normalize
        self.length_mm = length_mm
        self.parent = parent

    @abstractmethod
    def get_transformation_matrix(self, joint_angle_rad):
        """
        Compute the local transformation matrix for this joint.

        :param joint_angle_rad: The joint angle in radians
        :return: 4x4 transformation matrix
        """
        raise NotImplementedError()

    def get_global_position(self, joint_angles_rad):
        """
        Compute the global (x, y, z) position of this joint.

        :param joint_angles: List of joint angles (one per joint).
        :return: (x, y, z) coordinates
        """
        if self.parent is None:
            return np.array([0, 0, 0])  # Base joint starts at origin

        # Get cumulative transformation from the parent
        parent_transform = self.parent.get_cumulative_transformation(joint_angles_rad[:-1])
        local_transform = self.get_transformation_matrix(joint_angles_rad[-1])

        # Compute global transformation
        global_transform = parent_transform @ local_transform

        # Extract position
        return global_transform[:3, 3]

    def get_cumulative_transformation(self, joint_angles_rad):
        """
        Compute the cumulative transformation matrix from base to this joint.

        :param joint_angles: List of joint angles (one per joint).
        :return: 4x4 transformation matrix
        """
        if self.parent is None:
            return np.eye(4)  # Identity matrix for base

        parent_transform = self.parent.get_cumulative_transformation(joint_angles_rad[:-1])
        local_transform = self.get_transformation_matrix(joint_angles_rad[-1])

        return parent_transform @ local_transform


class RevoluteJoint3D(Joint3D):

    def get_transformation_matrix(self, theta_rad):
        # Compute rotation matrix using Rodrigues' formula
        kx, ky, kz = self.axis_of_rotation
        K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])

        R = np.eye(3) + np.sin(theta_rad) * K + (1 - np.cos(theta_rad)) * (K @ K)

        # Homogeneous transformation matrix (with rotation and translation)
        T = np.eye(4)
        T[:3, :3] = R  # Set rotation part
        T[:3, 3] = self.length_mm * self.axis_of_rotation  # Fixed translation along axis

        return T


class PrismaticJoint3D(Joint3D):

    def get_transformation_matrix(self, d):
        # Homogeneous transformation matrix (with rotation and translation)
        T = np.eye(4)
        T[:3, 3] = d * self.axis_of_rotation  # Fixed translation along axis

        return T


def _get_joint_positions(last_joint, joint_angles_rad):
    """Traverse back from the last joint to the base to collect positions."""
    joints = []
    joint = last_joint
    while joint is not None:
        joints.append(joint)
        joint = joint.parent
    joints.reverse()  # Ensure base is first, end-effector is last
    return np.array([joint.get_global_position(joint_angles_rad[: i + 1]) for i, joint in enumerate(joints)])


def _setup_figure():
    """Common function to set up a 3D figure and axis."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Kinematic Chain")
    return fig, ax


def _set_axis_limits(ax, positions):
    """Common function to set equal axis limits based on joint positions."""
    max_range = (
        np.array(
            [
                positions[:, 0].max() - positions[:, 0].min(),
                positions[:, 1].max() - positions[:, 1].min(),
                positions[:, 2].max() - positions[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    mid_x = (positions[:, 0].max() + positions[:, 0].min()) * 0.5
    mid_y = (positions[:, 1].max() + positions[:, 1].min()) * 0.5
    mid_z = (positions[:, 2].max() + positions[:, 2].min()) * 0.5
    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)


def _update(frame, last_joint, joint_angles_rad, update_functions, scatter_plot, line_plot, legend_handles):
    """Update function for the animation, modifying joint angles based on user-defined functions."""
    for i, update_fn in enumerate(update_functions):
        joint_angles_rad[i] = update_fn(frame)

    positions = _get_joint_positions(last_joint, joint_angles_rad)

    # Update positions
    line_plot.set_data(positions[:, 0], positions[:, 1])
    line_plot.set_3d_properties(positions[:, 2])
    scatter_plot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

    # Update legend colors dynamically
    for i, handle in enumerate(legend_handles):
        handle.set_color(scatter_plot.get_facecolor()[i])

    return scatter_plot, line_plot


def animate_kinematic_chain(last_joint, joint_angles_rad, update_functions, frames=200, interval=50):
    """Animate the kinematic chain in 3D with user-defined joint update functions."""
    fig, ax = _setup_figure()
    positions = _get_joint_positions(last_joint, joint_angles_rad)

    num_joints = len(positions)
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_joints))

    # Plot static black links
    (line_plot,) = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "black", linewidth=2)

    # Plot dynamic joints (colored)
    scatter_plot = ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], c=colors, s=100, edgecolors="black")

    # Create legend handles from scatter points
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color=colors[i], markersize=8, linestyle="", label=f"Joint {i+1}")
        for i in range(num_joints)
    ]
    legend_handles[0].set_label("Base")
    legend_handles[-1].set_label("End Effector")
    ax.legend(handles=legend_handles, loc="upper left", fontsize=10)

    _set_axis_limits(ax, positions)

    ani = FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=interval,
        fargs=(last_joint, joint_angles_rad, update_functions, scatter_plot, line_plot, legend_handles),
        blit=False,
    )

    plt.show()
