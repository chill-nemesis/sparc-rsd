"""Implements the solution code for exercises 1.

Author: Steffen Peikert, steffen.peikert@fau.de
Version & Changelog:
- 1.0 (2025-04-28)
- 1.1 (2025-05-01): Reorderd imports
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from e1.joint import Joint3D


class Joint3DSharedImpl(Joint3D):
    """Intermediate class that provides shared functionality for revolute and prismatic joints."""

    def get_global_position(self, joint_configurations):
        """
        Compute the global (x, y, z) position of this joint.

        :param joint_configurations: List of joint configurations (one per joint). Index 0 is the base joint, ... until index -1 describes the joint configuration of this joint.
        :return: (x, y, z) coordinates
        """
        if self.parent is None:
            return np.array([0, 0, 0])  # Base joint starts at origin

        # Extract position
        return self.get_cumulative_transformation(joint_configurations)[:3, 3]

    def get_cumulative_transformation(self, joint_configurations):
        """
        Compute the cumulative homogeneous transformation matrix from base to this joint.

        :param joint_configurations: List of joint configurations (one per joint). Index 0 is the base joint, ... until index -1 describes the joint configuration of this joint.
        :return: 4x4 homogenous transformation matrix
        """
        if self.parent is None:
            parent_transform = np.eye(4)  # Identity matrix for base
        else:
            parent_transform = self.parent.get_cumulative_transformation(joint_configurations[:-1])

        local_transform = self.get_transformation_matrix(joint_configurations[-1])

        return np.dot(parent_transform, local_transform)


class RevoluteJoint3D(Joint3DSharedImpl):

    def get_transformation_matrix(self, theta_rad):
        # Compute rotation matrix using Rodrigues' formula
        kx, ky, kz = self.axis_of_rotation
        K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])

        R = np.eye(3) + np.sin(theta_rad) * K + (1 - np.cos(theta_rad)) * np.dot(K, K)

        # Homogeneous transformation matrix (with rotation and translation)
        T = np.eye(4)
        T[:3, :3] = R  # Set rotation part
        T[:3, 3] = self.length_mm * self.axis_of_rotation  # Fixed translation along axis

        return T


class PrismaticJoint3D(Joint3DSharedImpl):

    def get_transformation_matrix(self, d):
        # Homogeneous transformation matrix (with rotation and translation)
        T = np.eye(4)
        T[:3, 3] = d * self.axis_of_rotation  # Fixed translation along axis

        return T


def _joints_to_list(joint):
    """Recursively collect all joints in the kinematic chain."""
    joints = []
    while joint is not None:
        joints.append(joint)
        joint = joint.parent
    return joints[::-1]  # Reverse to start from base


def _get_joint_types(last_joint):
    # If the joint class contains "prismatic", it is a prismatic joint, otherwise it is a revolute joint.
    return [
        "prismatic" if "Prismatic" in joint.__class__.__name__ else "revolute" for joint in _joints_to_list(last_joint)
    ]


def _get_joint_positions(last_joint, joint_angles_rad):
    """Traverse back from the last joint to the base to collect positions."""
    return np.array(
        [joint.get_global_position(joint_angles_rad[: i + 1]) for i, joint in enumerate(_joints_to_list(last_joint))]
    )


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


def _update(frame, last_joint, joint_configurations, update_functions, scatter_plot, line_plot, legend_handles):
    """Update function for the animation, modifying joint angles based on user-defined functions."""
    for i, update_fn in enumerate(update_functions):
        joint_configurations[i] = update_fn(frame)

    positions = _get_joint_positions(last_joint, joint_configurations)

    # Update positions
    line_plot.set_data(positions[:, 0], positions[:, 1])
    line_plot.set_3d_properties(positions[:, 2])
    scatter_plot._offsets3d = (positions[:, 0], positions[:, 1], positions[:, 2])

    # Update legend colors dynamically
    for i, handle in enumerate(legend_handles):
        handle.set_color(scatter_plot.get_facecolor()[i])

    return scatter_plot, line_plot


def animate_kinematic_chain(last_joint, joint_configurations, update_functions, frames=200, interval=50):
    """Animate the kinematic chain in 3D with user-defined joint update functions."""
    fig, ax = _setup_figure()
    positions = _get_joint_positions(last_joint, joint_configurations)
    joint_types = _get_joint_types(last_joint)

    num_joints = len(positions)
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_joints))

    # Plot static black links
    (line_plot,) = ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], "black", linewidth=2)

    # Plot dynamic joints (colored)
    scatter_plot = ax.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        c=colors,
        s=100,
        edgecolors="black",
    )

    # Create legend handles from scatter points
    legend_handles = [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color=colors[i],
            markersize=8,
            linestyle="",
            label=f"Joint {i+1} end effector ({joint_types[i]})",
        )
        for i in range(num_joints)
    ]

    ax.legend(handles=legend_handles, loc="upper left", fontsize=10)

    _set_axis_limits(ax, positions)

    ani = FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=interval,
        fargs=(last_joint, joint_configurations, update_functions, scatter_plot, line_plot, legend_handles),
        blit=False,
    )

    plt.show()
