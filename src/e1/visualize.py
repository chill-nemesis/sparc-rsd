"""Visualization helpers for exercise 1 kinematic chains.

Author: Steffen Peikert, steffen.peikert@fau.de
Version & Changelog:
- 2.0 (2026-04-20): Moved animation code to a dedicated file. Also improved visualization of different joints.
"""

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


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


def _get_joint_visualization_state(last_joint, joint_configurations):
    """Collect the data required to visualize the current kinematic chain.

    The visualization distinguishes between the joint frame origin, the motion axis
    of that joint, and the child anchor that is reached after the rigid link offset.

    Args:
        last_joint: The final joint in the kinematic chain.
        joint_configurations: Current joint configuration values for the chain.

    Returns:
        dict: Arrays describing joint origins, child anchors, link lengths and axis directions.
    """
    joints = _joints_to_list(last_joint)

    joint_origins = []
    motion_anchors = []
    child_anchors = []
    axis_directions = []
    link_lengths = []

    for joint_idx, joint in enumerate(joints):
        partial_config = joint_configurations[: joint_idx + 1]

        joint_frame_transform = joint.get_global_parent_from_base(partial_config)
        motion_anchor_transform = joint_frame_transform @ joint.get_local_motion(partial_config[-1])
        child_frame_transform = joint.get_global_child_from_base(partial_config)

        joint_origins.append(joint_frame_transform.position)
        motion_anchors.append(motion_anchor_transform.position)
        child_anchors.append(child_frame_transform.position)

        # The motion axis is defined in the joint frame, so we rotate it into the global frame
        # using the orientation of the current joint frame.
        axis_directions.append(joint_frame_transform.rotation @ joint.local_actuation_axis)
        link_lengths.append(np.linalg.norm(child_frame_transform.position - joint_frame_transform.position))

    joint_origins = np.asarray(joint_origins)
    motion_anchors = np.asarray(motion_anchors)
    child_anchors = np.asarray(child_anchors)
    axis_directions = np.asarray(axis_directions)
    link_lengths = np.asarray(link_lengths)

    max_link_length = np.max(link_lengths) if np.max(link_lengths) > 0 else 1.0
    axis_length = max(0.5, 0.35 * max_link_length)
    axis_endpoints = joint_origins + axis_directions * axis_length

    return {
        "joint_origins": joint_origins,
        "motion_anchors": motion_anchors,
        "child_anchors": child_anchors,
        "axis_endpoints": axis_endpoints,
        "joints": joints,
    }


def _setup_figure():
    """Common function to set up a 3D figure and axis."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    ax.set_zlabel("Z-axis")
    ax.set_title("3D Kinematic Chain")
    return fig, ax


def _set_axis_limits(ax, points):
    """Common function to set equal axis limits based on joint positions."""
    if len(points) == 0:
        return

    max_range = (
        np.array(
            [
                points[:, 0].max() - points[:, 0].min(),
                points[:, 1].max() - points[:, 1].min(),
                points[:, 2].max() - points[:, 2].min(),
            ]
        ).max()
        / 2.0
    )
    if max_range == 0:
        max_range = 1.0

    mid_x = (points[:, 0].max() + points[:, 0].min()) * 0.5
    mid_y = (points[:, 1].max() + points[:, 1].min()) * 0.5
    mid_z = (points[:, 2].max() + points[:, 2].min()) * 0.5
    padding = max_range * 1.25
    ax.set_xlim(mid_x - padding, mid_x + padding)
    ax.set_ylim(mid_y - padding, mid_y + padding)
    ax.set_zlim(mid_z - padding, mid_z + padding)


def _collect_animation_extent(last_joint, initial_joint_configurations, update_functions, frames):
    """Estimate the full spatial extent of the animation.

    The animation uses a fixed set of axis limits. To keep all poses visible, we sample
    every animation frame up front and collect the corresponding joint origins, motion anchors, link anchors
    and revolute-axis markers. This is cheap for the exercise-scale animations and yields
    much more robust limits than only looking at the initial pose.

    Args:
        last_joint: The final joint in the kinematic chain.
        initial_joint_configurations: Initial configuration values. These are copied locally.
        update_functions: Per-joint update functions used by the animation.
        frames: Either an integer frame count or an iterable of frame values.

    Returns:
        np.ndarray: Nx3 array of points that should fit into the plot.
    """
    if isinstance(frames, int):
        frame_values = range(frames)
    else:
        frame_values = list(frames)

    sampled_points = [np.array([[0.0, 0.0, 0.0]])]
    sampled_configurations = list(np.asarray(initial_joint_configurations, dtype=float).copy())

    for frame in frame_values:
        for joint_idx, update_fn in enumerate(update_functions):
            sampled_configurations[joint_idx] = update_fn(frame)

        visualization_state = _get_joint_visualization_state(last_joint, sampled_configurations)
        sampled_points.extend(
            [
                visualization_state["joint_origins"],
                visualization_state["motion_anchors"],
                visualization_state["child_anchors"],
                visualization_state["axis_endpoints"],
            ]
        )

    return np.vstack(sampled_points)


def _update(
    frame,
    last_joint,
    joint_configurations,
    update_functions,
    joint_origin_plot,
    rigid_link_lines,
    revolute_axis_lines,
    revolute_axis_joint_indices,
    prismatic_motion_lines,
    prismatic_motion_joint_indices,
):
    """Update function for the animation, modifying joint angles based on user-defined functions."""
    for i, update_fn in enumerate(update_functions):
        joint_configurations[i] = update_fn(frame)

    visualization_state = _get_joint_visualization_state(last_joint, joint_configurations)
    joint_origins = visualization_state["joint_origins"]
    motion_anchors = visualization_state["motion_anchors"]
    child_anchors = visualization_state["child_anchors"]
    axis_endpoints = visualization_state["axis_endpoints"]

    joint_origin_plot._offsets3d = (joint_origins[:, 0], joint_origins[:, 1], joint_origins[:, 2])

    for joint_idx, link_line in enumerate(rigid_link_lines):
        segment = np.vstack((motion_anchors[joint_idx], child_anchors[joint_idx]))
        link_line.set_data(segment[:, 0], segment[:, 1])
        link_line.set_3d_properties(segment[:, 2])

    for axis_line_idx, axis_line in enumerate(revolute_axis_lines):
        joint_idx = revolute_axis_joint_indices[axis_line_idx]
        segment = np.vstack((joint_origins[joint_idx], axis_endpoints[joint_idx]))
        axis_line.set_data(segment[:, 0], segment[:, 1])
        axis_line.set_3d_properties(segment[:, 2])

    for motion_line_idx, motion_line in enumerate(prismatic_motion_lines):
        joint_idx = prismatic_motion_joint_indices[motion_line_idx]
        segment = np.vstack((joint_origins[joint_idx], motion_anchors[joint_idx]))
        motion_line.set_data(segment[:, 0], segment[:, 1])
        motion_line.set_3d_properties(segment[:, 2])

    return [joint_origin_plot, *rigid_link_lines, *revolute_axis_lines, *prismatic_motion_lines]


def animate_kinematic_chain(last_joint, joint_configurations, update_functions, frames=200, interval=50):
    """Animate the kinematic chain in 3D with user-defined joint update functions."""
    fig, ax = _setup_figure()
    visualization_state = _get_joint_visualization_state(last_joint, joint_configurations)
    joint_origins = visualization_state["joint_origins"]
    motion_anchors = visualization_state["motion_anchors"]
    child_anchors = visualization_state["child_anchors"]
    axis_endpoints = visualization_state["axis_endpoints"]
    joints = visualization_state["joints"]
    joint_types = _get_joint_types(last_joint)

    num_joints = len(child_anchors)
    colors = plt.cm.coolwarm(np.linspace(0, 1, num_joints))

    # base_marker = ax.scatter(
    #     [0],
    #     [0],
    #     [0],
    #     c="black",
    #     s=120,
    #     marker="s",
    #     label="Base frame",
    #     edgecolors="black",
    # )

    joint_origin_plot = ax.scatter(
        joint_origins[:, 0],
        joint_origins[:, 1],
        joint_origins[:, 2],
        c=colors,
        s=110,
        marker="o",
        edgecolors="black",
    )
    rigid_link_lines = []
    revolute_axis_lines = []
    prismatic_motion_lines = []
    legend_handles = []
    revolute_axis_joint_indices = []
    prismatic_motion_joint_indices = []

    for joint_idx, _joint in enumerate(joints):
        link_segment = np.vstack((motion_anchors[joint_idx], child_anchors[joint_idx]))

        (link_line,) = ax.plot(
            link_segment[:, 0],
            link_segment[:, 1],
            link_segment[:, 2],
            color="black",
            linewidth=2,
        )
        rigid_link_lines.append(link_line)

        if "revolute" in joint_types[joint_idx]:
            axis_segment = np.vstack((joint_origins[joint_idx], axis_endpoints[joint_idx]))
            (axis_line,) = ax.plot(
                axis_segment[:, 0],
                axis_segment[:, 1],
                axis_segment[:, 2],
                color="tab:blue",
                linestyle="-",
                linewidth=2,
            )
            revolute_axis_lines.append(axis_line)
            revolute_axis_joint_indices.append(joint_idx)
        else:
            motion_segment = np.vstack((joint_origins[joint_idx], motion_anchors[joint_idx]))
            (motion_line,) = ax.plot(
                motion_segment[:, 0],
                motion_segment[:, 1],
                motion_segment[:, 2],
                color="tab:orange",
                linestyle="--",
                linewidth=2,
            )
            prismatic_motion_lines.append(motion_line)
            prismatic_motion_joint_indices.append(joint_idx)

    legend_handles.extend(
        [
            plt.Line2D(
                [0],
                [0],
                marker="o",
                color=colors[i],
                markersize=8,
                linestyle="",
                markeredgecolor="black",
                label=f"Joint {i + 1} origin ({joint_types[i]})",
            )
            for i in range(num_joints)
        ]
    )
    legend_handles.extend(
        [
            plt.Line2D([0], [0], color="black", linewidth=2, label="Rigid link"),
            plt.Line2D([0], [0], color="tab:blue", linewidth=2, label="Revolute axis"),
            plt.Line2D([0], [0], color="tab:orange", linewidth=2, linestyle="--", label="Prismatic motion"),
        ]
    )

    ax.legend(handles=legend_handles, loc="upper left", fontsize=9)

    # Use the whole animation envelope instead of only the initial pose. Otherwise a joint can
    # easily move behind one of the plot panes once the chain reaches farther than its start pose.
    _set_axis_limits(
        ax,
        _collect_animation_extent(
            last_joint,
            joint_configurations,
            update_functions,
            frames,
        ),
    )

    ani = FuncAnimation(
        fig,
        _update,
        frames=frames,
        interval=interval,
        fargs=(
            last_joint,
            joint_configurations,
            update_functions,
            joint_origin_plot,
            rigid_link_lines,
            revolute_axis_lines,
            revolute_axis_joint_indices,
            prismatic_motion_lines,
            prismatic_motion_joint_indices,
        ),
        blit=False,
    )

    # Keep a reference alive for matplotlib so the animation is not garbage-collected.
    fig._animation = ani

    plt.show()
