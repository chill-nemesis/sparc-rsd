"""Exercise 2: Inverse Kinematics

Author: Steffen Peikert, steffen.peikert@fau.de
Version & Changelog:
- 1.0 (2025-05-01)
"""

import numpy as np

from e2._module_loader import (
    RevoluteJoint3D,
    PrismaticJoint3D,
    animate_kinematic_chain,
    JacobianIKSolver,
    PositionConstraint,
)


def _main():
    base = base = RevoluteJoint3D([0, 0, 1], 0)
    joint1 = PrismaticJoint3D(
        axis_of_rotation=np.asarray([0, 0, 1]),
        length_mm=3,
        parent=base,
    )
    joint2 = RevoluteJoint3D(
        axis_of_rotation=np.asarray([1, 0, 0]),
        length_mm=1.5,
        parent=joint1,
    )
    joint3 = RevoluteJoint3D(
        axis_of_rotation=np.asarray([0, 1, 0]),
        length_mm=1.5,
        parent=joint2,
    )
    end_effector = RevoluteJoint3D(
        axis_of_rotation=np.asarray([1, 0, 0]),
        length_mm=1,
        parent=joint3,
    )


    solver = JacobianIKSolver()

    initial_joint_config = [0, 0, 0, 0, 0]
    constraints = [
        PositionConstraint([2, 0, 0]),
    ]

    _, intermediate_configs = solver.solve(
        constraints,
        end_effector,
        initial_joint_config,
    )

    # TODO for students:
    # unstable with larger alpha
    # implement DLS

    animate_kinematic_chain(
        end_effector,
        initial_joint_config,
        [
            lambda i: intermediate_configs[min(i, len(intermediate_configs) - 1)][0],
            lambda i: intermediate_configs[min(i, len(intermediate_configs) - 1)][1],
            lambda i: intermediate_configs[min(i, len(intermediate_configs) - 1)][2],
            lambda i: intermediate_configs[min(i, len(intermediate_configs) - 1)][3],
            lambda i: intermediate_configs[min(i, len(intermediate_configs) - 1)][4],
        ],
        frames=len(intermediate_configs) + 10,
    )


if __name__ == "__main__":
    _main()
