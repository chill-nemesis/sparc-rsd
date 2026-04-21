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
    DLSJacobianUpdate,
)


def _main():
    base = RevoluteJoint3D(
        actuation_axis=np.asarray([0, 0, 1]),
        link_offset_mm=np.asarray([2.0, 0.0, 0.0]),
    )
    joint1 = PrismaticJoint3D(
        actuation_axis=np.asarray([1, 0, 0]),
        link_offset_mm=np.asarray([0.0, 0.0, 3.0]),
        parent=base,
    )
    end_effector = RevoluteJoint3D(
        actuation_axis=np.asarray([0, 1, 0]),
        link_offset_mm=np.asarray([2.0, 0.0, 0.0]),
        parent=joint1,
    )

    solver = JacobianIKSolver(
        # delta_theta_update=DLSJacobianUpdate(),
    )

    initial_joint_config = [0, 0, 0]
    constraints = [
        PositionConstraint([4, 2, 4.5]),
        # PositionConstraint([0, 5, 4.5]),
    ]

    _, intermediate_configs = solver.solve(
        constraints,
        end_effector,
        initial_joint_config,
    )

    animate_kinematic_chain(
        end_effector,
        initial_joint_config,
        [
            lambda i: intermediate_configs[min(i, len(intermediate_configs) - 1)][0],
            lambda i: intermediate_configs[min(i, len(intermediate_configs) - 1)][1],
            lambda i: intermediate_configs[min(i, len(intermediate_configs) - 1)][2],
        ],
        frames=len(intermediate_configs) + 10,
    )


if __name__ == "__main__":
    _main()
