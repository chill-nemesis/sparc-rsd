"""Exercise 1: Kinematic chains

Author: Steffen Peikert, steffen.peikert@fau.de
Version & Changelog:
- 1.0 (2025-04-28)
- 1.1 (2025-05-01): Animation is now smooth
"""

import argparse

import numpy as np


from e1.solution.joint import animate_kinematic_chain


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
        axis_of_rotation=np.asarray([1, 1, 0]),
        length_mm=2,
        parent=joint3,
    )

    update_functions = [
        lambda frame: frame / 100 * np.pi,  # Base rotates around z axis
        lambda frame: 3 + .5 * np.sin(frame / 100 * np.pi),  # prismatic joint extending/retracting
        lambda frame: frame / 100 * np.pi,  # Revolute joint rotating around its axis
        lambda frame: frame / 100 * np.pi,  # Revolute joint rotating around its axis
        lambda frame: 0,  # Fixed ee, since we cannot see the rotation anyways
    ]
    animate_kinematic_chain(
        end_effector,
        [fn(0) for fn in update_functions],
        update_functions,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kinematic chain animation")
    parser.add_argument(
        "--solution",
        action="store_true",
        default=False,
        help="Use the solution code for the kinematic chain animation.",
    )

    args = parser.parse_args()

    if args.solution:
        from e1.solution.joint import RevoluteJoint3D, PrismaticJoint3D
    else:
        from e1.joint import RevoluteJoint3D, PrismaticJoint3D

    _main()
