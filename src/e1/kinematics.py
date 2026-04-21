"""Exercise 1: Kinematic chains

Author: Steffen Peikert, steffen.peikert@fau.de
Version & Changelog:
- 1.0 (2025-04-28)
- 1.1 (2025-05-01): Animation is now smooth
- 2.0 (2026-04-20): Refactored to use the new module loader. Joints now have a proper frame hierarchy.
"""

import numpy as np

from e1._module_loader import RevoluteJoint3D, PrismaticJoint3D, animate_kinematic_chain


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

    update_functions = [
        lambda frame: frame / 100 * np.pi,  # Base rotates around z axis
        lambda frame: 1.5 + np.sin(frame / 100 * np.pi),  # Prismatic extension
        lambda frame: -(np.cos(frame / 100 * np.pi) + 1) * np.pi / 2,  # Wavy-Hand motion
    ]
    animate_kinematic_chain(
        end_effector,
        [fn(0) for fn in update_functions],
        update_functions,
    )


if __name__ == "__main__":
    _main()
