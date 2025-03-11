import numpy as np

from e1.joint import RevoluteJoint3D, PrismaticJoint3D, animate_kinematic_chain


def _main():

    base = base = RevoluteJoint3D([0, 0, 1], 0)
    joint1 = RevoluteJoint3D(
        axis_of_rotation=np.asarray([0, 0, 1]),
        length_mm=5,
        parent=base,
    )
    joint2 = PrismaticJoint3D(
        axis_of_rotation=np.asarray([1, 0, 0]),
        length_mm=10,
        parent=joint1,
    )
    joint3 = RevoluteJoint3D(
        axis_of_rotation=np.asarray([0, 0, 1]),
        length_mm=3,
        parent=joint2,
    )
    ee = RevoluteJoint3D(
        axis_of_rotation=np.asarray([0, 0, 1]),
        length_mm=2,
        parent=joint3,
    )

    update_functions = [
        lambda frame: np.radians(30 * np.sin(frame / 20)),  # Revolute joint oscillating
        lambda frame: 2 + np.sin(frame / 20),  # Prismatic joint extending/retracting
        lambda frame: np.radians(45 * np.cos(frame / 20)),  # Revolute joint oscillating
        lambda frame: np.radians(0),  # Fixed ee, since we cannot see the rotation anyways
    ]
    animate_kinematic_chain(ee, [fn(0) for fn in update_functions], update_functions)


if __name__ == "__main__":
    _main()
