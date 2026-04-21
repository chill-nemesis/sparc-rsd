import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--solution",
    action="store_true",
    default=False,
    help="Use the solution code for the kinematic chain animation.",
)

args, _ = parser.parse_known_args()

# Switch loading of modules depending if the solution is requested.
if args.solution:
    from e1.solution.joint import RevoluteJoint3D, PrismaticJoint3D, Joint3DSharedImpl as Joint3D
    from e1.solution.math import Transformation, homogeneous_transform, rotation_matrix_from_axis_angle
else:
    from e1.joint import RevoluteJoint3D, PrismaticJoint3D, Joint3D
    from e1.math import Transformation, homogeneous_transform, rotation_matrix_from_axis_angle

from e1.visualize import animate_kinematic_chain

__all__ = [
    "Joint3D",
    "RevoluteJoint3D",
    "PrismaticJoint3D",
    "Transformation",
    "rotation_matrix_from_axis_angle",
    "homogeneous_transform",
    "animate_kinematic_chain",
]
