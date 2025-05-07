import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--solution",
    action="store_true",
    default=False,
    help="Use the solution code for the inverse kinematic chain animation.",
)

args, _ = parser.parse_known_args()

# Switch loading of modules depending if the solution is requested
if args.solution:
    from e1.solution.joint import (
        Joint3DSharedImpl as E1_Joint3D,
        RevoluteJoint3D as E1_RevoluteJoint3D,
        PrismaticJoint3D as E1_PrismaticJoint3D,
    )
    from e2.solution.ik_solver import (
        JacobianIKSolver,
        PositionConstraint,
        PlainJacobianUpdate,
        DLSJacobianUpdate,
    )
else:
    from e1.joint import (
        Joint3D as E1_Joint3D,
        RevoluteJoint3D as E1_RevoluteJoint3D,
        PrismaticJoint3D as E1_PrismaticJoint3D,
    )
    from e2.ik_solver import (
        JacobianIKSolver,
        PositionConstraint,
        PlainJacobianUpdate,
        DLSJacobianUpdate,
    )

from e1.solution.joint import animate_kinematic_chain
from e2.solution.joint import RevoluteJoint3D, PrismaticJoint3D, Joint3D


__all__ = [
    "E1_Joint3D",
    "E1_RevoluteJoint3D",
    "E1_PrismaticJoint3D",
    "RevoluteJoint3D",
    "PrismaticJoint3D",
    "Joint3D",
    "animate_kinematic_chain",
    "JacobianIKSolver",
    "PositionConstraint",
    "PlainJacobianUpdate",
    "DLSJacobianUpdate",
]
