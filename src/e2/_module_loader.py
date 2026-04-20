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
from e1._module_loader import Joint3D, RevoluteJoint3D, PrismaticJoint3D, animate_kinematic_chain

if args.solution:
    from e2.solution.ik_solver import (
        JacobianIKSolver,
        PositionConstraint,
        PlainJacobianUpdate,
        DLSJacobianUpdate,
    )
else:
    from e2.ik_solver import (
        JacobianIKSolver,
        PositionConstraint,
        PlainJacobianUpdate,
        DLSJacobianUpdate,
    )


__all__ = [
    "RevoluteJoint3D",
    "PrismaticJoint3D",
    "Joint3D",
    "animate_kinematic_chain",
    "JacobianIKSolver",
    "PositionConstraint",
    "PlainJacobianUpdate",
    "DLSJacobianUpdate",
]
