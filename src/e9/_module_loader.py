import argparse

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument(
    "--solution",
    action="store_true",
    default=False,
    help="Use the solution code for the magnetic dipole.",
)

args, _ = parser.parse_known_args()

# Switch loading of modules depending if the solution is requested
if args.solution:
    from e9.solution.icp import ICPSolver
else:
    from e9.icp import ICPSolver


__all__ = [
    "ICPSolver",
]
