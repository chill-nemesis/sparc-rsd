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
    from e4.solution.dipole import MagneticDipole
else:
    from e4.dipole import MagneticDipole


__all__ = [
    "MagneticDipole",
]
