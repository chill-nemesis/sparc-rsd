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
    from e7.solution.bowyer_watson import BowyerWatsonVisualiser
else:
    from e7.bowyer_watson import BowyerWatsonVisualiser


__all__ = [
    "BowyerWatsonVisualiser",
]
