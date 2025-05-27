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
    from e6.solution.filter import get_filters, IFilter2D
    from common.fancy_colormap import _get_fancy_colormap_solution as get_fancy_colormap
else:
    from e6.filter import get_filters, IFilter2D
    from common.fancy_colormap import get_fancy_colormap


__all__ = [
    "get_filters",
    "IFilter2D",
    "get_fancy_colormap",
]
