"""Code template for exercise 7 - delaunay triangulation"""

import argparse

import numpy as np

from e7._module_loader import BowyerWatsonVisualiser


def _create_random_points(xdim, ydim, zdim, n):

    return np.vstack(
        (
            np.random.uniform(-xdim, xdim, n),
            np.random.uniform(-ydim, zdim, n),
            np.random.uniform(-zdim, zdim, n),
        )
    ).T


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-points",
        "-n",
        type=int,
        default=7,
    )
    parser.add_argument(
        "--xdim",
        "-x",
        type=float,
        default=100,
    )
    parser.add_argument(
        "--ydim",
        "-y",
        type=float,
        default=100,
    )
    parser.add_argument(
        "--zdim",
        "-z",
        type=float,
        default=100,
    )

    # Only read known arguments to avoid issues with the solution code
    args = parser.parse_known_args()[0]

    points = _create_random_points(
        args.xdim,
        args.ydim,
        args.zdim,
        args.num_points,
    )

    BowyerWatsonVisualiser(points)


if __name__ == "__main__":
    _main()
