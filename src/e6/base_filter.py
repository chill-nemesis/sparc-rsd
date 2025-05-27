"""Template for exercise 6 - Robotics in Medicine"""

from abc import abstractmethod
from typing import Any

import numpy as np


class IFilter2D:
    """The interface for a 2D image filter."""

    def __init__(
        self,
        dim_x: int,
        dim_y: int | None = None,
    ) -> None:

        if dim_x % 2 == 0:
            raise ValueError("Kernel size must be odd!")
        if dim_y and dim_y % 2 == 0:
            raise ValueError("Kernel size must be odd!")

        self._shape = (dim_x, dim_y if dim_y else dim_x)

    @property
    def get_x_dim(self) -> int:
        return self._shape[0]

    @property
    def get_y_dim(self) -> int:
        return self._shape[1]

    @property
    def shape(self):
        return self._shape

    def __call__(
        self,
        image: np.ndarray,
    ) -> Any:
        return self.filter(image)

    @abstractmethod
    def filter(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Apply the filter to an image.

        This function returns a copy of the input image with the filtered values.

        Args:
            image (np.ndarray): The input image. Must be a 2D image!

        Returns:
            np.ndarray: A filtered image.
        """

    def invalidate_cache(self) -> None:
        """Invalidates the cache of the filter.
        Use this callback to invalidate cache. This happens for example if
        the current slice in the ui changes. Otherwise you can ignore this method.
        """
