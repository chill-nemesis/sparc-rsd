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


class BaseFilter2D(IFilter2D):
    def __init__(
        self,
        dim_x: int,
        dim_y: int | None = None,
    ) -> None:
        super().__init__(dim_x, dim_y)

        self._cached_image: None | np.ndarray = None

    def invalidate_cache(self):
        self._cached_image = None

    def __call__(
        self,
        image: np.ndarray,
        force_refilter: bool = False,
        border_behaviour="wrap",
        **kwargs,
    ) -> Any:
        return self.filter(image, force_refilter, border_behaviour, **kwargs)

    def filter(
        self,
        image: np.ndarray,
        force_refilter: bool = False,
        border_behaviour="wrap",
        **kwargs,
    ) -> np.ndarray:
        """Apply the filter to an image.

        This function returns a copy of the input image with the filtered values.

        Args:
            image (np.ndarray): The input image. Must be a 2D image!

        Returns:
            np.ndarray: A filtered image.
        """
        if self._cached_image is not None and not force_refilter:
            return self._cached_image

        sliding_window_views = self._get_sub_views(image, border_behaviour, **kwargs)

        vec_filter_func = np.vectorize(self._apply_filter_func, signature="(m,n)->()")
        filtered_values = vec_filter_func(sliding_window_views)

        out_image = np.zeros((sliding_window_views.shape[0], sliding_window_views.shape[1]))
        out_image[:] = filtered_values.reshape(out_image.shape)

        self._cached_image = out_image

        return out_image

    def _get_sub_views(self, image: np.ndarray, border_behaviour, **kwargs):

        # Do not add any new information/data to the array, only use the existing, valid pixels
        if border_behaviour is None:
            return np.lib.stride_tricks.sliding_window_view(image, self.shape)

        padded_image = np.pad(
            image,
            pad_width=(self.get_x_dim // 2 + 1, self.get_y_dim // 2 + 1),
            mode=border_behaviour,
            **kwargs,
        )

        return np.lib.stride_tricks.sliding_window_view(padded_image, self.shape)

    @abstractmethod
    def _apply_filter_func(self, selected_pixels: np.ndarray):
        pass


class BaseKernelFilter2D(BaseFilter2D):
    def __init__(self, dim_x: int, dim_y: int | None = None):
        super().__init__(dim_x, dim_y)

        self.kernel = self._create_kernel()

    def _apply_filter_func(self, selected_pixels: np.ndarray):
        return np.sum(self.kernel * selected_pixels, axis=(0, 1))

    @abstractmethod
    def _create_kernel(self):
        pass
