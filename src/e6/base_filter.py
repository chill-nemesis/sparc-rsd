"""Template for exercise 6 - Robotics in Medicine"""

from abc import abstractmethod
from enum import Enum, auto
from typing import Any

import numpy as np


class BorderBehaviour(Enum):
    """Defines how filters handle border pixels.

    WRAP: Wrap the values
    REPEAT: Repeat the last value
    EMPTY: Assume zero
    ONLY_VALID: Only filter regions where we have valid data. This reduces the output picture size!
    """

    WRAP = auto()
    REPEAT = auto()
    ZERO = auto()


class BaseFilter2D:
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

        self._cached_image: None | np.ndarray = None

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
        border_behaviour: BorderBehaviour = BorderBehaviour.WRAP,
        force_refilter: bool = False,
    ) -> Any:
        return self.filter(image, border_behaviour, force_refilter)

    def filter(
        self,
        image: np.ndarray,
        border_behaviour: BorderBehaviour = BorderBehaviour.WRAP,
        force_refilter: bool = False,
    ) -> np.ndarray:
        """Apply the filter to an image.

        This function returns a copy of the input image with the filtered values.

        Args:
            image (np.ndarray): The input image. Must be a 2D image!
            border_behaviour (BorderBehaviour, optional): Defines, how the filter handles pixels outside the input image boundaries. Defaults to BorderBehaviour.WRAP.

        Returns:
            np.ndarray: A filtered image.
        """
        if self._cached_image is not None and not force_refilter:
            return self._cached_image

        out_image = np.zeros(image.shape)

        for x in range(image.shape[0]):
            for y in range(image.shape[1]):
                out_image[x, y] = self._apply_filter_func(
                    self._get_image_subarray(
                        image,
                        x,
                        y,
                        border_behaviour,
                    )
                )

        self._cached_image = out_image

        return out_image

    def _get_image_subarray(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        border_behaviour: BorderBehaviour = BorderBehaviour.WRAP,
    ):
        subimage = np.zeros(
            (self.get_x_dim, self.get_y_dim),
            dtype=image.dtype,
        )

        half_filter_x = self.get_x_dim // 2
        half_filter_y = self.get_y_dim // 2

        for i in range(-half_filter_x, half_filter_x + 1):
            for k in range(-half_filter_y, half_filter_y + 1):
                sub_idx_x = i + half_filter_x
                sub_idx_y = k + half_filter_y

                img_idx_x = i + x
                img_idx_y = k + y

                match border_behaviour:
                    ### REGION SOLUTION
                    case BorderBehaviour.REPEAT:
                        img_idx_x = min(max(img_idx_x, 0), image.shape[0] - 1)
                        img_idx_y = min(max(img_idx_y, 0), image.shape[1] - 1)
                        image_value = image[img_idx_x, img_idx_y]

                    case BorderBehaviour.ZERO:
                        if 0 <= img_idx_x < image.shape[0] and 0 <= img_idx_y < image.shape[1]:
                            image_value = image[img_idx_x, img_idx_y]
                        else:
                            image_value = 0
                    ### END REGION SOLUTION

                    case _:  # default to wrapping if we do not know the border behaviour
                        img_idx_x %= image.shape[0]
                        img_idx_y %= image.shape[1]
                        image_value = image[img_idx_x, img_idx_y]

                subimage[sub_idx_x, sub_idx_y] = image_value

        return subimage

    @abstractmethod
    def _apply_filter_func(self, selected_pixels: np.ndarray):
        pass


class BaseKernelFilter2D(BaseFilter2D):
    def __init__(self, dim_x: int, dim_y: int | None = None):
        super().__init__(dim_x, dim_y)

        self.kernel = self._create_kernel()

    @abstractmethod
    def _create_kernel(self) -> np.ndarray:
        pass
