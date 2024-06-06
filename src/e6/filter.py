import numpy as np
from pathlib import Path

from common.read_dicom_data import read_dicom_data
from e6.base_filter import BaseFilter2D, BaseKernelFilter2D, IFilter2D
from e6.ui import show_window


# region solution
class MedianFilter(BaseFilter2D):
    def _apply_filter_func(self, selected_pixels: np.ndarray):
        return np.median(selected_pixels)


class GaussFilter(BaseKernelFilter2D):
    def __init__(self, dim_x: int, dim_y: int | None = None, sigma: float = 1.0):
        self._sigma = sigma

        super().__init__(dim_x, dim_y)

    def _create_kernel(self) -> np.ndarray:
        kernel = np.zeros(self.shape, dtype=float)

        # Populate the kernel
        for i in range(self.get_x_dim):
            for k in range(self.get_y_dim):
                x = i - self.get_x_dim // 2
                y = k - self.get_y_dim // 2

                kernel[i, k] = np.exp(-(x**2 + y**2) / (2 * self._sigma**2))

        # Normalize
        kernel /= 2 * np.pi * self._sigma**2
        kernel /= kernel.sum()

        return kernel


class SobelFilter(BaseKernelFilter2D):

    def _apply_filter_func(self, selected_pixels: np.ndarray):
        # Apply x sobel first
        x_filter = np.sum(self.kernel[0] * selected_pixels, axis=(0, 1))
        y_filter = np.sum(self.kernel[1] * selected_pixels, axis=(0, 1))

        return (x_filter + y_filter) / 2

    def _create_kernel(self):

        x = self.get_x_dim // 2
        y = self.get_y_dim // 2

        x_kernel = np.zeros(self.shape, dtype=int)
        y_kernel = np.zeros(self.shape, dtype=int)

        for i in range(self.get_x_dim):
            for k in range(self.get_y_dim):
                x_kernel[i, k] = (k - x) * (y - abs(i - y))
                y_kernel[i, k] = (i - y) * (x - abs(k - x))

        return x_kernel, y_kernel


class XSobelFilter(SobelFilter):

    def _apply_filter_func(self, selected_pixels: np.ndarray):
        return np.sum(self.kernel[0] * selected_pixels, axis=(0, 1))


class YSobelFilter(SobelFilter):

    def _apply_filter_func(self, selected_pixels: np.ndarray):
        return np.sum(self.kernel[1] * selected_pixels, axis=(0, 1))


class ErosionFilter(BaseFilter2D):
    def _apply_filter_func(self, selected_pixels: np.ndarray):
        return selected_pixels.min()


class DilationFilter(BaseFilter2D):
    def _apply_filter_func(self, selected_pixels: np.ndarray):
        return selected_pixels.max()


class OpeningFilter(BaseFilter2D):

    def __init__(self, dim_x: int, dim_y: int | None = None) -> None:
        self._erosion_filter = ErosionFilter(dim_x, dim_y)
        self._dilation_filter = ErosionFilter(dim_x, dim_y)

        self._cached_image = None

    def __call__(
        self,
        image: np.ndarray,
        force_refilter: bool = False,
        border_behaviour="wrap",
        **kwargs,
    ):
        return self.filter(image, force_refilter, border_behaviour, **kwargs)

    def filter(
        self,
        image: np.ndarray,
        force_refilter: bool = False,
        border_behaviour="wrap",
        **kwargs,
    ) -> np.ndarray:
        if self._cached_image is not None and not force_refilter:
            return self._cached_image

        temp = self._erosion_filter(image, force_refilter, border_behaviour, **kwargs)
        result = self._dilation_filter(temp, force_refilter, border_behaviour, **kwargs)

        self._cached_image = result

        return result

    def _apply_filter_func(self, selected_pixels: np.ndarray):
        pass


class ClosingFilter(BaseFilter2D):

    def __init__(self, dim_x: int, dim_y: int | None = None) -> None:
        self._erosion_filter = ErosionFilter(dim_x, dim_y)
        self._dilation_filter = ErosionFilter(dim_x, dim_y)

        self._cached_image = None

    def __call__(
        self,
        image: np.ndarray,
        force_refilter: bool = False,
        border_behaviour="wrap",
        **kwargs,
    ):
        return self.filter(image, force_refilter, border_behaviour, **kwargs)

    def filter(
        self,
        image: np.ndarray,
        force_refilter: bool = False,
        border_behaviour="wrap",
        **kwargs,
    ) -> np.ndarray:
        if self._cached_image is not None and not force_refilter:
            return self._cached_image

        temp = self._dilation_filter(image, force_refilter, border_behaviour, **kwargs)
        result = self._erosion_filter(temp, force_refilter, border_behaviour, **kwargs)

        self._cached_image = result

        return result

    def _apply_filter_func(self, selected_pixels: np.ndarray):
        pass


# end region solution


def _main():
    # region solution
    kernel_size = 3

    gauss_filter = GaussFilter(kernel_size)
    median_filter = MedianFilter(kernel_size)
    sobel_filter = SobelFilter(kernel_size)
    x_sobel_filter = XSobelFilter(kernel_size)
    y_sobel_filter = YSobelFilter(kernel_size)
    min_filter = ErosionFilter(kernel_size)
    max_filter = DilationFilter(kernel_size)

    opening_filter = OpeningFilter(kernel_size)
    closing_filter = ClosingFilter(kernel_size)
    # end region

    image_data = read_dicom_data(Path("data/series/"))

    show_window(
        [
            # add your filter instances here!
            # E.g.:
            # GaussFilter(3),
            # MedianFilter(3)
            # region solution
            gauss_filter,
            median_filter,
            sobel_filter,
            x_sobel_filter,
            y_sobel_filter,
            min_filter,
            max_filter,
            opening_filter,
            closing_filter,
            # end region solution
        ],
        image_data,
    )


if __name__ == "__main__":
    _main()
