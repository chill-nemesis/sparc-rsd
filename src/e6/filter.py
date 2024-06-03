import numpy as np

from common.read_dicom_data import read_dicom_data
from e6.base_filter import BaseFilter2D, BaseKernelFilter2D
from e6.ui import show_window


class MinIntensityFilter(BaseFilter2D):
    def _apply_filter_func(self, selected_pixels: np.ndarray):
        return selected_pixels.min()


class MaxIntensityFilter(BaseFilter2D):
    def _apply_filter_func(self, selected_pixels: np.ndarray):
        return selected_pixels.max()


class GaussFilter(BaseKernelFilter2D):
    def __init__(self, dim_x: int, dim_y: int | None = None, sigma: float = 1.0):
        self._sigma = sigma

        super().__init__(dim_x, dim_y)

    def _apply_filter_func(self, selected_pixels: np.ndarray):
        return np.sum(self.kernel * selected_pixels, axis=(0, 1))

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


class MedianFilter(BaseFilter2D):
    def _apply_filter_func(self, selected_pixels: np.ndarray):
        return np.median(selected_pixels)


def _main():
    kernel_size = 3
    min_filter = MinIntensityFilter(kernel_size)
    max_filter = MaxIntensityFilter(kernel_size)
    gauss_filter = GaussFilter(kernel_size)
    median_filter = MedianFilter(kernel_size)
    # image_data = read_dicom_data(Path("data/e5/series/"))

    image_data = [[None]]

    image_data[0] = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 3000, 3000, 0],
            [0, 3000, 3000, 3000, 0],
            [0, 3000, 3000, 3000, 0],
            [0, 0, 0, 0, 0],
        ]
    )

    show_window(
        [
            min_filter,
            max_filter,
            gauss_filter,
            median_filter,
        ],
        image_data[0],
    )


if __name__ == "__main__":
    _main()
