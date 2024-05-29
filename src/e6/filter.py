from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from common.read_dicom_data import read_dicom_data
from e6.base_filter import BaseFilter2D, BaseKernelFilter2D


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


def _main():
    min_filter = MinIntensityFilter(7)
    max_filter = MaxIntensityFilter(7)
    gauss_filter = GaussFilter(7)
    image_data = read_dicom_data(Path("data/e5/series/"))

    # image_data[0] = np.array(
    #     [
    #         [0, 0, 0, 0, 0],
    #         [0, 0, 1, 1, 0],
    #         [0, 1, 1, 1, 0],
    #         [0, 1, 1, 1, 0],
    #         [0, 0, 0, 0, 0],
    #     ]
    # )

    min_img = min_filter(image_data[0])
    max_img = max_filter(image_data[0])
    gauss_img = gauss_filter(image_data[0])

    normalized_cmap = Normalize(vmin=0, vmax=4000)

    fig, axis = plt.subplots(1, 4)
    axes = axis.flatten()

    axes[0].imshow(
        image_data[0],
        cmap="gray",
        norm=normalized_cmap,
    )
    axes[0].set_title("Input image")

    axes[1].imshow(
        min_img,
        cmap="gray",
        norm=normalized_cmap,
    )
    axes[1].set_title("Min image")

    axes[2].imshow(
        max_img,
        cmap="gray",
        norm=normalized_cmap,
    )
    axes[2].set_title("Max image")

    axes[3].imshow(
        gauss_img,
        cmap="gray",
        norm=normalized_cmap,
    )
    axes[3].set_title("Gaussed image")

    plt.show()


if __name__ == "__main__":
    _main()
