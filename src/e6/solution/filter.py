import numpy as np


from e6.solution.base_filter import BaseFilter2D, BaseKernelFilter2D, IFilter2D


def get_filters() -> list[IFilter2D]:
    kernel_size = 9
    
    return [
        MotionBlurFilterX(kernel_size),
        GaussFilter(kernel_size),
        MedianFilter(kernel_size),
        SobelFilter(3),
        XSobelFilter(3),
        YSobelFilter(3),
        ErosionFilter(kernel_size),
        DilationFilter(kernel_size),
        OpeningFilter(kernel_size),
        ClosingFilter(kernel_size),
    ]


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


class MotionBlurFilterX(BaseKernelFilter2D):
    def _create_kernel(self):
        kernel = np.zeros(self.shape)
        kernel[self.get_x_dim // 2 + 1, :] = 1

        return kernel / kernel.sum()


class SobelFilter(BaseKernelFilter2D):

    def _apply_filter_func(self, selected_pixels: np.ndarray):
        g_x = np.convolve(self.kernel[0].flatten(), selected_pixels.flatten(), mode="valid")[0]
        g_y = np.convolve(self.kernel[1].flatten(), selected_pixels.flatten(), mode="valid")[0]

        return np.sqrt(np.square(g_x) + np.square(g_y))

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
        return np.convolve(self.kernel[0].flatten(), selected_pixels.flatten(), mode="valid")[0]


class YSobelFilter(SobelFilter):

    def _apply_filter_func(self, selected_pixels: np.ndarray):
        return np.convolve(self.kernel[1].flatten(), selected_pixels.flatten(), mode="valid")[0]


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
