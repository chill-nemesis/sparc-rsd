from e6.base_filter import IFilter2D


def get_filters() -> list[IFilter2D]:
    raise NotImplementedError("place your implemented filters here")
    return [
        # Add your filter instances here!
        # E.g.:
        # MedianFilter(3),
        # GaussFilter(3, 1.0),
    ]


class MedianFilter(IFilter2D):
    pass


class GaussFilter(IFilter2D):
    pass


class MotionBlurFilterX(IFilter2D):
    pass


class SobelFilter(IFilter2D):
    pass


class XSobelFilter(IFilter2D):
    pass


class YSobelFilter(IFilter2D):
    pass


class ErosionFilter(IFilter2D):
    pass


class DilationFilter(IFilter2D):
    pass


class OpeningFilter(IFilter2D):
    pass


class ClosingFilter(IFilter2D):
    pass
