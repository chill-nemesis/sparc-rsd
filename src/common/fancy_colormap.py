import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba, rgb_to_hsv, hsv_to_rgb, ListedColormap
import numpy as np


def _get_fancy_colormap_solution():
    colormap_data = plt.cm.gray(np.linspace(0, 1, 4000))

    # modify the colormap here
    # The indices equal the HU+1000
    # colormap_data[0:100, :] = to_rgba("lightblue")

    colormap_data = _map_color(colormap_data, "lightblue", 0, 100)
    colormap_data = _map_color(colormap_data, "blue", 100, 500)
    colormap_data = _map_color(colormap_data, "lightyellow", 1300, 3000)
    colormap_data = _map_color(colormap_data, "red", 3000, 4000)
    colormap_data = _map_color(colormap_data, "purple", 1050, 1250)

    return ListedColormap(colormap_data)


def get_fancy_colormap():
    colormap_data = plt.cm.gray(np.linspace(0, 1, 4000))

    # modify the colormap here
    # The indices equal the HU+1000
    # colormap_data[0:100, :] = to_rgba("lightblue")


    return ListedColormap(colormap_data)


def _map_color(
    colormap,
    base_color,
    start,
    end,
    value_start: float = 0.5,
    value_end: float = 1,
):
    """Creates a more "fancy" color map - more depth and shades for better visualization.
    The "fancy" colors are achieved by converting the given rgb color to hsv and
    creating a linear interpolation of the saturation and value across the colormap range.

    Args:
        colormap: The colormap array data to modify.
        base_color: The base rgb color. Can also be a matplotlib color string.
        start: The start index in the colormap data for the improved colors.
        end: The end index in the colormap data for improved colors.
        value_start (float, optional): Start value for color value and saturation (at start). Defaults to 0.5.
        value_end (float, optional): End value for color value and saturation (at end). Defaults to 1.

    Returns:
        The updated colormap
    """
    rgb = to_rgba(base_color)[:3]
    hsv = rgb_to_hsv(rgb)

    num_colors = end - start
    values = np.linspace(value_start, value_end, num_colors)

    hsv_colors = np.zeros((num_colors, 3))
    hsv_colors[:, 0] = hsv[0]  # hue
    hsv_colors[:, 1] = values  # saturation
    hsv_colors[:, 2] = values  # value

    rgba_colors = [(*hsv_to_rgb(c), 1) for c in hsv_colors]

    colormap[start:end, :] = rgba_colors

    return colormap
