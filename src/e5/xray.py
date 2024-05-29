"""Template for Exercise 5 - Robotics in Medicine."""

import pickle
import hashlib
from pathlib import Path
import pydicom
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import to_rgba, ListedColormap, Normalize, rgb_to_hsv, hsv_to_rgb


def _get_dicom_pixel(file_path: Path):
    """Load a single file of dicom pixel data."""
    print(f"Loading file {file_path}")
    dicom_file = pydicom.read_file(file_path)
    return dicom_file.pixel_array


def _read_dicom_data(path: Path, force_reload: bool = False):
    """Read the dicom content of a file or directory.
    This also creates a pickled cache for the directory for faster reloading.

    Args:
        path (Path): The file or directory to load.
        force_reload (bool, optional): If true, loads the given directory regardless if a cache exists. Defaults to False.

    Returns:
        A list of dicom image sequences.
    """
    # see if we already have a hash for faster loading
    # we need to hash the path string, as that is consistent.
    # just using the path object would result in a new object each time we execute this method
    sha256_file = hashlib.sha256()
    sha256_file.update(str(path).encode("utf-8"))
    hashed_path = Path(sha256_file.hexdigest())
    if not force_reload:
        # a pickled object is available, and we do not want to force reload
        if hashed_path.is_file():
            return pickle.load(open(hashed_path, "rb"))

    # Silently accept a single file, too
    if path.is_file():
        return [_get_dicom_pixel(path)]

    if not path.is_dir:
        raise NotADirectoryError("Path is not a file or directory!")

    result = []

    # get all sub entries in a dir.
    # this can be dirs, files or other things
    # Note: This list is NOT sorted!
    files = list(path.iterdir())
    # select only files, and sort it alphabetically
    files = sorted([file for file in files if file.is_file()])

    result = [_get_dicom_pixel(file) for file in files]

    # Create a hash in case we want to load the dataset later again
    pickle.dump(result, open(hashed_path, "wb"))

    return result

### REGION SOLUTION
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
### END REGION SOLUTION

def create_xray(dicom_data: list[np.ndarray]):
    """Create "xray" images for a dicom sequence.

    Args:
        dicom_data (list[np.ndarray]): The input dicom sequence.

    Returns:
        A tuple containing
            (the maximum intensity image,
            the minimum intensity image,
            average intensity image).
    """
    # Create an ("empty") result array
    # Each element is initially filled with Air (HU == -1000)
    # However, the dicom data we are using is shifted, so 0 is HU == -1000
    max_image = np.zeros(dicom_data[0].shape)
    # Just use a really large number that should not occur in the input data
    min_image = np.ones(dicom_data[0].shape) * 10000
    sum_image = np.zeros(dicom_data[0].shape)

    ##################
    # YOUR CODE HERE #
    ##################
    ### REGION SOLUTION
    for dicom_slice in dicom_data:
        min_image = np.minimum(min_image, dicom_slice)
        max_image = np.maximum(max_image, dicom_slice)
        sum_image += dicom_slice
    ### END REGION SOLUTION

    min_val = np.min(min_image)
    max_val = np.max(max_image)

    print(f"Min: {min_val}\tMax: {max_val}")

    return [max_image, min_image, sum_image / len(dicom_data)]


def get_colormap():
    """Get the colormap for drawing xray images."""
    # We use the basic "gray" colormap
    colormap_data = plt.cm.gray(np.linspace(0, 1, 4000))

    # modify the colormap here
    # The indices equal the HU+1000
    # colormap_data[0:100, :] = to_rgba("lightblue")
    ### REGION SOLUTION
    colormap_data = _map_color(colormap_data, "lightblue", 0, 100)
    colormap_data = _map_color(colormap_data, "blue", 100, 500)
    colormap_data = _map_color(colormap_data, "lightyellow", 1300, 3000)
    colormap_data = _map_color(colormap_data, "red", 3000, 4000)
    colormap_data = _map_color(colormap_data, "purple", 1050, 1250)
    ### END REGION SOLUTION

    return ListedColormap(colormap_data)


def _main():
    dicom_data = _read_dicom_data(Path("data/e5/series/"))

    print(f"Read {len(dicom_data)} entries")

    xrays = create_xray(dicom_data)

    # color map normalization
    cmap = get_colormap()
    normalized_cmap = Normalize(vmin=0, vmax=4000)
    # create the figure
    fig, axis = plt.subplots(1, 4)
    axes = axis.flatten()

    # draw the slice plot
    slice_plot = axes[0].imshow(
        dicom_data[0],
        cmap=cmap,
        norm=normalized_cmap,
    )
    axes[0].axis("off")
    axes[0].set_title("slice")

    # create the slice slider
    slider_axis = plt.axes((0.1, 0.1, 0.8, 0.03))
    slice_slider = Slider(
        ax=slider_axis,
        label="Slice",
        valmin=0,
        valmax=len(dicom_data) - 1,
        valinit=0,
        valstep=1,
    )

    # update function for the slider
    def slider_update(val):
        slice_plot.set_data(dicom_data[val])
        fig.canvas.draw_idle()

    slice_slider.on_changed(slider_update)

    # draw the rest of the unicorn
    for i in range(1, 4):
        axes[i].imshow(
            xrays[i - 1],
            cmap=cmap,
            norm=normalized_cmap,
        )
        axes[i].axis("off")
        axes[i].set_title(f'{["maximum", "minimum", "average"][i-1]}')

    plt.show()


if __name__ == "__main__":
    _main()
