"""Template for Exercise 5 - Robotics in Medicine."""

from pathlib import Path
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib.colors import Normalize

from common.read_dicom_data import read_dicom_data
from common.fancy_colormap import get_fancy_colormap


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


def _main():
    dicom_data = read_dicom_data(Path("data/e5/series/"))

    print(f"Read {len(dicom_data)} entries")

    xrays = create_xray(dicom_data)

    # color map normalization
    cmap = get_fancy_colormap()
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
