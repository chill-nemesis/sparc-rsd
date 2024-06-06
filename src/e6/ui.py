"""Provides a basic UI window for displaying filtered images."""

import sys

from PySide6.QtWidgets import (
    QMainWindow,
    QApplication,
    QWidget,
    QVBoxLayout,
    QComboBox,
    QStatusBar,
    QSlider,
    QHBoxLayout,
    QLayout,
)
from PySide6.QtCore import Slot, Qt

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import numpy as np

from common.matplotlib_canvas import MPLCanvas
from common.fancy_colormap import get_fancy_colormap

from e6.base_filter import IFilter2D


class E6Window(QMainWindow):
    def __init__(self, filter_list: list[IFilter2D], images: list[np.ndarray]) -> None:
        super().__init__()

        self._images = images
        self._filters = filter_list

        self.setWindowTitle("RSD: Exercise 6 - Image processing")

        _central_widget = QWidget()
        _layout = QVBoxLayout(_central_widget)
        self.setCentralWidget(_central_widget)

        self._filter_selector = QComboBox()
        self._filter_selector.addItems([type(x).__name__ for x in self._filters])
        self._filter_selector.currentIndexChanged.connect(self.update_plot)

        self._cmap_selector = QComboBox()
        self._cmap_selector.addItems(["Grayscale color map", "Fancy color map"])
        self._cmap_selector.setCurrentIndex(1)
        self._cmap_selector.currentIndexChanged.connect(self._update_colormaps)

        # SLIDER MUST BE CREATED BEFORE THE PLOT IS DRAWN!
        self._slider = QSlider()
        self._slider.setMinimum(0)
        self._slider.setMaximum(self.num_images - 1)
        self._slider.setValue(0)
        self._slider.setOrientation(Qt.Orientation.Horizontal)
        # Triggered every time the slider value changes
        self._slider.valueChanged.connect(self._update_preview_plot)
        self._slider.sliderReleased.connect(self._slider_value_changed)

        self.create_plot()
        _canvas = MPLCanvas(self._fig)
        _canvas.plot_value.connect(self._update_status_message)

        temp = QWidget()
        _horizontal_layout = QHBoxLayout(temp)
        _horizontal_layout.addWidget(self._filter_selector)
        _horizontal_layout.addWidget(self._cmap_selector)
        _horizontal_layout.setSizeConstraint(QLayout.SizeConstraint.SetFixedSize)
        _layout.addWidget(temp)
        _layout.addWidget(_canvas)
        _layout.addWidget(self._slider)

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)

    @Slot(int, int, float)
    def _update_status_message(self, x: int, y: int, value: float):
        self.statusBar().showMessage(f"{x}, {y}:\t{value}")

    @Slot()
    def _slider_value_changed(self):
        for f in self._filters:
            f.invalidate_cache()

        self.update_plot(0)

    @Slot(int)
    def _update_preview_plot(self, _):
        self._slice_plot.set_data(self.active_image)
        self._slice_plot.set_cmap(self._get_colormap())
        self._fig.canvas.draw_idle()

    @Slot(int)
    def _update_colormaps(self, index):
        self._update_preview_plot(index)
        self.update_plot(index)

    @property
    def num_images(self) -> int:
        return len(self._images)

    @property
    def active_image(self) -> np.ndarray:
        return self._images[self._slider.value()]

    def create_plot(self):
        self._fig, axis = plt.subplots(1, 2)
        axes = axis.flatten()

        normalized_cmap = Normalize(vmin=0, vmax=4000)
        color_map = self._get_colormap()

        self._slice_plot = axes[0].imshow(
            self.active_image,
            cmap=color_map,
            norm=normalized_cmap,
        )
        axes[0].set_title("Input image")

        self._filtered_plot = axes[1].imshow(
            self._get_filtered_image(),
            cmap=color_map,
            norm=normalized_cmap,
        )
        self._filter_axis = axes[1]
        self._filter_axis.set_title(f"{type(self._get_active_filter()).__name__}ed image")

    def update_plot(self, _: int):
        self._filtered_plot.set_data(self._get_active_filter()(self.active_image))
        self._filter_axis.set_title(f"{type(self._get_active_filter()).__name__}ed image")
        self._filtered_plot.set_cmap(self._get_colormap())
        self._fig.canvas.draw_idle()

    def _get_filtered_image(self):
        return self._get_active_filter()(self.active_image)

    def _get_active_filter(self) -> IFilter2D:
        return self._filters[self._filter_selector.currentIndex()]

    def _get_colormap(self):
        if self._cmap_selector.currentIndex() == 0:
            return "gray"
        else:
            return get_fancy_colormap()


def show_window(filter_list: list[IFilter2D], images: list[np.ndarray]):
    app = QApplication(sys.argv)

    window = E6Window(filter_list, images)
    window.show()
    sys.exit(app.exec())
