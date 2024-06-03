import sys

from PySide6.QtWidgets import QMainWindow, QApplication, QWidget, QVBoxLayout, QComboBox, QStatusBar
from PySide6.QtCore import Slot

from common.matplotlib_canvas import MPLCanvas
from common.fancy_colormap import get_fancy_colormap

from e6.base_filter import BaseFilter2D


import matplotlib.pyplot as plt
from matplotlib.colors import Normalize


import numpy as np


class E6Window(QMainWindow):
    def __init__(self, filter_list: list[BaseFilter2D], image: np.ndarray) -> None:
        super().__init__()

        self._image = image
        self._filters = filter_list

        self.setWindowTitle("RSD: Exercise 6 - Image processing")

        self._central_widget = QWidget()
        self.setCentralWidget(self._central_widget)
        self._layout = QVBoxLayout(self._central_widget)

        self._selector = QComboBox(self)
        self._selector.addItems([type(x).__name__ for x in self._filters])
        self._selector.currentIndexChanged.connect(self.update_plot)
        self._layout.addWidget(self._selector)

        self.create_plot()
        self._canvas = MPLCanvas(self._fig)
        self._layout.addWidget(self._canvas)

        status_bar = QStatusBar()
        self.setStatusBar(status_bar)
        self._canvas.plot_value.connect(self._update_status_message)

    @Slot(int, int, float)
    def _update_status_message(self, x: int, y: int, value: float):
        self.statusBar().showMessage(f"{x}, {y}:\t{value}")

    def create_plot(self):
        self._fig, axis = plt.subplots(1, 2)
        axes = axis.flatten()

        normalized_cmap = Normalize(vmin=0, vmax=4000)
        color_map = get_fancy_colormap()

        self._slice_plot = axes[0].imshow(
            self._image,
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
        self._filtered_plot.set_data(self._get_active_filter()(self._image))
        self._filter_axis.set_title(f"{type(self._get_active_filter()).__name__}ed image")
        self._fig.canvas.draw_idle()

    def _get_filtered_image(self):
        return self._get_active_filter()(self._image)

    def _get_active_filter(self) -> BaseFilter2D:
        return self._filters[self._selector.currentIndex()]


def show_window(filter_list: list[BaseFilter2D], image: np.ndarray):
    app = QApplication(sys.argv)

    window = E6Window(filter_list, image)
    window.show()
    sys.exit(app.exec())
