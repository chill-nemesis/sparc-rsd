from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

from PySide6.QtCore import Signal


class MPLCanvas(FigureCanvas):
    plot_value = Signal(int, int, float)

    def __init__(self, figure=None):
        super().__init__(figure)

        self.mpl_connect("motion_notify_event", self._on_mouse_move)

    def _on_mouse_move(self, event):
        if event.inaxes:
            for ax in event.canvas.figure.axes:
                for image in ax.get_images():
                    x, y = round(event.xdata), round(event.ydata)
                    # Get the data array from the image
                    data = image.get_array().data
                    # Ensure the coordinates are within the bounds of the data array
                    if 0 <= x < data.shape[1] and 0 <= y < data.shape[0]:
                        pixel_value = data[y, x]
                        self.plot_value.emit(x, y, pixel_value)
                        return
