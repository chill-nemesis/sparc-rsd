import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QTextEdit,
    QCheckBox,
)
from PySide6.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from e4._module_loader import MagneticDipole


PLOT_RANGE = 5


class MagneticDipoleSimulator(QWidget):
    def __init__(
        self,
        dipole: MagneticDipole = MagneticDipole(),
        simulation_frequency_hz=30,
    ):
        super().__init__()
        self.setWindowTitle("Magnetic Dipole Simulator")

        self.dipole = dipole
        self.t = 0

        self._show_b_field = True

        self._create_ui()

        self._simulation_timer = QTimer()
        self._simulation_timer.timeout.connect(lambda: self.simulation_step(1 / simulation_frequency_hz))
        self._simulation_timer.start(1 / simulation_frequency_hz * 1000)  # Convert to milliseconds

    def _create_ui(self):
        layout = QVBoxLayout()

        # B field update function input
        layout.addWidget(QLabel("Magnetic Field Function B(pos, t): (pos=[x,y,z], return [Bx, By, Bz])"))
        b_field_text = QTextEdit()
        b_field_text.setPlainText("return np.sin(t / 10 * 2 * np.pi) * (2000000*pos + [0, 0, 1/200000])")
        self._update_b_field(b_field_text.toPlainText())
        layout.addWidget(b_field_text)

        hbox = QHBoxLayout()

        update_b_field = QPushButton("Update B field")
        update_b_field.clicked.connect(lambda: self._update_b_field(b_field_text.toPlainText()))
        hbox.addWidget(update_b_field)

        reset_simulation = QPushButton("Reset Simulation")
        reset_simulation.clicked.connect(self.reset_simulation)
        hbox.addWidget(reset_simulation)

        # Checkbox to normalize the plot arrows
        show_b_field = QCheckBox("Show B field")
        show_b_field.setChecked(True)
        show_b_field.toggled.connect(lambda checked: setattr(self, "_show_b_field", checked))
        hbox.addWidget(show_b_field)

        layout.addLayout(hbox)

        # 3D Plot setup
        self.fig = plt.figure()
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas, stretch=1)

        self.ax = self.fig.add_subplot(111, projection="3d")

        # Set the main layout
        self.setLayout(layout)

    def _update_ui(self, B, T, F):
        self.ax.clear()

        self.ax.set_xlim([-PLOT_RANGE, PLOT_RANGE])
        self.ax.set_ylim([-PLOT_RANGE, PLOT_RANGE])
        self.ax.set_zlim([-PLOT_RANGE, PLOT_RANGE])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.ax.quiver(
            *self.dipole.position,
            *self.dipole.global_magnetization,
            color="g",
            label="Magnetization M",
            normalize=True,
        )
        self.ax.quiver(
            *self.dipole.position,
            *B,
            color="b",
            label="Electromagnetic Field B",
            normalize=True,
        )
        self.ax.quiver(
            *self.dipole.position,
            *T,
            color="r",
            label="Torque axis",
            normalize=True,
        )
        self.ax.quiver(
            *self.dipole.position,
            *F,
            color="orange",
            label="Force axis",
            normalize=True,
        )

        if self._show_b_field:
            # create magnetic field quiver plot
            num_evals = 5
            xs, ys, zs = np.meshgrid(
                np.linspace(-PLOT_RANGE, PLOT_RANGE, num_evals),
                np.linspace(-PLOT_RANGE, PLOT_RANGE, num_evals),
                np.linspace(-PLOT_RANGE, PLOT_RANGE, num_evals),
            )

            # B is a list of the B field evalutated at each point of the mesh grid
            Bs = np.array(
                [
                    self._b_field_func(np.asarray([x, y, z]), self.t)
                    for x, y, z in zip(xs.flatten(), ys.flatten(), zs.flatten())
                ]
            )

            self.ax.quiver(
                xs,
                ys,
                zs,
                Bs[:, 0].reshape(xs.shape),
                Bs[:, 1].reshape(xs.shape),
                Bs[:, 2].reshape(xs.shape),
                color="lightblue",
                label="Electromagnetic Field B",
                normalize=True,
                length=1,
            )

        self.ax.legend()
        self.canvas.draw()

    def _update_b_field(self, new_text):
        # Update the B field function
        self._b_field_text = new_text  #
        local_vars = {"np": np}
        exec(
            "def B_func(pos, t):\n    " + "\n    ".join("    " + line for line in self._b_field_text.splitlines()),
            globals(),
            local_vars,
        )
        self._b_field_func = local_vars["B_func"]

    @property
    def b_field_func(self):
        return self._b_field_func

    def get_local_B(self, t):
        return self.b_field_func(self.dipole.position, t)

    def get_grad_B(self, t):
        def _numerical_jacobian(f, x, eps=1e-6):
            f0 = f(x)
            jac = np.zeros((len(f0), len(x)))
            for i in range(len(x)):
                x1 = np.array(x)
                x1[i] += eps
                f1 = f(x1)
                jac[:, i] = (f1 - f0) / eps
            return jac

        # Compute the gradient of the B field
        def B_vectorized(x):
            return np.array(self.b_field_func(x, t))

        return _numerical_jacobian(B_vectorized, self.dipole.position)

    def reset_simulation(self):
        self.dipole.reset()

        self.t = 0

    def simulation_step(self, dt):

        # Get the local B field and its gradient
        local_B = self.get_local_B(self.t)
        grad_B = self.get_grad_B(self.t)

        # Update the dipole's position and orientation
        T, F = self.dipole.update(local_B, grad_B, dt)

        self._update_ui(local_B, T, F)

        self.t += dt


def _main():
    app = QApplication(sys.argv)
    sim = MagneticDipoleSimulator()
    sim.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    _main()
