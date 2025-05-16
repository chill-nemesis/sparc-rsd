import sys
import numpy as np
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QPushButton,
    QHBoxLayout,
    QDoubleSpinBox,
    QTextEdit,
)
from PySide6.QtGui import QColor
from PySide6.QtCore import QTimer
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from mpl_toolkits.mplot3d import Axes3D


class MagneticDipoleSimulator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Magnetic Dipole Simulator")
        self.resize(600, 600)

        layout = QVBoxLayout()

        # Volume input
        self.volume_input = QDoubleSpinBox()
        # self.volume_input.setRange(1e-10, 1e-6)
        self.volume_input.setDecimals(10)
        self.volume_input.setValue(1e-9)
        layout.addWidget(QLabel("Volume (m^3):"))
        layout.addWidget(self.volume_input)

        # Magnetization vector input
        self.m_input = [QDoubleSpinBox() for _ in range(3)]
        m_layout = QHBoxLayout()
        for i, spin in enumerate(self.m_input):
            spin.setRange(-1e6, 1e6)
            spin.setValue(1e5 if i == 0 else 0)
            m_layout.addWidget(spin)
        layout.addWidget(QLabel("Magnetization M (A/m):"))
        layout.addLayout(m_layout)

        # B field update function input
        layout.addWidget(
            QLabel(
                "Magnetic Field Update Function B(t): (in terms of t, return a list [Bx, By, Bz])"
            )
        )
        self.b_field_text = QTextEdit()
        self.b_field_text.setPlainText("return [0, 0, 0.005]  # Constant field")
        layout.addWidget(self.b_field_text)

        # Output and simulate button
        self.result_label = QLabel("Torque T: [0, 0, 0]")
        layout.addWidget(self.result_label)

        self.simulate_btn = QPushButton("Run Simulation")
        self.simulate_btn.clicked.connect(self.run_simulation)
        layout.addWidget(self.simulate_btn)

        # 3D Plot setup
        self.fig = plt.figure(figsize=(5, 4))
        self.canvas = FigureCanvas(self.fig)
        layout.addWidget(self.canvas)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlim([-1, 1])
        self.ax.set_ylim([-1, 1])
        self.ax.set_zlim([-1, 1])
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")

        self.setLayout(layout)

        # Timer for updates
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_simulation)
        self.t = 0

    def run_simulation(self):
        self.t = 0
        self.timer.start(100)  # Update every 100 ms

    def update_simulation(self):
        try:
            nu = self.volume_input.value()
            M = np.array([spin.value() for spin in self.m_input])

            # Evaluate B field
            b_code = self.b_field_text.toPlainText()
            local_vars = {"t": self.t, "np": np}
            exec(
                "def B_func(t):\n    "
                + "\n    ".join("    " + line for line in b_code.splitlines()),
                globals(),
                local_vars,
            )
            B = np.array(local_vars["B_func"](self.t))

            T = nu * np.cross(M, B)
            self.result_label.setText(f"Torque T: {T.tolist()}")

            # Update 3D plot
            self.ax.clear()
            self.ax.set_xlim([-1, 1])
            self.ax.set_ylim([-1, 1])
            self.ax.set_zlim([-1, 1])
            self.ax.set_xlabel("X")
            self.ax.set_ylabel("Y")
            self.ax.set_zlabel("Z")
            self.ax.quiver(0, 0, 0, *M, color="b", label="M")
            self.ax.quiver(0, 0, 0, *B, color="g", label="B")
            if np.linalg.norm(T) > 0:
                self.ax.quiver(0, 0, 0, *T, color="r", label="T")
            self.ax.legend()
            self.canvas.draw()

            self.t += 0.1
        except Exception as e:
            self.result_label.setText(f"Error: {str(e)}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    sim = MagneticDipoleSimulator()
    sim.show()
    sys.exit(app.exec())
