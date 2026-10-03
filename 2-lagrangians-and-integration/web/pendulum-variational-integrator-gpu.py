import numpy as np

from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Line, Markers, Text

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer


def pendulum_variational_integrator(theta0, omega0, length, gravity, mass, dt, n_steps):
    theta = np.zeros(n_steps, dtype=np.float32)
    omega = np.zeros(n_steps, dtype=np.float32)
    theta[0] = theta0
    omega[0] = omega0

    for idx in range(n_steps - 1):
        omega[idx + 1] = omega[idx] - (gravity / length) * np.sin(theta[idx]) * dt
        theta[idx + 1] = theta[idx] + omega[idx + 1] * dt

    kinetic = 0.5 * mass * (length * omega) ** 2
    potential = mass * gravity * length * (1.0 - np.cos(theta))
    energy = kinetic + potential
    return theta, omega, energy


def pendulum_positions(theta, length):
    return np.column_stack([
        length * np.sin(theta),
        np.zeros_like(theta),
        -length * np.cos(theta),
    ]).astype(np.float32)


class PlotPanel:
    def __init__(self, view, title, x_label, y_label, line_color, marker_color):
        self.view = view
        self.view.camera = scene.PanZoomCamera()
        self.x_axis = Line(color=(0.82, 0.82, 0.82, 0.55), connect='segments', parent=self.view.scene)
        self.y_axis = Line(color=(0.82, 0.82, 0.82, 0.55), connect='segments', parent=self.view.scene)
        self.line = Line(color=line_color, connect='strip', parent=self.view.scene)
        self.cursor = Line(color=(1, 1, 1, 0.45), connect='segments', parent=self.view.scene)
        self.marker = Markers(parent=self.view.scene)
        self.cursor.set_data(pos=np.zeros((2, 3), dtype=np.float32))
        self.cursor.visible = False
        self.marker.set_data(pos=np.zeros((1, 3), dtype=np.float32),
                             face_color=(1, 1, 1, 0.95),
                             edge_color=(0, 0, 0, 0.8), size=0)
        self.title = Text(title, color='white', font_size=11, anchor_x='center',
                          anchor_y='bottom', parent=self.view.scene)
        self.x_label = Text(x_label, color=(0.86, 0.86, 0.86, 1), font_size=9,
                            anchor_x='center', anchor_y='top', parent=self.view.scene)
        self.y_label = Text(y_label, color=(0.86, 0.86, 0.86, 1), font_size=9,
                            anchor_x='center', anchor_y='bottom', rotation=-90,
                            parent=self.view.scene)
        self.x = np.array([], dtype=np.float32)
        self.y = np.array([], dtype=np.float32)
        self.x_range = (0.0, 1.0)
        self.y_range = (0.0, 1.0)
        self.is_phase = False

    def set_data(self, x, y, is_phase=False):
        self.x = np.asarray(x, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.float32)
        self.is_phase = is_phase

        pos = np.column_stack([self.x, self.y, np.zeros_like(self.x)]).astype(np.float32)
        self.line.set_data(pos=pos)

        x_min, x_max = float(np.min(self.x)), float(np.max(self.x))
        y_min, y_max = float(np.min(self.y)), float(np.max(self.y))
        x_pad = max((x_max - x_min) * 0.05, 0.1)
        y_pad = max((y_max - y_min) * 0.10, 0.1)
        self.x_range = (x_min - x_pad, x_max + x_pad)
        self.y_range = (y_min - y_pad, y_max + y_pad)

        self.view.camera.set_range(
            x=self.x_range,
            y=self.y_range,
        )
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        x_axis_y = 0.0 if y0 <= 0.0 <= y1 else y0
        y_axis_x = 0.0 if x0 <= 0.0 <= x1 else x0
        self.x_axis.set_data(pos=np.array([[x0, x_axis_y, 0], [x1, x_axis_y, 0]], dtype=np.float32))
        self.y_axis.set_data(pos=np.array([[y_axis_x, y0, 0], [y_axis_x, y1, 0]], dtype=np.float32))

        width = x1 - x0
        height = y1 - y0
        self.title.pos = ((x0 + x1) * 0.5, y1 - 0.03 * height, 0)
        self.x_label.pos = ((x0 + x1) * 0.5, y0 + 0.02 * height, 0)
        self.y_label.pos = (x0 + 0.025 * width, (y0 + y1) * 0.5, 0)
        self.update_marker(0)

    def update_marker(self, index):
        if len(self.x) == 0:
            return

        index = int(np.clip(index, 0, len(self.x) - 1))
        marker_pos = np.array([[self.x[index], self.y[index], 0]], dtype=np.float32)
        self.marker.set_data(pos=marker_pos, face_color=(1, 1, 1, 0.95),
                             edge_color=(0, 0, 0, 0.8), size=8)

        if self.is_phase:
            self.cursor.visible = False
        else:
            self.cursor.visible = True
            x_now = self.x[index]
            y0, y1 = self.y_range
            self.cursor.set_data(pos=np.array([[x_now, y0, 0], [x_now, y1, 0]], dtype=np.float32))


class MainWindow(QtWidgets.QMainWindow):
    SPEED_SETS = [
        ('rest release: 0.00 rad/s', 0.0),
        ('slow launch: 0.75 rad/s', 0.75),
        ('medium launch: 1.50 rad/s', 1.50),
        ('fast launch: 2.50 rad/s', 2.50),
        ('high-energy launch: 4.00 rad/s', 4.00),
    ]

    def __init__(self):
        super().__init__()
        self.mass = 1.0
        self.length = 1.0
        self.gravity = 9.81
        self.theta0 = 0.75
        self.dt = 0.01
        self.n_steps = 6000
        self.time = np.arange(self.n_steps, dtype=np.float32) * self.dt
        self.current_index = 0
        self.play_stride = 3

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        instructions = QtWidgets.QLabel(
            'Pendulum variational integrator: choose a predefined initial angular speed, then use Play or the time slider '
            'to move through the generated simulation. The left panel is the 3D pendulum; the right panels show angle over '
            'time, total energy over time, and phase space for the selected speed. Mouse wheel zooms the panel under the '
            'cursor, and mouse drag orbits the 3D pendulum view.'
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet('color: #d8d8d8; background: #111; padding: 6px; font-size: 10pt;')
        layout.addWidget(instructions)

        self.canvas = SceneCanvas(title='Pendulum Variational Integrator', size=(1280, 760))
        self.canvas.native.setParent(central)
        layout.addWidget(self.canvas.native)

        self.grid = self.canvas.central_widget.add_grid(spacing=8)
        self.view_pendulum = self.grid.add_view(0, 0, row_span=3)
        self.view_theta = self.grid.add_view(0, 1)
        self.view_energy = self.grid.add_view(1, 1)
        self.view_phase = self.grid.add_view(2, 1)

        self.view_pendulum.camera = scene.TurntableCamera(elevation=18, azimuth=35, distance=3.1)
        self.view_pendulum.camera.set_range(x=(-1.25, 1.25), y=(-1.25, 1.25), z=(-1.25, 0.25))

        self.title_pendulum = Text('3D pendulum', color='white', font_size=14, anchor_x='center',
                                   anchor_y='bottom', parent=self.view_pendulum.scene)
        self.title_pendulum.pos = (0, 0, 0.2)

        self.pivot = np.array([0, 0, 0], dtype=np.float32)
        self.rod = Line(color=(0.82, 0.88, 1.0, 0.95), width=2, connect='segments',
                        parent=self.view_pendulum.scene)
        self.rest_line = Line(pos=np.array([[0, 0, 0], [0, 0, -self.length]], dtype=np.float32),
                              color=(0.5, 0.5, 0.5, 0.35), connect='segments',
                              parent=self.view_pendulum.scene)
        self.arc_trace = Line(color=(0.2, 0.95, 1.0, 0.45), connect='strip',
                              parent=self.view_pendulum.scene)
        self.pivot_marker = Markers(parent=self.view_pendulum.scene)
        self.bob_marker = Markers(parent=self.view_pendulum.scene)
        self.pivot_marker.set_data(pos=np.array([[0, 0, 0]], dtype=np.float32),
                                   face_color=(1, 1, 1, 1), edge_color=(0, 0, 0, 1), size=9)

        self.theta_plot = PlotPanel(self.view_theta, 'Angle theta(t)', 'Time (s)', 'Angle theta(t)',
                                    (0.2, 0.55, 1.0, 0.95), (1, 1, 1, 1))
        self.energy_plot = PlotPanel(self.view_energy, 'Total energy E(t)', 'Time (s)', 'Total energy',
                                     (1.0, 0.2, 0.3, 0.95), (1, 1, 1, 1))
        self.phase_plot = PlotPanel(self.view_phase, 'Phase space: theta vs omega',
                                    'Angle theta', 'Angular velocity omega',
                                    (0.2, 0.9, 0.35, 0.85), (1, 1, 1, 1))

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        layout.addWidget(controls)

        controls_layout.addWidget(QtWidgets.QLabel('Initial angular speed:'))
        self.speed_combo = QtWidgets.QComboBox()
        for label, _ in self.SPEED_SETS:
            self.speed_combo.addItem(label)
        self.speed_combo.currentIndexChanged.connect(self.on_speed_changed)
        controls_layout.addWidget(self.speed_combo)

        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        reset_button = QtWidgets.QPushButton('Reset')
        reset_button.clicked.connect(self.reset_animation)
        controls_layout.addWidget(reset_button)

        self.time_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.setMaximum(self.n_steps - 1)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        controls_layout.addWidget(self.time_slider)

        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setMinimumWidth(360)
        controls_layout.addWidget(self.stats_label)

        self.timer = QTimer()
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.advance_frame)

        self.status_bar = self.statusBar()
        self.simulate_selected_speed()
        self.update_frame(0)

    def simulate_selected_speed(self):
        _, omega0 = self.SPEED_SETS[self.speed_combo.currentIndex()]
        self.theta, self.omega, self.energy = pendulum_variational_integrator(
            self.theta0, omega0, self.length, self.gravity, self.mass, self.dt, self.n_steps
        )
        self.positions = pendulum_positions(self.theta, self.length)

        self.theta_plot.set_data(self.time, self.theta)
        self.energy_plot.set_data(self.time, self.energy)
        self.phase_plot.set_data(self.theta, self.omega, is_phase=True)

        energy_drift = (self.energy[-1] - self.energy[0]) / self.energy[0] * 100.0
        self.stats_label.setText(f'Energy drift: {energy_drift:+.3e}%')
        self.status_bar.showMessage(
            f'Speed set changed: omega0={omega0:.2f} rad/s, initial energy={self.energy[0]:.4f}'
        )

    def on_speed_changed(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.current_index = 0
        self.simulate_selected_speed()
        self.time_slider.blockSignals(True)
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)
        self.update_frame(0)

    def on_time_slider_changed(self, value):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(value)

    def reset_animation(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(0)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play')
        else:
            if self.current_index >= self.n_steps - 1:
                self.update_frame(0)
            self.timer.start()
            self.play_button.setText('Pause')

    def advance_frame(self):
        next_index = min(self.current_index + self.play_stride, self.n_steps - 1)
        self.update_frame(next_index)
        if next_index >= self.n_steps - 1:
            self.timer.stop()
            self.play_button.setText('Play')

    def update_frame(self, index):
        self.current_index = int(np.clip(index, 0, self.n_steps - 1))
        bob = self.positions[self.current_index]
        self.rod.set_data(pos=np.array([self.pivot, bob], dtype=np.float32))
        self.bob_marker.set_data(pos=np.array([bob], dtype=np.float32),
                                 face_color=(1.0, 0.74, 0.18, 1.0),
                                 edge_color=(0, 0, 0, 1), size=18)

        trace_start = max(0, self.current_index - 500)
        self.arc_trace.set_data(pos=self.positions[trace_start:self.current_index + 1])

        self.theta_plot.update_marker(self.current_index)
        self.energy_plot.update_marker(self.current_index)
        self.phase_plot.update_marker(self.current_index)

        self.time_slider.blockSignals(True)
        self.time_slider.setValue(self.current_index)
        self.time_slider.blockSignals(False)

        self.setWindowTitle(
            f'Pendulum variational integrator | t={self.time[self.current_index]:.2f}s '
            f'| theta={self.theta[self.current_index]:+.3f} rad '
            f'| omega={self.omega[self.current_index]:+.3f} rad/s'
        )
        self.canvas.update()


if __name__ == '__main__':
    qt_app = QtWidgets.QApplication([])
    app.use_app('pyqt5')
    window = MainWindow()
    window.show()
    qt_app.exec_()
