import numpy as np

from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Line, Markers, Text

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer


KICK_COLOR = (1.0, 0.56, 0.12, 1.0)
DRIFT_COLOR = (0.2, 0.78, 1.0, 1.0)
VERLET_COLOR = (0.3, 1.0, 0.55, 0.95)
EULER_COLOR = (1.0, 0.24, 0.18, 0.9)
AXIS_COLOR = (0.78, 0.82, 0.88, 0.55)


def force(x, stiffness):
    return -stiffness * x


def oscillator_energy(x, v, mass, stiffness):
    return 0.5 * mass * v * v + 0.5 * stiffness * x * x


def isolated_vertical_kick(x0, v0, mass, stiffness, dt, n_steps):
    x = np.zeros(n_steps, dtype=np.float32)
    v = np.zeros(n_steps, dtype=np.float32)
    x[0] = x0
    v[0] = v0

    for idx in range(n_steps - 1):
        x[idx + 1] = x[idx]
        v[idx + 1] = v[idx] + dt * force(x[idx], stiffness) / mass

    return x, v


def stormer_verlet(x0, v0, mass, stiffness, dt, n_steps):
    x = np.zeros(n_steps, dtype=np.float32)
    v = np.zeros(n_steps, dtype=np.float32)
    v_half = np.zeros(n_steps - 1, dtype=np.float32)
    x[0] = x0
    v[0] = v0

    for idx in range(n_steps - 1):
        kick_1 = 0.5 * dt * force(x[idx], stiffness) / mass
        v_half[idx] = v[idx] + kick_1
        x[idx + 1] = x[idx] + dt * v_half[idx]
        kick_2 = 0.5 * dt * force(x[idx + 1], stiffness) / mass
        v[idx + 1] = v_half[idx] + kick_2

    return x, v, v_half


def explicit_euler(x0, v0, mass, stiffness, dt, n_steps):
    x = np.zeros(n_steps, dtype=np.float32)
    v = np.zeros(n_steps, dtype=np.float32)
    x[0] = x0
    v[0] = v0

    for idx in range(n_steps - 1):
        x[idx + 1] = x[idx] + dt * v[idx]
        v[idx + 1] = v[idx] + dt * force(x[idx], stiffness) / mass

    return x, v


def arrow_segments_2d(start, end, x_range, y_range):
    p0 = np.asarray(start, dtype=np.float32)
    p1 = np.asarray(end, dtype=np.float32)
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    if length < 1e-7:
        return np.zeros((0, 3), dtype=np.float32)

    unit = delta / length
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
    scale = max(min(x_range[1] - x_range[0], y_range[1] - y_range[0]), 1e-6)
    head_len = min(max(0.025 * scale, 0.25 * length), 0.8 * length)
    head_w = 0.45 * head_len
    h1 = p1 - head_len * unit + head_w * perp
    h2 = p1 - head_len * unit - head_w * perp

    return np.array([
        [p0[0], p0[1], 0.0],
        [p1[0], p1[1], 0.0],
        [p1[0], p1[1], 0.0],
        [h1[0], h1[1], 0.0],
        [p1[0], p1[1], 0.0],
        [h2[0], h2[1], 0.0],
    ], dtype=np.float32)


def arrow_segments_xz(start, end, scale):
    p0 = np.asarray(start, dtype=np.float32)
    p1 = np.asarray(end, dtype=np.float32)
    delta = p1 - p0
    flat = np.array([delta[0], delta[2]], dtype=np.float32)
    length = float(np.linalg.norm(flat))
    if length < 1e-7:
        return np.zeros((0, 3), dtype=np.float32)

    unit = flat / length
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
    head_len = min(max(0.045 * scale, 0.25 * length), 0.8 * length)
    head_w = 0.45 * head_len
    head_1 = np.array([
        p1[0] - head_len * unit[0] + head_w * perp[0],
        p1[1],
        p1[2] - head_len * unit[1] + head_w * perp[1],
    ], dtype=np.float32)
    head_2 = np.array([
        p1[0] - head_len * unit[0] - head_w * perp[0],
        p1[1],
        p1[2] - head_len * unit[1] - head_w * perp[1],
    ], dtype=np.float32)

    return np.array([p0, p1, p1, head_1, p1, head_2], dtype=np.float32)


class PlotPanel:
    def __init__(self, view, title, x_label, y_label, line_colors, legend_labels=None):
        self.view = view
        self.view.camera = scene.PanZoomCamera()
        self.lines = [
            Line(color=color, connect='strip', parent=self.view.scene)
            for color in line_colors
        ]
        self.x_axis = Line(color=AXIS_COLOR, connect='segments', parent=self.view.scene)
        self.y_axis = Line(color=AXIS_COLOR, connect='segments', parent=self.view.scene)
        self.cursor = Line(color=(1, 1, 1, 0.35), connect='segments', parent=self.view.scene)
        self.marker = Markers(parent=self.view.scene)
        self.arrow = Line(color=(1, 1, 1, 0), width=3, connect='segments', parent=self.view.scene)
        self.arrow.visible = False
        self.cursor.visible = False
        self.marker.set_data(pos=np.zeros((1, 3), dtype=np.float32),
                             face_color=(1, 1, 1, 0.95),
                             edge_color=(0, 0, 0, 0.9), size=0)

        self.title = Text(title, color='white', font_size=11, anchor_x='center',
                          anchor_y='bottom', parent=self.view.scene)
        self.x_label = Text(x_label, color=(0.86, 0.88, 0.9, 1), font_size=9,
                            anchor_x='center', anchor_y='top', parent=self.view.scene)
        self.y_label = Text(y_label, color=(0.86, 0.88, 0.9, 1), font_size=9,
                            anchor_x='center', anchor_y='bottom', rotation=-90,
                            parent=self.view.scene)
        self.legend = []
        for label, color in zip(legend_labels or [], line_colors):
            self.legend.append(Text(label, color=color, font_size=8, anchor_x='left',
                                    anchor_y='top', parent=self.view.scene))

        self.series = []
        self.marker_series = 0
        self.x_range = (0.0, 1.0)
        self.y_range = (0.0, 1.0)
        self.is_phase = False

    def set_series(self, series, marker_series=0, is_phase=False):
        self.series = [
            (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
            for x, y in series
        ]
        self.marker_series = marker_series
        self.is_phase = is_phase

        all_x = []
        all_y = []
        for line, (x, y) in zip(self.lines, self.series):
            pos = np.column_stack([x, y, np.zeros_like(x)]).astype(np.float32)
            line.set_data(pos=pos)
            all_x.append(x[np.isfinite(x)])
            all_y.append(y[np.isfinite(y)])

        x_values = np.concatenate(all_x)
        y_values = np.concatenate(all_y)
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        x_pad = max((x_max - x_min) * 0.055, 0.08)
        y_pad = max((y_max - y_min) * 0.10, 0.08)
        self.x_range = (x_min - x_pad, x_max + x_pad)
        self.y_range = (y_min - y_pad, y_max + y_pad)
        self.view.camera.set_range(x=self.x_range, y=self.y_range)
        self._layout_labels()
        self.update_marker(0)

    def _layout_labels(self):
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        width = x1 - x0
        height = y1 - y0
        x_axis_y = 0.0 if y0 <= 0.0 <= y1 else y0
        y_axis_x = 0.0 if x0 <= 0.0 <= x1 else x0
        self.x_axis.set_data(pos=np.array([[x0, x_axis_y, 0], [x1, x_axis_y, 0]], dtype=np.float32))
        self.y_axis.set_data(pos=np.array([[y_axis_x, y0, 0], [y_axis_x, y1, 0]], dtype=np.float32))
        self.title.pos = ((x0 + x1) * 0.5, y1 - 0.035 * height, 0)
        self.x_label.pos = ((x0 + x1) * 0.5, y0 + 0.025 * height, 0)
        self.y_label.pos = (x0 + 0.03 * width, (y0 + y1) * 0.5, 0)
        for idx, label in enumerate(self.legend):
            label.pos = (x0 + 0.065 * width, y1 - (0.13 + idx * 0.075) * height, 0)

    def update_marker(self, index):
        if not self.series:
            return

        x, y = self.series[self.marker_series]
        index = int(np.clip(index, 0, len(x) - 1))
        self.update_marker_xy(float(x[index]), float(y[index]))

        if self.is_phase:
            self.cursor.visible = False
        else:
            self.cursor.visible = True
            y0, y1 = self.y_range
            self.cursor.set_data(pos=np.array([[x[index], y0, 0], [x[index], y1, 0]], dtype=np.float32))

    def update_marker_xy(self, x, y):
        self.marker.set_data(pos=np.array([[x, y, 0]], dtype=np.float32),
                             face_color=(1, 1, 1, 0.95),
                             edge_color=(0, 0, 0, 0.9), size=8)

    def set_arrow(self, start, end, color):
        segments = arrow_segments_2d(start, end, self.x_range, self.y_range)
        if len(segments) == 0:
            self.arrow.visible = False
            return
        self.arrow.visible = True
        self.arrow.set_data(pos=segments, color=color)


class MainWindow(QtWidgets.QMainWindow):
    PRESETS = [
        ('Clear kick/drift demo', 1.0, 0.0, 0.12, 1200),
        ('Large timestep stress test', 1.0, 0.0, 0.20, 750),
        ('Offset velocity loop', 0.65, 0.85, 0.10, 1300),
    ]

    def __init__(self):
        super().__init__()
        self.mass = 1.0
        self.stiffness = 1.0
        self.current_substep = 0
        self.n_substeps = 1
        self.play_stride = 1
        self.geometry_v_scale = 0.85

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        instructions = QtWidgets.QLabel(
            'Vertical integrator / Lagrangian demo: L = 1/2 m v^2 - 1/2 k x^2. '
            'In geometric-algebra language, x is the base 1-vector and v is the vertical fiber 1-vector at that base point. '
            'The orange vertical kick applies p <- p + h F(x), so x is fixed while velocity changes. '
            'The blue horizontal drift applies x <- x + h v, so velocity is fixed while position changes. '
            'The full trajectory composes half-kick, drift, half-kick into the Stormer-Verlet variational integrator; '
            'the red energy curve shows how explicit Euler loses the geometry.'
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet('color: #dfe7ee; background: #0c1118; padding: 7px; font-size: 10pt;')
        layout.addWidget(instructions)

        self.canvas = SceneCanvas(title='Vertical Integrator in GA/Lagrangian Form',
                                  size=(1440, 820), bgcolor='#05070b')
        self.canvas.native.setParent(central)
        layout.addWidget(self.canvas.native)

        self.grid = self.canvas.central_widget.add_grid(spacing=8)
        self.view_geometry = self.grid.add_view(0, 0, row_span=2)
        self.view_kick_x = self.grid.add_view(0, 1)
        self.view_kick_v = self.grid.add_view(0, 2)
        self.view_phase = self.grid.add_view(1, 1)
        self.view_energy = self.grid.add_view(1, 2)

        self._build_geometry_view()
        self._build_plot_views()
        self._build_controls(layout)

        self.timer = QTimer()
        self.timer.setInterval(170)
        self.timer.timeout.connect(self.advance_frame)
        self.status_bar = self.statusBar()

        self.simulate_selected_preset()
        self.update_frame(0)

    def _build_geometry_view(self):
        self.view_geometry.camera = scene.TurntableCamera(elevation=18, azimuth=34, distance=4.1)
        self.view_geometry.camera.set_range(x=(-1.8, 1.8), y=(-0.8, 0.8), z=(-1.35, 1.35))

        self.geometry_title = Text('GA picture: base point plus vertical velocity fiber',
                                   color='white', font_size=14, anchor_x='center',
                                   anchor_y='bottom', parent=self.view_geometry.scene)
        self.geometry_title.pos = (0, 0, 1.22)
        self.base_label = Text('configuration base x', color=(0.82, 0.88, 1.0, 1),
                               font_size=10, anchor_x='center', anchor_y='top',
                               parent=self.view_geometry.scene)
        self.base_label.pos = (0, -0.08, -0.16)
        self.fiber_label = Text('vertical fiber v', color=(1.0, 0.82, 0.25, 1),
                                font_size=10, anchor_x='left', anchor_y='center',
                                parent=self.view_geometry.scene)
        self.fiber_label.pos = (1.22, 0, 0.75)

        self.base_axis = Line(pos=np.array([[-1.55, 0, 0], [1.55, 0, 0]], dtype=np.float32),
                              color=(0.8, 0.86, 0.95, 0.62), width=2,
                              connect='segments', parent=self.view_geometry.scene)
        self.base_axis_head = Line(color=(0.8, 0.86, 0.95, 0.62), width=2,
                                   connect='segments', parent=self.view_geometry.scene)
        self.base_axis_head.set_data(pos=arrow_segments_xz(
            np.array([1.25, 0, 0], dtype=np.float32),
            np.array([1.55, 0, 0], dtype=np.float32),
            2.0,
        ))

        self.phase_trace_3d = Line(color=(0.28, 1.0, 0.55, 0.24), width=1,
                                   connect='strip', parent=self.view_geometry.scene)
        self.fiber_axis = Line(color=(0.86, 0.86, 0.86, 0.26), connect='segments',
                               parent=self.view_geometry.scene)
        self.velocity_vector = Line(color=(1.0, 0.82, 0.25, 1), width=3, connect='segments',
                                    parent=self.view_geometry.scene)
        self.substep_arrow_3d = Line(color=KICK_COLOR, width=4, connect='segments',
                                     parent=self.view_geometry.scene)
        self.mass_marker = Markers(parent=self.view_geometry.scene)
        self.fiber_tip_marker = Markers(parent=self.view_geometry.scene)
        self.start_marker = Markers(parent=self.view_geometry.scene)
        self.end_marker = Markers(parent=self.view_geometry.scene)

    def _build_plot_views(self):
        self.kick_x_plot = PlotPanel(
            self.view_kick_x,
            'Isolated vertical kick: position stays fixed',
            'Time (s)',
            'Position x(t)',
            [(0.35, 0.68, 1.0, 0.95)],
        )
        self.kick_v_plot = PlotPanel(
            self.view_kick_v,
            'Isolated vertical kick: velocity changes',
            'Time (s)',
            'Velocity v(t)',
            [KICK_COLOR],
        )
        self.phase_plot = PlotPanel(
            self.view_phase,
            'Phase space: vertical kick plus horizontal drift',
            'Position x',
            'Velocity v',
            [VERLET_COLOR],
        )
        self.energy_plot = PlotPanel(
            self.view_energy,
            'Energy comparison',
            'Time (s)',
            'log10(Total energy)',
            [VERLET_COLOR, EULER_COLOR],
            ['Stormer-Verlet bounded', 'Explicit Euler drift'],
        )

    def _build_controls(self, layout):
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        layout.addWidget(controls)

        controls_layout.addWidget(QtWidgets.QLabel('Preset:'))
        self.preset_combo = QtWidgets.QComboBox()
        for label, x0, v0, dt, n_steps in self.PRESETS:
            self.preset_combo.addItem(f'{label}: x0={x0:.2f}, v0={v0:.2f}, dt={dt:.2f}')
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        controls_layout.addWidget(self.preset_combo)

        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        step_button = QtWidgets.QPushButton('Step')
        step_button.clicked.connect(self.step_once)
        controls_layout.addWidget(step_button)

        reset_button = QtWidgets.QPushButton('Reset')
        reset_button.clicked.connect(self.reset_animation)
        controls_layout.addWidget(reset_button)

        self.time_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.time_slider.setMinimum(0)
        self.time_slider.valueChanged.connect(self.on_time_slider_changed)
        controls_layout.addWidget(self.time_slider)

        self.stats_label = QtWidgets.QLabel()
        self.stats_label.setMinimumWidth(620)
        controls_layout.addWidget(self.stats_label)

    def simulate_selected_preset(self):
        preset_name, x0, v0, dt, n_steps = self.PRESETS[self.preset_combo.currentIndex()]
        self.preset_name = preset_name
        self.x0 = x0
        self.v0 = v0
        self.dt = dt
        self.n_steps = n_steps
        self.time = np.arange(n_steps, dtype=np.float32) * dt

        kick_steps = 260
        self.kick_dt = min(0.035, dt * 0.3)
        self.kick_time = np.arange(kick_steps, dtype=np.float32) * self.kick_dt
        self.kick_x, self.kick_v = isolated_vertical_kick(
            x0, v0, self.mass, self.stiffness, self.kick_dt, kick_steps
        )

        self.verlet_x, self.verlet_v, self.verlet_v_half = stormer_verlet(
            x0, v0, self.mass, self.stiffness, dt, n_steps
        )
        self.euler_x, self.euler_v = explicit_euler(
            x0, v0, self.mass, self.stiffness, dt, n_steps
        )

        self.verlet_energy = oscillator_energy(
            self.verlet_x, self.verlet_v, self.mass, self.stiffness
        )
        self.euler_energy = oscillator_energy(
            self.euler_x, self.euler_v, self.mass, self.stiffness
        )
        self.log_verlet_energy = np.log10(np.maximum(self.verlet_energy, 1e-12))
        self.log_euler_energy = np.log10(np.maximum(self.euler_energy, 1e-12))

        self.n_substeps = max(1, 3 * (self.n_steps - 1))
        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(self.n_substeps - 1)
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)

        self.kick_x_plot.set_series([(self.kick_time, self.kick_x)])
        self.kick_v_plot.set_series([(self.kick_time, self.kick_v)])
        self.phase_plot.set_series([(self.verlet_x, self.verlet_v)], is_phase=True)
        self.energy_plot.set_series([
            (self.time, self.log_verlet_energy),
            (self.time, self.log_euler_energy),
        ])

        self._update_3d_trace()
        initial_energy = float(self.verlet_energy[0])
        final_drift = (float(self.verlet_energy[-1]) - initial_energy) / initial_energy * 100.0
        max_deviation = float(np.max(np.abs(self.verlet_energy - initial_energy))) / initial_energy * 100.0
        euler_ratio = float(self.euler_energy[-1] / initial_energy)
        self.status_bar.showMessage(
            f'{preset_name} | Stormer-Verlet final drift {final_drift:+.3e}% | '
            f'max bounded variation {max_deviation:.3e}% | Euler final energy ratio {euler_ratio:.3e}'
        )

    def _update_3d_trace(self):
        trace = np.column_stack([
            self.verlet_x,
            np.zeros_like(self.verlet_x),
            self.geometry_v_scale * self.verlet_v,
        ]).astype(np.float32)
        self.phase_trace_3d.set_data(pos=trace)

    def on_preset_changed(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.current_substep = 0
        self.simulate_selected_preset()
        self.update_frame(0)

    def on_time_slider_changed(self, value):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(value)

    def reset_animation(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(0)

    def step_once(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(min(self.current_substep + 1, self.n_substeps - 1))

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play')
        else:
            if self.current_substep >= self.n_substeps - 1:
                self.update_frame(0)
            self.timer.start()
            self.play_button.setText('Pause')

    def advance_frame(self):
        next_substep = min(self.current_substep + self.play_stride, self.n_substeps - 1)
        self.update_frame(next_substep)
        if next_substep >= self.n_substeps - 1:
            self.timer.stop()
            self.play_button.setText('Play')

    def substep_info(self, substep):
        step = int(np.clip(substep // 3, 0, self.n_steps - 2))
        phase = int(substep % 3)
        x0 = float(self.verlet_x[step])
        v0 = float(self.verlet_v[step])
        v_half = float(self.verlet_v_half[step])
        x1 = float(self.verlet_x[step + 1])
        v1 = float(self.verlet_v[step + 1])

        if phase == 0:
            return {
                'step': step,
                'name': 'vertical half-kick',
                'description': 'x fixed, velocity changes',
                'start': np.array([x0, v0], dtype=np.float32),
                'end': np.array([x0, v_half], dtype=np.float32),
                'state': np.array([x0, v_half], dtype=np.float32),
                'color': KICK_COLOR,
                'proof_label': 'delta x',
                'proof_value': x0 - x0,
            }
        if phase == 1:
            return {
                'step': step,
                'name': 'horizontal drift',
                'description': 'velocity fixed, x changes',
                'start': np.array([x0, v_half], dtype=np.float32),
                'end': np.array([x1, v_half], dtype=np.float32),
                'state': np.array([x1, v_half], dtype=np.float32),
                'color': DRIFT_COLOR,
                'proof_label': 'delta v',
                'proof_value': v_half - v_half,
            }
        return {
            'step': step,
            'name': 'vertical half-kick',
            'description': 'x fixed, velocity changes',
            'start': np.array([x1, v_half], dtype=np.float32),
            'end': np.array([x1, v1], dtype=np.float32),
            'state': np.array([x1, v1], dtype=np.float32),
            'color': KICK_COLOR,
            'proof_label': 'delta x',
            'proof_value': x1 - x1,
        }

    def update_frame(self, substep):
        self.current_substep = int(np.clip(substep, 0, self.n_substeps - 1))
        info = self.substep_info(self.current_substep)
        step = info['step']
        state_x, state_v = map(float, info['state'])

        self._update_geometry_state(info)
        self.phase_plot.update_marker_xy(state_x, state_v)
        self.phase_plot.set_arrow(info['start'], info['end'], info['color'])
        self.energy_plot.update_marker(step)

        kick_marker = int(round((self.current_substep / max(self.n_substeps - 1, 1)) * (len(self.kick_x) - 1)))
        self.kick_x_plot.update_marker(kick_marker)
        self.kick_v_plot.update_marker(kick_marker)

        self.time_slider.blockSignals(True)
        self.time_slider.setValue(self.current_substep)
        self.time_slider.blockSignals(False)

        current_energy = oscillator_energy(state_x, state_v, self.mass, self.stiffness)
        self.stats_label.setText(
            f'{info["name"]}: {info["description"]}   '
            f'x={state_x:+.3f}, v={state_v:+.3f}, E={current_energy:.6f}   '
            f'{info["proof_label"]}={info["proof_value"]:+.1e}'
        )
        self.setWindowTitle(
            f'Vertical integrator GA demo | {self.preset_name} | '
            f'step {step}/{self.n_steps - 1} | {info["name"]}'
        )
        self.canvas.update()

    def _update_geometry_state(self, info):
        start = info['start']
        end = info['end']
        state = info['state']
        state_x = float(state[0])
        state_v = float(state[1])
        state_z = self.geometry_v_scale * state_v
        start_3d = np.array([start[0], 0.0, self.geometry_v_scale * start[1]], dtype=np.float32)
        end_3d = np.array([end[0], 0.0, self.geometry_v_scale * end[1]], dtype=np.float32)

        self.fiber_axis.set_data(pos=np.array([
            [state_x, 0.0, -1.15],
            [state_x, 0.0, 1.15],
        ], dtype=np.float32))
        self.velocity_vector.set_data(pos=arrow_segments_xz(
            np.array([state_x, 0.0, 0.0], dtype=np.float32),
            np.array([state_x, 0.0, state_z], dtype=np.float32),
            2.0,
        ))
        self.substep_arrow_3d.set_data(pos=arrow_segments_xz(start_3d, end_3d, 2.0), color=info['color'])
        self.mass_marker.set_data(pos=np.array([[state_x, 0.0, 0.0]], dtype=np.float32),
                                  face_color=(0.22, 0.58, 1.0, 1.0),
                                  edge_color=(0, 0, 0, 1), size=15)
        self.fiber_tip_marker.set_data(pos=np.array([[state_x, 0.0, state_z]], dtype=np.float32),
                                       face_color=(1.0, 0.82, 0.25, 1.0),
                                       edge_color=(0, 0, 0, 1), size=10)
        self.start_marker.set_data(pos=np.array([start_3d], dtype=np.float32),
                                   face_color=(0.95, 0.95, 0.95, 0.9),
                                   edge_color=(0, 0, 0, 0.9), size=7)
        self.end_marker.set_data(pos=np.array([end_3d], dtype=np.float32),
                                 face_color=info['color'],
                                 edge_color=(0, 0, 0, 0.9), size=9)


if __name__ == '__main__':
    qt_app = QtWidgets.QApplication([])
    app.use_app('pyqt5')
    window = MainWindow()
    window.show()
    qt_app.exec_()
