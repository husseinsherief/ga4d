import numpy as np

from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Line, Markers, Text

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer


AXIS_COLOR = (0.78, 0.82, 0.88, 0.50)
PARTICLE_COLOR = (0.25, 0.58, 1.0, 1.0)
FORCE_COLOR = (1.0, 0.52, 0.12, 1.0)
MASS_ACC_COLOR = (0.35, 1.0, 0.52, 0.95)
KINETIC_COLOR = (0.35, 0.72, 1.0, 0.95)
POTENTIAL_COLOR = (1.0, 0.74, 0.22, 0.95)
LAGRANGIAN_COLOR = (0.9, 0.44, 1.0, 0.95)
FIELD_COLOR = (0.35, 1.0, 0.78, 0.95)
FIELD_TERM_COLOR = (0.25, 0.72, 1.0, 0.95)
FIELD_RHS_COLOR = (1.0, 0.50, 0.16, 0.85)
RESIDUAL_COLOR = (1.0, 0.25, 0.34, 0.95)
ENERGY_DENSITY_COLOR = (1.0, 0.82, 0.28, 0.90)


def particle_solution(time, mass, stiffness, x0, v0):
    omega = np.sqrt(stiffness / mass)
    x = x0 * np.cos(omega * time) + (v0 / omega) * np.sin(omega * time)
    v = -x0 * omega * np.sin(omega * time) + v0 * np.cos(omega * time)
    acceleration = -(stiffness / mass) * x
    force = -stiffness * x
    kinetic = 0.5 * mass * v * v
    potential = 0.5 * stiffness * x * x
    lagrangian = kinetic - potential
    ma = mass * acceleration
    residual = ma - force
    return x, v, acceleration, force, kinetic, potential, lagrangian, ma, residual


def field_rhs(psi, dx, c, mu):
    rhs = np.zeros_like(psi)
    laplacian = (psi[:-2] - 2.0 * psi[1:-1] + psi[2:]) / (dx * dx)
    rhs[1:-1] = c * c * laplacian - mu * mu * psi[1:-1]
    return rhs


def simulate_field(x_grid, dt, n_steps, c, mu):
    dx = float(x_grid[1] - x_grid[0])
    psi = np.zeros((n_steps, len(x_grid)), dtype=np.float32)

    psi0 = (
        0.72 * np.exp(-((x_grid + 0.70) / 0.33) ** 2)
        - 0.32 * np.exp(-((x_grid - 0.45) / 0.28) ** 2)
    )
    velocity0 = 0.24 * np.exp(-((x_grid + 0.15) / 0.55) ** 2)
    psi0[0] = 0.0
    psi0[-1] = 0.0
    velocity0[0] = 0.0
    velocity0[-1] = 0.0

    accel0 = field_rhs(psi0, dx, c, mu)
    psi[0] = psi0
    psi[1] = psi0 + dt * velocity0 + 0.5 * dt * dt * accel0
    psi[1, 0] = 0.0
    psi[1, -1] = 0.0

    for idx in range(1, n_steps - 1):
        psi[idx + 1] = 2.0 * psi[idx] - psi[idx - 1] + dt * dt * field_rhs(psi[idx], dx, c, mu)
        psi[idx + 1, 0] = 0.0
        psi[idx + 1, -1] = 0.0

    psi_t = np.zeros_like(psi)
    psi_t[1:-1] = (psi[2:] - psi[:-2]) / (2.0 * dt)
    psi_t[0] = velocity0
    psi_t[-1] = (psi[-1] - psi[-2]) / dt

    psi_tt = np.zeros_like(psi)
    rhs = np.zeros_like(psi)
    for idx in range(n_steps):
        rhs[idx] = field_rhs(psi[idx], dx, c, mu)
    psi_tt[1:-1] = (psi[2:] - 2.0 * psi[1:-1] + psi[:-2]) / (dt * dt)
    psi_tt[0] = rhs[0]
    psi_tt[-1] = rhs[-1]

    psi_x = np.zeros_like(psi)
    psi_x[:, 1:-1] = (psi[:, 2:] - psi[:, :-2]) / (2.0 * dx)
    psi_x[:, 0] = (psi[:, 1] - psi[:, 0]) / dx
    psi_x[:, -1] = (psi[:, -1] - psi[:, -2]) / dx

    lagrangian_density = 0.5 * psi_t * psi_t - 0.5 * c * c * psi_x * psi_x - 0.5 * mu * mu * psi * psi
    energy_density = 0.5 * psi_t * psi_t + 0.5 * c * c * psi_x * psi_x + 0.5 * mu * mu * psi * psi
    field_lagrangian = np.trapezoid(lagrangian_density, x_grid, axis=1)
    field_energy = np.trapezoid(energy_density, x_grid, axis=1)
    residual = psi_tt - rhs

    return {
        'psi': psi,
        'psi_t': psi_t,
        'psi_tt': psi_tt,
        'rhs': rhs,
        'lagrangian_density': lagrangian_density,
        'energy_density': energy_density,
        'field_lagrangian': field_lagrangian,
        'field_energy': field_energy,
        'residual': residual,
    }


def arrow_segments_xz(start, end, scale):
    p0 = np.asarray(start, dtype=np.float32)
    p1 = np.asarray(end, dtype=np.float32)
    flat = np.array([p1[0] - p0[0], p1[2] - p0[2]], dtype=np.float32)
    length = float(np.linalg.norm(flat))
    if length < 1e-7:
        return np.zeros((0, 3), dtype=np.float32)

    unit = flat / length
    perp = np.array([-unit[1], unit[0]], dtype=np.float32)
    head_len = min(max(0.045 * scale, 0.25 * length), 0.70 * length)
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


def make_spring(anchor_x, mass_x, turns=12, amplitude=0.075):
    if abs(mass_x - anchor_x) < 0.05:
        return np.array([[anchor_x, 0.0, 0.0], [mass_x, 0.0, 0.0]], dtype=np.float32)
    samples = turns * 8
    xs = np.linspace(anchor_x, mass_x, samples, dtype=np.float32)
    wiggle = amplitude * np.sin(np.linspace(0.0, turns * 2.0 * np.pi, samples))
    wiggle[0] = 0.0
    wiggle[-1] = 0.0
    return np.column_stack([xs, wiggle, np.zeros_like(xs)]).astype(np.float32)


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
        self.x_range = (0.0, 1.0)
        self.y_range = (0.0, 1.0)
        self.data_x_range = (0.0, 1.0)
        self.data_y_range = (0.0, 1.0)
        self.marker_series = 0

    def set_series(self, series, marker_series=0, fixed_y_range=None):
        self.series = [
            (np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32))
            for x, y in series
        ]
        self.marker_series = marker_series
        self._draw_lines()
        self._set_ranges(fixed_y_range)
        self.update_marker(0)

    def update_series_y(self, y_values):
        if not self.series:
            return
        next_series = []
        for (x, _old_y), new_y in zip(self.series, y_values):
            next_series.append((x, np.asarray(new_y, dtype=np.float32)))
        self.series = next_series
        self._draw_lines()

    def _draw_lines(self):
        for line, (x, y) in zip(self.lines, self.series):
            pos = np.column_stack([x, y, np.zeros_like(x)]).astype(np.float32)
            line.set_data(pos=pos)

    def _set_ranges(self, fixed_y_range):
        all_x = []
        all_y = []
        for x, y in self.series:
            all_x.append(x[np.isfinite(x)])
            all_y.append(y[np.isfinite(y)])
        x_values = np.concatenate(all_x)
        y_values = np.concatenate(all_y)
        x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
        y_min, y_max = float(np.min(y_values)), float(np.max(y_values))
        if fixed_y_range is not None:
            y_min, y_max = fixed_y_range
        x_width = max(x_max - x_min, 1e-6)
        y_height = max(y_max - y_min, 1e-6)
        self.data_x_range = (x_min, x_max)
        self.data_y_range = (y_min, y_max)

        left_gutter = max(x_width * 0.22, 0.12)
        right_gutter = max(x_width * 0.08, 0.08)
        bottom_gutter = max(y_height * 0.28, 0.12)
        top_gutter = max(y_height * 0.30, 0.12)
        self.x_range = (x_min - left_gutter, x_max + right_gutter)
        self.y_range = (y_min - bottom_gutter, y_max + top_gutter)
        self.view.camera.set_range(x=self.x_range, y=self.y_range)
        self._layout_labels()

    def _layout_labels(self):
        x0, x1 = self.x_range
        y0, y1 = self.y_range
        data_x0, data_x1 = self.data_x_range
        data_y0, data_y1 = self.data_y_range
        width = x1 - x0
        height = y1 - y0
        data_center_x = (data_x0 + data_x1) * 0.5
        data_center_y = (data_y0 + data_y1) * 0.5
        x_axis_y = 0.0 if data_y0 <= 0.0 <= data_y1 else data_y0
        y_axis_x = 0.0 if data_x0 <= 0.0 <= data_x1 else data_x0
        self.x_axis.set_data(pos=np.array([[data_x0, x_axis_y, 0], [data_x1, x_axis_y, 0]], dtype=np.float32))
        self.y_axis.set_data(pos=np.array([[y_axis_x, data_y0, 0], [y_axis_x, data_y1, 0]], dtype=np.float32))
        self.title.pos = (data_center_x, y1 - 0.070 * height, 0)
        self.x_label.pos = (data_center_x, y0 + 0.085 * height, 0)
        self.y_label.pos = (x0 + 0.070 * width, data_center_y, 0)
        for idx, label in enumerate(self.legend):
            label.pos = (x0 + 0.085 * width, y1 - (0.15 + idx * 0.075) * height, 0)

    def update_marker(self, index):
        if not self.series:
            return
        x, y = self.series[self.marker_series]
        index = int(np.clip(index, 0, len(x) - 1))
        self.marker.set_data(pos=np.array([[x[index], y[index], 0]], dtype=np.float32),
                             face_color=(1, 1, 1, 0.95),
                             edge_color=(0, 0, 0, 0.9), size=8)
        self.cursor.visible = True
        y0, y1 = self.data_y_range
        self.cursor.set_data(pos=np.array([[x[index], y0, 0], [x[index], y1, 0]], dtype=np.float32))

    def hide_marker(self):
        self.cursor.visible = False
        self.marker.set_data(pos=np.zeros((1, 3), dtype=np.float32), size=0)


class MainWindow(QtWidgets.QMainWindow):
    PRESETS = [
        ('Standard oscillator and massive field', 1.0, 1.0, 1.0, 1.0),
        ('Stiffer particle, lighter field mass', 1.0, 1.8, 1.0, 0.45),
        ('Heavy particle, massless wave field', 1.8, 1.0, 1.0, 0.0),
    ]

    def __init__(self):
        super().__init__()
        self.dt = 0.012
        self.n_steps = 1800
        self.current_index = 0
        self.play_stride = 3
        self.x0 = 1.0
        self.v0 = 0.0
        self.c = 1.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        instructions = QtWidgets.QLabel(
            'Euler-Lagrange demo. Particle side: L = T - V is only the integrand; '
            'the EL equation differentiates it into d/dt(dL/dv) - dL/dx = 0, which becomes ma = F. '
            'Field side: the Lagrangian density mathcal L is local at each x; integrate it over space to get L(t), '
            'then over time to get action S. The field EL equation becomes psi_tt = c^2 psi_xx - mu^2 psi.'
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet('color: #dfe7ee; background: #0c1118; padding: 7px; font-size: 10pt;')
        layout.addWidget(instructions)

        self.canvas = SceneCanvas(title='Euler-Lagrange: Particle Mechanics and Field Theory',
                                  size=(1500, 860), bgcolor='#05070b')
        self.canvas.native.setParent(central)
        layout.addWidget(self.canvas.native)

        self.grid = self.canvas.central_widget.add_grid(spacing=8)
        self.view_particle = self.grid.add_view(0, 0, row_span=2)
        self.view_integrand = self.grid.add_view(0, 1)
        self.view_particle_el = self.grid.add_view(1, 1)
        self.view_field = self.grid.add_view(0, 2, row_span=2)
        self.view_density = self.grid.add_view(0, 3)
        self.view_field_el = self.grid.add_view(1, 3)

        self._build_particle_view()
        self._build_field_view()
        self._build_plot_views()
        self._build_controls(layout)

        self.timer = QTimer()
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.advance_frame)
        self.status_bar = self.statusBar()

        self.simulate_selected_preset()
        self.update_frame(0)

    def _build_particle_view(self):
        self.view_particle.camera = scene.TurntableCamera(elevation=15, azimuth=31, distance=4.0)
        self.view_particle.camera.set_range(x=(-1.6, 1.6), y=(-0.6, 0.6), z=(-0.7, 0.9))

        self.particle_title = Text('Particle mechanics: EL turns L into ma = F',
                                   color='white', font_size=14, anchor_x='center',
                                   anchor_y='bottom', parent=self.view_particle.scene)
        self.particle_title.pos = (0.0, 0.0, 0.78)

        self.track = Line(pos=np.array([[-1.35, 0, 0], [1.35, 0, 0]], dtype=np.float32),
                          color=AXIS_COLOR, width=2, connect='segments',
                          parent=self.view_particle.scene)
        self.spring = Line(color=(0.86, 0.90, 1.0, 0.72), width=2,
                           connect='strip', parent=self.view_particle.scene)
        self.force_arrow = Line(color=FORCE_COLOR, width=4, connect='segments',
                                parent=self.view_particle.scene)
        self.ma_arrow = Line(color=MASS_ACC_COLOR, width=3, connect='segments',
                             parent=self.view_particle.scene)
        self.particle_marker = Markers(parent=self.view_particle.scene)
        self.anchor_marker = Markers(parent=self.view_particle.scene)
        self.anchor_marker.set_data(pos=np.array([[0, 0, 0]], dtype=np.float32),
                                    face_color=(0.8, 0.86, 0.95, 1),
                                    edge_color=(0, 0, 0, 1), size=8)

        self.force_label = Text('F = -dV/dx', color=FORCE_COLOR, font_size=10,
                                anchor_x='center', anchor_y='top', parent=self.view_particle.scene)
        self.ma_label = Text('d/dt(dL/dv) = ma', color=MASS_ACC_COLOR, font_size=10,
                             anchor_x='center', anchor_y='bottom', parent=self.view_particle.scene)

    def _build_field_view(self):
        self.view_field.camera = scene.TurntableCamera(elevation=25, azimuth=39, distance=5.3)
        self.view_field.camera.set_range(x=(-2.1, 2.1), y=(-0.8, 0.8), z=(-1.0, 1.0))

        self.field_title = Text('Field theory: every x has its own psi(t, x)',
                                color='white', font_size=14, anchor_x='center',
                                anchor_y='bottom', parent=self.view_field.scene)
        self.field_title.pos = (0.0, 0.0, 0.86)
        self.field_base = Line(pos=np.array([[-2.0, 0, 0], [2.0, 0, 0]], dtype=np.float32),
                               color=AXIS_COLOR, width=2, connect='segments',
                               parent=self.view_field.scene)
        self.field_curve = Line(color=FIELD_COLOR, width=3, connect='strip',
                                parent=self.view_field.scene)
        self.field_zero = Line(pos=np.array([[-2.0, 0.03, 0], [2.0, 0.03, 0]], dtype=np.float32),
                               color=(0.78, 0.82, 0.88, 0.25), width=1,
                               connect='segments', parent=self.view_field.scene)
        self.field_markers = Markers(parent=self.view_field.scene)

    def _build_plot_views(self):
        self.integrand_plot = PlotPanel(
            self.view_integrand,
            'Particle integrand: L = T - V',
            'Time (s)',
            'Energy / L',
            [KINETIC_COLOR, POTENTIAL_COLOR, LAGRANGIAN_COLOR],
            ['T kinetic', 'V potential', 'L = T - V'],
        )
        self.particle_el_plot = PlotPanel(
            self.view_particle_el,
            'Particle EL terms: ma equals force',
            'Time (s)',
            'Term value',
            [MASS_ACC_COLOR, FORCE_COLOR, RESIDUAL_COLOR],
            ['ma = d/dt(dL/dv)', 'F = dL/dx', 'ma - F'],
        )
        self.density_plot = PlotPanel(
            self.view_density,
            'Field density at current time',
            'Space x',
            'Density',
            [LAGRANGIAN_COLOR, ENERGY_DENSITY_COLOR],
            ['mathcal L density', 'energy density'],
        )
        self.field_el_plot = PlotPanel(
            self.view_field_el,
            'Field EL terms: local equation at each x',
            'Space x',
            'Term value',
            [FIELD_TERM_COLOR, FIELD_RHS_COLOR, RESIDUAL_COLOR],
            ['psi_tt', 'c^2 psi_xx - mu^2 psi', 'EL residual'],
        )

    def _build_controls(self, layout):
        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        layout.addWidget(controls)

        controls_layout.addWidget(QtWidgets.QLabel('Preset:'))
        self.preset_combo = QtWidgets.QComboBox()
        for label, mass, stiffness, c, mu in self.PRESETS:
            self.preset_combo.addItem(f'{label}: m={mass:.1f}, k={stiffness:.1f}, c={c:.1f}, mu={mu:.2f}')
        self.preset_combo.currentIndexChanged.connect(self.on_preset_changed)
        controls_layout.addWidget(self.preset_combo)

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
        self.stats_label.setMinimumWidth(680)
        controls_layout.addWidget(self.stats_label)

    def simulate_selected_preset(self):
        preset_name, mass, stiffness, c, mu = self.PRESETS[self.preset_combo.currentIndex()]
        self.preset_name = preset_name
        self.mass = mass
        self.stiffness = stiffness
        self.c = c
        self.mu = mu
        self.time = np.arange(self.n_steps, dtype=np.float32) * self.dt

        particle = particle_solution(self.time, mass, stiffness, self.x0, self.v0)
        (
            self.particle_x,
            self.particle_v,
            self.particle_accel,
            self.particle_force,
            self.kinetic,
            self.potential,
            self.particle_lagrangian,
            self.particle_ma,
            self.particle_residual,
        ) = particle

        self.field_x = np.linspace(-2.0, 2.0, 161, dtype=np.float32)
        self.field_data = simulate_field(self.field_x, self.dt, self.n_steps, c, mu)

        max_density = float(np.max(np.abs(np.concatenate([
            self.field_data['lagrangian_density'].ravel(),
            self.field_data['energy_density'].ravel(),
        ]))))
        max_field_term = float(np.max(np.abs(np.concatenate([
            self.field_data['psi_tt'].ravel(),
            self.field_data['rhs'].ravel(),
        ]))))

        self.integrand_plot.set_series([
            (self.time, self.kinetic),
            (self.time, self.potential),
            (self.time, self.particle_lagrangian),
        ])
        self.particle_el_plot.set_series([
            (self.time, self.particle_ma),
            (self.time, self.particle_force),
            (self.time, self.particle_residual),
        ])
        self.density_plot.set_series([
            (self.field_x, self.field_data['lagrangian_density'][0]),
            (self.field_x, self.field_data['energy_density'][0]),
        ], fixed_y_range=(-max_density, max_density))
        self.field_el_plot.set_series([
            (self.field_x, self.field_data['psi_tt'][0]),
            (self.field_x, self.field_data['rhs'][0]),
            (self.field_x, self.field_data['residual'][0]),
        ], fixed_y_range=(-max_field_term, max_field_term))
        self.density_plot.hide_marker()
        self.field_el_plot.hide_marker()

        self.time_slider.blockSignals(True)
        self.time_slider.setMaximum(self.n_steps - 1)
        self.time_slider.setValue(0)
        self.time_slider.blockSignals(False)
        self.status_bar.showMessage(
            f'{preset_name} | particle equation: ma - F = 0 | '
            f'field equation: psi_tt - c^2 psi_xx + mu^2 psi = 0'
        )

    def on_preset_changed(self):
        self.timer.stop()
        self.play_button.setText('Play')
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
        idx = self.current_index
        self._update_particle_view(idx)
        self._update_field_view(idx)

        self.integrand_plot.update_marker(idx)
        self.particle_el_plot.update_marker(idx)
        self.density_plot.update_series_y([
            self.field_data['lagrangian_density'][idx],
            self.field_data['energy_density'][idx],
        ])
        self.field_el_plot.update_series_y([
            self.field_data['psi_tt'][idx],
            self.field_data['rhs'][idx],
            self.field_data['residual'][idx],
        ])

        self.time_slider.blockSignals(True)
        self.time_slider.setValue(idx)
        self.time_slider.blockSignals(False)

        particle_residual = float(abs(self.particle_residual[idx]))
        field_residual = float(np.max(np.abs(self.field_data['residual'][idx, 1:-1])))
        field_l = float(self.field_data['field_lagrangian'][idx])
        field_e = float(self.field_data['field_energy'][idx])
        self.stats_label.setText(
            f't={self.time[idx]:.2f}s   particle |ma-F|={particle_residual:.2e}   '
            f'field max |EL residual|={field_residual:.2e}   '
            f'L_field(t)=int dx mathcal L={field_l:+.4f}   E_field={field_e:.4f}'
        )
        self.setWindowTitle(
            f'Euler-Lagrange particle and field demo | {self.preset_name} | t={self.time[idx]:.2f}s'
        )
        self.canvas.update()

    def _update_particle_view(self, idx):
        x = float(self.particle_x[idx])
        force = float(self.particle_force[idx])
        ma = float(self.particle_ma[idx])
        force_scale = 0.40 / max(self.stiffness * abs(self.x0), 1e-6)

        self.spring.set_data(pos=make_spring(0.0, x))
        self.particle_marker.set_data(pos=np.array([[x, 0, 0]], dtype=np.float32),
                                      face_color=PARTICLE_COLOR,
                                      edge_color=(0, 0, 0, 1), size=18)

        force_start = np.array([x, -0.10, 0.10], dtype=np.float32)
        force_end = np.array([x + force * force_scale, -0.10, 0.10], dtype=np.float32)
        ma_start = np.array([x, 0.12, -0.12], dtype=np.float32)
        ma_end = np.array([x + ma * force_scale, 0.12, -0.12], dtype=np.float32)
        self.force_arrow.set_data(pos=arrow_segments_xz(force_start, force_end, 1.7))
        self.ma_arrow.set_data(pos=arrow_segments_xz(ma_start, ma_end, 1.7))
        self.force_label.pos = ((force_start[0] + force_end[0]) * 0.5, -0.10, 0.26)
        self.ma_label.pos = ((ma_start[0] + ma_end[0]) * 0.5, 0.12, -0.28)

    def _update_field_view(self, idx):
        psi = self.field_data['psi'][idx]
        pos = np.column_stack([
            self.field_x,
            np.zeros_like(self.field_x),
            psi,
        ]).astype(np.float32)
        self.field_curve.set_data(pos=pos)
        sample = pos[::12]
        self.field_markers.set_data(pos=sample, face_color=FIELD_COLOR,
                                    edge_color=(0, 0, 0, 1), size=7)


if __name__ == '__main__':
    qt_app = QtWidgets.QApplication([])
    app.use_app('pyqt5')
    window = MainWindow()
    window.show()
    qt_app.exec_()
