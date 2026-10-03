"""Coupled flood--earthquake 4D phase visualization.

This is an intentionally small, deterministic *modal* visualization of the
coupled equations in the accompanying notes.  A full shallow-water/elastic
PDE solve needs a spatial mesh and boundary conditions; here one Fourier mode
of (h, hu, u_y, du_y/dt_2) is evolved with explicit Euler.  The binary swap
schedule is applied exactly as

    X[k + 1] = X[k] + dt * ((1-s[k]) I + s[k] P) F(X[k]).

The resulting four-component trajectory is shown as the six pairwise angles
theta_ij = atan2(X_j, X_i), arranged on three great S² spheres:
(theta12, theta34), (theta13, theta24), and (theta14, theta23).  All three
sit on one parent S³ hypersphere in R⁴ and are sent through one 4D rotation +
perspective map.

Requirements: numpy, PyQt5, vispy
Run: python coupled-flood-earthquake-4d-spheres.py
"""

import sys
from datetime import datetime, timedelta, timezone
from urllib.request import urlopen

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Line, Markers, Text


PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
SPHERES = (
    ('S²₁  flood planes', (0, 5), 'θ₁₂ latitude  •  θ₃₄ longitude', (0, 1, 2)),
    ('S²₂  cross planes', (1, 4), 'θ₁₃ latitude  •  θ₂₄ longitude', (0, 1, 3)),
    ('S²₃  cross planes', (2, 3), 'θ₁₄ latitude  •  θ₂₃ longitude', (0, 2, 3)),
)
COLORS = ((0.14, 0.78, 1.0, 0.96), (0.98, 0.52, 0.23, 0.96), (0.40, 0.91, 0.56, 0.96))
GRID_COLORS = ((0.14, 0.78, 1.0, 0.62), (0.98, 0.52, 0.23, 0.62), (0.40, 0.91, 0.56, 0.62))
RADIUS = 0.92
PROJECTION_DISTANCE = 4.2
TRAIL_LENGTH = 260  # recent samples rendered as the bright directional tail
TOHOKU_EVENT_TIME = datetime(2011, 3, 11, 5, 46, 24, tzinfo=timezone.utc)
TOHOKU_DART_21418_URL = (
    'https://www.ngdc.noaa.gov/hazard/data/DART/20110311_honshu/'
    'dart21418_20110301to20110320_meter.txt'
)


def deterministic_schedule(n_steps, period):
    """Swap on every `period`th step.  period=0 is the uncoupled run."""
    if period <= 0:
        return np.zeros(n_steps, dtype=np.int8)
    return ((np.arange(1, n_steps + 1) % period) == 0).astype(np.int8)


def permutation_for(choice):
    """Permutation vectors for selectable, fixed coordinate transpositions."""
    return {
        0: np.array((1, 0, 2, 3)),  # h <-> hu
        1: np.array((0, 2, 1, 3)),  # hu <-> u_y
        2: np.array((0, 1, 3, 2)),  # u_y <-> v_y
    }[choice]


def modal_field(x):
    """One-mode flood/interface/seismic surrogate F(X).

    X = (eta, q, u_y, v_y): free-surface perturbation, flood momentum mode,
    vertical bed displacement, and its t2 velocity.  z_b = -u_y supplies the
    moving-bed interface source.  Constants are normalized for stable display.
    """
    eta, q, uy, vy = x
    gravity, depth, wave_number = 1.0, 1.0, 0.85
    flood_damping, seismic_damping = 0.075, 0.045
    omega_s, flood_to_seismic, bed_to_flood = 1.12, 0.40, 0.33

    d_eta = -wave_number * q
    # -g h ∂x z_b with z_b=-u_y, represented by the coupled bed mode.
    d_q = -(gravity * depth * wave_number) * eta - flood_damping * q + bed_to_flood * uy
    d_uy = vy
    d_vy = -(omega_s ** 2) * uy - seismic_damping * vy + flood_to_seismic * eta
    return np.array((d_eta, d_q, d_uy, d_vy), dtype=np.float64)


def simulate(n_steps=2400, dt=0.012, period=6, permutation_choice=0):
    """Integrate the deterministic swap-coupled explicit Euler system."""
    schedule = deterministic_schedule(n_steps, period)
    permutation = permutation_for(permutation_choice)
    states = np.empty((n_steps + 1, 4), dtype=np.float64)
    states[0] = (0.72, 0.0, -0.36, 0.48)
    for k, swapped in enumerate(schedule):
        field = modal_field(states[k])
        if swapped:
            field = field[permutation]
        states[k + 1] = states[k] + dt * field

    angles = np.column_stack([np.arctan2(states[:, j], states[:, i]) for i, j in PAIRS])
    return states, angles, schedule, np.arange(n_steps + 1, dtype=np.float64) * dt


def _robust_scale(values):
    """Center and scale a measurement without changing its temporal structure."""
    values = np.asarray(values, dtype=np.float64)
    centered = values - np.median(values)
    scale = np.percentile(np.abs(centered), 95)
    return centered / max(scale, 1e-12)


def load_tohoku_dart_21418(n_steps=2400, hours=6):
    """Build a real, observational phase embedding from NOAA DART station 21418.

    NOAA distributes this station's bottom-pressure observations, fitted tide,
    and residual water column height in metres.  The four plotted channels are
    measured residual height plus three explicitly data-derived water-signal
    features.  They are *not* asserted to be direct seismic displacement.
    """
    try:
        with urlopen(TOHOKU_DART_21418_URL, timeout=30) as response:
            text = response.read().decode('utf-8')
    except OSError as exc:
        raise RuntimeError(f'Could not download NOAA DART 21418 observations: {exc}') from exc

    begin = TOHOKU_EVENT_TIME - timedelta(minutes=15)
    end = TOHOKU_EVENT_TIME + timedelta(hours=hours)
    samples_time, residual = [], []
    for line in text.splitlines():
        fields = line.split()
        if len(fields) < 10:
            continue
        try:
            timestamp = datetime(
                int(fields[1]), int(fields[2]), int(fields[3]), int(fields[4]),
                int(fields[5]), int(fields[6]), tzinfo=timezone.utc,
            )
            residual_value = float(fields[9])
        except ValueError:
            continue
        # NOAA uses 9999.0 as a missing-value sentinel in this event file.
        if begin <= timestamp <= end and np.isfinite(residual_value) and abs(residual_value) < 100.0:
            samples_time.append((timestamp - begin).total_seconds())
            residual.append(residual_value)
    if len(samples_time) < 20:
        raise RuntimeError('NOAA DART 21418 response did not contain the requested event window.')

    target_time = np.linspace(0.0, (end - begin).total_seconds(), n_steps + 1)
    if samples_time[0] > target_time[0] + 300 or samples_time[-1] < target_time[-1] - 300:
        raise RuntimeError('NOAA DART 21418 data have a gap at the edge of the requested event window.')
    eta_m = np.interp(target_time, samples_time, residual)
    eta_m -= np.median(eta_m[:max(5, n_steps // 80)])
    rate_m_per_s = np.gradient(eta_m, target_time)
    acceleration_m_per_s2 = np.gradient(rate_m_per_s, target_time)
    # A 10-minute moving average separates long tsunami motion from the
    # short-period component. Both remain direct transforms of the observation.
    window = max(5, int(n_steps / (hours * 6)))
    smooth = np.convolve(eta_m, np.ones(window) / window, mode='same')
    short_period_m = eta_m - smooth
    states = np.column_stack((
        _robust_scale(eta_m),
        _robust_scale(rate_m_per_s),
        _robust_scale(short_period_m),
        _robust_scale(acceleration_m_per_s2),
    ))
    angles = np.column_stack([np.arctan2(states[:, j], states[:, i]) for i, j in PAIRS])
    return states, angles, np.zeros(n_steps, dtype=np.int8), target_time, eta_m, short_period_m


def sphere_coordinates(latitude_angle, longitude_angle):
    """Map two pairwise angles to local Cartesian coordinates on S²."""
    latitude = np.arcsin(np.sin(latitude_angle))
    longitude = np.mod(longitude_angle, 2.0 * np.pi)
    c = np.cos(latitude)
    return np.column_stack((
        RADIUS * c * np.cos(longitude),
        RADIUS * c * np.sin(longitude),
        RADIUS * np.sin(latitude),
    )).astype(np.float32)


def embed_in_r4(local_points, axes):
    """Embed a local S² in the R⁴ hyperplane spanned by `axes`."""
    embedded = np.zeros((len(local_points), 4), dtype=np.float32)
    embedded[:, axes] = local_points
    return embedded


def rotate_r4(points, xw_degrees, yw_degrees, zw_degrees,
              xy_degrees=0, xz_degrees=0, yz_degrees=0):
    """Apply all six independent rotation planes of R⁴.

    The xy/xz/yz rotations are the familiar spatial rotations; xw/yw/zw are
    rotations involving the fourth axis.  Applying every plane here means the
    bottom controls act on the real R⁴ data before its final 3D projection.
    """
    result = np.asarray(points, dtype=np.float64).copy()
    for axis_a, axis_b, degrees in (
        (0, 1, xy_degrees), (0, 2, xz_degrees), (1, 2, yz_degrees),
        (0, 3, xw_degrees), (1, 3, yw_degrees), (2, 3, zw_degrees),
    ):
        angle = np.deg2rad(degrees)
        a, b = result[:, axis_a].copy(), result[:, axis_b].copy()
        result[:, axis_a] = np.cos(angle) * a - np.sin(angle) * b
        result[:, axis_b] = np.sin(angle) * a + np.cos(angle) * b
    return result


def project_r4_to_r3(points, xw_degrees=26, yw_degrees=-19, zw_degrees=14,
                     xy_degrees=0, xz_degrees=0, yz_degrees=0):
    """Rotate R⁴ and project along w; depth in w changes apparent 3D scale."""
    rotated = rotate_r4(
        points, xw_degrees, yw_degrees, zw_degrees,
        xy_degrees, xz_degrees, yz_degrees,
    )
    scale = PROJECTION_DISTANCE / np.maximum(0.25, PROJECTION_DISTANCE - rotated[:, 3])
    return (rotated[:, :3] * scale[:, None]).astype(np.float32)


def wireframe_r4(axes, parallels=10, meridians=14, points=48):
    lines = []
    longitudes = np.linspace(0.0, 2.0 * np.pi, points + 1)
    latitudes = np.linspace(-np.pi / 2.0, np.pi / 2.0, points + 1)
    for latitude in np.linspace(-np.pi / 2.0, np.pi / 2.0, parallels):
        lines.append(embed_in_r4(sphere_coordinates(np.full_like(longitudes, latitude), longitudes), axes))
    for longitude in np.linspace(0.0, 2.0 * np.pi, meridians, endpoint=False):
        lines.append(embed_in_r4(sphere_coordinates(latitudes, np.full_like(latitudes, longitude)), axes))
    return lines


def hypersphere_wireframe_r4(n_chi=4, n_fixed=4, points=40):
    """Sparse coordinate net for the one parent 3-sphere S³ in R⁴.

    Hopf-like coordinates give x=R cos(chi) cos(phi), y=R cos(chi) sin(phi),
    z=R sin(chi) cos(psi), w=R sin(chi) sin(psi).  Every returned point has
    norm R, so the neutral envelope really is one hypersphere, not a fourth
    ordinary 3D sphere.
    """
    lines = []
    phase = np.linspace(0.0, 2.0 * np.pi, points + 1)
    chis = np.linspace(0.14, np.pi / 2.0 - 0.14, n_chi)
    fixed = np.linspace(0.0, 2.0 * np.pi, n_fixed, endpoint=False)
    for chi in chis:
        for psi in fixed:
            lines.append(np.column_stack((
                RADIUS * np.cos(chi) * np.cos(phase),
                RADIUS * np.cos(chi) * np.sin(phase),
                RADIUS * np.sin(chi) * np.cos(psi) * np.ones_like(phase),
                RADIUS * np.sin(chi) * np.sin(psi) * np.ones_like(phase),
            )).astype(np.float32))
        for phi in fixed:
            lines.append(np.column_stack((
                RADIUS * np.cos(chi) * np.cos(phi) * np.ones_like(phase),
                RADIUS * np.cos(chi) * np.sin(phi) * np.ones_like(phase),
                RADIUS * np.sin(chi) * np.cos(phase),
                RADIUS * np.sin(chi) * np.sin(phase),
            )).astype(np.float32))
    return lines


class DiagnosticWidget(QtWidgets.QWidget):
    """Compact accessible text-and-plot diagnostic, independent of color."""

    def __init__(self, title, color, parent=None):
        super().__init__(parent)
        self.title, self.color = title, QtGui.QColor(*color)
        self.values = np.empty(0)
        self.index = 0
        self.setMinimumHeight(136)
        self.setStyleSheet('background:#0b1220; border:1px solid #26364c;')

    def set_series(self, values):
        self.values = np.asarray(values, dtype=np.float64)
        self.index = 0
        self.update()

    def set_frame(self, index):
        self.index = int(np.clip(index, 0, max(0, len(self.values) - 1)))
        self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#0b1220'))
        painter.setPen(QtGui.QPen(QtGui.QColor('#e6eefb'), 1))
        painter.drawText(12, 20, self.title)
        if len(self.values) < 2:
            painter.end()
            return
        rect = self.rect().adjusted(12, 31, -12, -22)
        painter.setPen(QtGui.QPen(QtGui.QColor('#26364c'), 1))
        painter.drawLine(rect.left(), rect.center().y(), rect.right(), rect.center().y())
        visible = self.values[:self.index + 1]
        lo, hi = float(np.min(self.values)), float(np.max(self.values))
        if abs(hi - lo) < 1e-9:
            lo, hi = lo - 1.0, hi + 1.0
        x = np.linspace(rect.left(), rect.right(), len(visible))
        y = rect.bottom() - (visible - lo) / (hi - lo) * rect.height()
        path = QtGui.QPainterPath(QtCore.QPointF(float(x[0]), float(y[0])))
        for px, py in zip(x[1:], y[1:]):
            path.lineTo(float(px), float(py))
        painter.setPen(QtGui.QPen(self.color, 2))
        painter.drawPath(path)
        cx, cy = float(x[-1]), float(y[-1])
        painter.setBrush(QtGui.QBrush(self.color))
        painter.drawEllipse(QtCore.QPointF(cx, cy), 3.5, 3.5)
        painter.setPen(QtGui.QPen(QtGui.QColor('#b9c7da'), 1))
        painter.drawText(12, self.height() - 6, f'current {visible[-1]:+.3f}     full range [{lo:+.3f}, {hi:+.3f}]')
        painter.end()


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Coupled Flood–Earthquake | 4D Phase Spheres')
        # Request a normal, decorated, resizable top-level window from the
        # window manager.  Do not use dialog/fixed-size flags here.
        self.setWindowFlags(
            QtCore.Qt.Window
            | QtCore.Qt.WindowTitleHint
            | QtCore.Qt.WindowSystemMenuHint
            | QtCore.Qt.WindowMinMaxButtonsHint
            | QtCore.Qt.WindowCloseButtonHint
        )
        self.setMinimumSize(640, 480)
        self.resize(1480, 980)
        self.n_steps, self.dt, self.play_stride, self.current_index = 2400, 0.012, 3, 0
        self._make_ui()
        self.regenerate()

    def _make_ui(self):
        central = QtWidgets.QWidget()
        central.setMinimumSize(1, 1)
        central.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        central.setStyleSheet('background:#020617; color:#e6eefb;')
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(8)

        heading = QtWidgets.QLabel('COUPLED FLOOD–EARTHQUAKE  •  ONE PARENT S³ HYPERSPHERE CONTAINING THREE GREAT S² SPHERES')
        heading.setStyleSheet('font-size:15px; font-weight:700; letter-spacing:1px; color:#f8fafc; padding:5px 2px;')
        layout.addWidget(heading)
        explainer = QtWidgets.QLabel(
            'Deterministic explicit-Euler modal reduction: X = (surface η, flood momentum q, bed displacement uᵧ, seismic velocity vᵧ). '
            'The neutral wire envelope is one S³ hypersphere. Inside it, the coloured great spheres share one origin in R⁴: S²₁⊂xyz, S²₂⊂xyw, S²₃⊂xzw. The six plane controls rotate the entire construction before w-perspective projection.'
        )
        explainer.setWordWrap(True)
        explainer.setStyleSheet('background:#0f172a; border:1px solid #26364c; color:#cbd5e1; padding:8px; font-size:11px;')
        layout.addWidget(explainer)
        legend = QtWidgets.QLabel(
            '<b style="color:#94a3b8">S³ — parent hypersphere</b> &nbsp; | &nbsp; '
            '<b style="color:#24c7ff">S²₁ / xyz — flood-angle sphere</b> &nbsp; | &nbsp; '
            '<b style="color:#fa853a">S²₂ / xyw — cross-angle sphere</b> &nbsp; | &nbsp; '
            '<b style="color:#66e88d">S²₃ / xzw — cross-angle sphere</b>'
            ' &nbsp; • &nbsp; <b>trajectory:</b> square = start, bright fading tail = recent direction, white-edged disc = current point'
        )
        legend.setTextFormat(QtCore.Qt.RichText)
        legend.setStyleSheet('background:#08101d; border:1px solid #26364c; padding:6px 8px; font-size:11px;')
        layout.addWidget(legend)
        viewport = QtWidgets.QWidget()
        viewport.setMinimumSize(1, 1)
        viewport.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        viewport_layout = QtWidgets.QHBoxLayout(viewport)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(8)
        layout.addWidget(viewport, stretch=8)
        self.canvas = SceneCanvas(size=(1040, 760), bgcolor='#020617', keys='interactive')
        self.canvas.native.setParent(viewport)
        self.canvas.native.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas.native.setMinimumSize(1, 1)
        self.canvas.native.setMaximumSize(16777215, 16777215)
        viewport_layout.addWidget(self.canvas.native, stretch=7)
        self.viewport_layout = viewport_layout
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(elevation=17, azimuth=22, distance=4.3, fov=44)
        self.view.camera.set_range(x=(-1.6, 1.6), y=(-1.6, 1.6), z=(-1.6, 1.6))

        self.hypersphere_r4 = hypersphere_wireframe_r4()
        self.hypersphere_lines = [
            Line(color=(0.58, 0.64, 0.74, 0.19), width=0.9, connect='strip', parent=self.view.scene)
            for _ in self.hypersphere_r4
        ]
        self.wireframes_r4 = []
        self.wire_colors = []
        self.wire_surface_ids = []
        for surface_id, ((_, _, _, axes), grid_color) in enumerate(zip(SPHERES, GRID_COLORS)):
            lines = wireframe_r4(axes)
            self.wireframes_r4.extend(lines)
            self.wire_colors.extend([grid_color] * len(lines))
            self.wire_surface_ids.extend([surface_id] * len(lines))
        self.wire_lines = [
            Line(color=color, width=1.25, connect='strip', parent=self.view.scene)
            for color in self.wire_colors
        ]
        self.axis_lines_r4 = []
        for dimension in range(4):
            axis = np.zeros((2, 4), dtype=np.float32)
            axis[0, dimension], axis[1, dimension] = -1.12, 1.12
            self.axis_lines_r4.append(axis)
        self.axis_lines = [
            Line(color=color, width=2, connect='strip', parent=self.view.scene)
            for color in ((0.55, 0.75, 1.0, 0.72), (1.0, 0.66, 0.36, 0.72), (0.48, 0.92, 0.64, 0.72), (0.89, 0.45, 0.95, 0.72))
        ]
        self.axis_labels = [
            Text(label, color=color, font_size=10, anchor_x='center', anchor_y='center', parent=self.view.scene)
            for label, color in zip(
                ('X₁ = η', 'X₂ = q', 'X₃ = uᵧ', 'X₄ = vᵧ'),
                ((0.55, 0.75, 1.0, 1.0), (1.0, 0.66, 0.36, 1.0), (0.48, 0.92, 0.64, 1.0), (0.89, 0.45, 0.95, 1.0)),
            )
        ]
        self.paths = [Line(color=color, width=1.4, connect='strip', parent=self.view.scene) for color in COLORS]
        self.trails = [Line(color=color, width=4.2, connect='strip', parent=self.view.scene) for color in COLORS]
        self.starts = [Markers(parent=self.view.scene) for _ in COLORS]
        self.heads = [Markers(parent=self.view.scene) for _ in COLORS]
        self.swap_heads = [Markers(parent=self.view.scene) for _ in COLORS]

        panels = QtWidgets.QWidget()
        panels.setMinimumWidth(480)
        panels.setMaximumWidth(620)
        panels_layout = QtWidgets.QVBoxLayout(panels)
        panels_layout.setContentsMargins(0, 0, 0, 0)
        panels_layout.setSpacing(8)
        self.flood_trace = DiagnosticWidget('FLOOD MODE   η (cyan)', (36, 199, 255))
        self.bed_trace = DiagnosticWidget('BED MODE   uᵧ (orange)', (250, 133, 58))
        self.schedule_trace = DiagnosticWidget('BINARY SCHEDULE   sₖ (green)', (102, 232, 141))
        for panel in (self.flood_trace, self.bed_trace, self.schedule_trace):
            panels_layout.addWidget(panel)
        self.viewport_layout.insertWidget(0, panels, stretch=2)

        controls = QtWidgets.QWidget()
        controls.setStyleSheet('QPushButton, QComboBox { background:#172554; border:1px solid #3b82f6; border-radius:4px; padding:6px 10px; color:#f8fafc; } QPushButton:pressed { background:#1d4ed8; }')
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(5)
        row = QtWidgets.QHBoxLayout()
        controls_layout.addLayout(row)
        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.setAccessibleName('Play or pause animation')
        self.play_button.clicked.connect(self.toggle_play)
        row.addWidget(self.play_button)
        step = QtWidgets.QPushButton('Step')
        step.setAccessibleName('Advance one visualization frame')
        step.clicked.connect(self.step_once)
        row.addWidget(step)
        reset = QtWidgets.QPushButton('Reset')
        reset.clicked.connect(lambda: self.update_frame(0))
        row.addWidget(reset)
        row.addSpacing(12)
        row.addWidget(QtWidgets.QLabel('data source:'))
        self.data_box = QtWidgets.QComboBox()
        self.data_box.setAccessibleName('Choose simulated or observational data source')
        self.data_box.addItems(('Synthetic modal example', '2011 Tōhoku — NOAA DART 21418 observations'))
        self.data_box.setCurrentIndex(1)
        row.addWidget(self.data_box)
        row.addWidget(QtWidgets.QLabel('swap cadence:'))
        self.period_box = QtWidgets.QComboBox()
        self.period_box.addItems(('never', 'every 12 steps', 'every 6 steps', 'every 3 steps', 'every step'))
        self.period_box.setCurrentIndex(2)
        row.addWidget(self.period_box)
        row.addWidget(QtWidgets.QLabel('permutation:'))
        self.permutation_box = QtWidgets.QComboBox()
        self.permutation_box.addItems(('η ↔ q  (flood)', 'q ↔ uᵧ  (interface)', 'uᵧ ↔ vᵧ  (seismic)'))
        row.addWidget(self.permutation_box)
        regenerate = QtWidgets.QPushButton('Regenerate deterministic run')
        regenerate.clicked.connect(self.regenerate)
        row.addWidget(regenerate)
        projection_row = QtWidgets.QHBoxLayout()
        controls_layout.addLayout(projection_row)
        projection_row.addWidget(QtWidgets.QLabel('shared R⁴ projection:'))
        self.rotation_sliders = {}
        for label, initial in (('xw', 26), ('yw', -19), ('zw', 14)):
            projection_row.addWidget(QtWidgets.QLabel(f'{label} rotation:'))
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setAccessibleName(f'{label} four-dimensional rotation angle')
            slider.setRange(-180, 180)
            slider.setValue(initial)
            slider.setFixedWidth(76)
            slider.valueChanged.connect(self.update_projection)
            projection_row.addWidget(slider)
            self.rotation_sliders[label] = slider
        projection_row.addWidget(QtWidgets.QLabel('surface focus:'))
        self.focus_box = QtWidgets.QComboBox()
        self.focus_box.setAccessibleName('Select hyperplane surface to emphasize')
        self.focus_box.addItems(('All great spheres (compare)', 'Focus S²₁ / xyz', 'Focus S²₂ / xyw', 'Focus S²₃ / xzw'))
        self.focus_box.setCurrentIndex(1)
        self.focus_box.currentIndexChanged.connect(self.update_projection)
        projection_row.addWidget(self.focus_box)
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setAccessibleName('Timeline frame')
        self.slider.valueChanged.connect(self.on_slider)
        projection_row.addWidget(self.slider, stretch=1)
        self.status = QtWidgets.QLabel()
        self.status.setMinimumWidth(360)
        self.status.setStyleSheet('color:#cbd5e1; padding-left:8px;')
        projection_row.addWidget(self.status)
        spatial_row = QtWidgets.QHBoxLayout()
        controls_layout.addLayout(spatial_row)
        spatial_row.addWidget(QtWidgets.QLabel('spatial planes in the same R⁴ transform:'))
        for label in ('xy', 'xz', 'yz'):
            spatial_row.addWidget(QtWidgets.QLabel(f'{label} rotation:'))
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setAccessibleName(f'{label} spatial rotation angle')
            slider.setRange(-180, 180)
            slider.setValue(0)
            slider.setFixedWidth(130)
            slider.valueChanged.connect(self.update_projection)
            spatial_row.addWidget(slider)
            self.rotation_sliders[label] = slider
        spatial_row.addStretch(1)
        layout.addWidget(controls)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(24)
        self.timer.timeout.connect(self.advance)

    def _period(self):
        return (0, 12, 6, 3, 1)[self.period_box.currentIndex()]

    def regenerate(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.observed_eta = None
        self.observed_short_period = None
        if self.data_box.currentIndex() == 1:
            try:
                (self.states, self.angles, self.schedule, time_seconds,
                 self.observed_eta, self.observed_short_period) = load_tohoku_dart_21418(self.n_steps)
                self.time = time_seconds / 60.0
                self.time_unit = 'min'
                self.state_display = ('η_obs', 'dη/dt', 'η_short')
                self.flood_trace.title = 'OBSERVED TSUNAMI RESIDUAL η  (metres water column)'
                self.bed_trace.title = 'OBSERVED SHORT-PERIOD WATER COMPONENT  (metres)'
                self.schedule_trace.title = 'NO SWAP SCHEDULE — observational trajectory'
            except RuntimeError as exc:
                QtWidgets.QMessageBox.warning(self, 'NOAA data unavailable', f'{exc}\n\nReverting to the synthetic modal example.')
                self.data_box.blockSignals(True)
                self.data_box.setCurrentIndex(0)
                self.data_box.blockSignals(False)
        if self.data_box.currentIndex() == 0:
            self.states, self.angles, self.schedule, self.time = simulate(
                self.n_steps, self.dt, self._period(), self.permutation_box.currentIndex()
            )
            self.time_unit = 's'
            self.state_display = ('η', 'q', 'uᵧ')
            self.flood_trace.title = 'FLOOD MODE   η (normalized)'
            self.bed_trace.title = 'BED/INTERFACE MODE   uᵧ (normalized)'
            self.schedule_trace.title = 'BINARY SWAP SCHEDULE   sₖ'
        # The three paths occupy different R⁴ hyperplanes, not different 3D locations.
        self.sphere_paths_r4 = [
            embed_in_r4(sphere_coordinates(self.angles[:, pair[0]], self.angles[:, pair[1]]), axes)
            for (_, pair, _, axes) in SPHERES
        ]
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.n_steps)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.flood_trace.set_series(self.observed_eta if self.observed_eta is not None else self.states[:, 0])
        self.bed_trace.set_series(self.observed_short_period if self.observed_short_period is not None else self.states[:, 2])
        self.schedule_trace.set_series(np.r_[self.schedule, self.schedule[-1]])
        self.update_frame(0)

    def _project(self, points_r4):
        return project_r4_to_r3(
            points_r4,
            self.rotation_sliders['xw'].value(),
            self.rotation_sliders['yw'].value(),
            self.rotation_sliders['zw'].value(),
            self.rotation_sliders['xy'].value(),
            self.rotation_sliders['xz'].value(),
            self.rotation_sliders['yz'].value(),
        )

    def update_projection(self, *_unused):
        """Refresh the scene, keeping one chosen S² legible amid 4D overlap."""
        if not hasattr(self, 'sphere_paths_r4'):
            return
        focus = self.focus_box.currentIndex() - 1  # -1 means comparison mode.
        for line, raw in zip(self.hypersphere_lines, self.hypersphere_r4):
            line.set_data(pos=self._project(raw))
        for line, raw, surface_id in zip(self.wire_lines, self.wireframes_r4, self.wire_surface_ids):
            is_focus = focus < 0 or surface_id == focus
            alpha = 0.28 if focus < 0 else (0.86 if is_focus else 0.055)
            color = (*GRID_COLORS[surface_id][:3], alpha)
            line.set_data(pos=self._project(raw), color=color)
        for line, label, raw in zip(self.axis_lines, self.axis_labels, self.axis_lines_r4):
            projected_axis = self._project(raw)
            line.set_data(pos=projected_axis)
            label.pos = projected_axis[1] * 1.08
        for surface_id, (path, trail, start, head, flash, raw_path, color) in enumerate(zip(
            self.paths, self.trails, self.starts, self.heads, self.swap_heads, self.sphere_paths_r4, COLORS
        )):
            is_focus = focus < 0 or surface_id == focus
            path_alpha = 0.23 if focus < 0 else (0.30 if is_focus else 0.035)
            head_alpha = 0.92 if focus < 0 else (1.0 if is_focus else 0.20)
            visible = self._project(raw_path[:self.current_index + 1])
            path.set_data(pos=visible, color=(*color[:3], path_alpha))
            tail = visible[max(0, len(visible) - TRAIL_LENGTH):]
            tail_opacity = np.linspace(0.12, 1.0, len(tail), dtype=np.float32)
            tail_opacity *= 0.74 if focus < 0 else (1.0 if is_focus else 0.10)
            tail_colors = np.column_stack((
                np.full(len(tail), color[0]), np.full(len(tail), color[1]),
                np.full(len(tail), color[2]), tail_opacity,
            )).astype(np.float32)
            trail.set_data(pos=tail, color=tail_colors)
            start.set_data(
                pos=visible[:1], face_color=(0.04, 0.07, 0.12, head_alpha),
                edge_color=(*color[:3], head_alpha), size=11, symbol='square',
            )
            head.set_data(
                pos=visible[-1:], face_color=(*color[:3], head_alpha),
                edge_color=(0.96, 0.98, 1.0, head_alpha), size=12, symbol='disc',
            )
            is_swap = self.current_index > 0 and bool(self.schedule[self.current_index - 1]) and is_focus
            flash.set_data(
                pos=visible[-1:],
                face_color=(1.0, 0.2, 0.2, 0.92) if is_swap else (0, 0, 0, 0),
                size=17,
            )
        self.canvas.update()

    def on_slider(self, value):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(value)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play')
        else:
            if self.current_index >= self.n_steps:
                self.update_frame(0)
            self.timer.start()
            self.play_button.setText('Pause')

    def step_once(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(min(self.current_index + 1, self.n_steps))

    def advance(self):
        next_index = min(self.current_index + self.play_stride, self.n_steps)
        self.update_frame(next_index)
        if next_index >= self.n_steps:
            self.timer.stop()
            self.play_button.setText('Play')

    def update_frame(self, index):
        self.current_index = int(np.clip(index, 0, self.n_steps))
        self.flood_trace.set_frame(self.current_index)
        self.bed_trace.set_frame(self.current_index)
        self.schedule_trace.set_frame(self.current_index)
        current_swap = int(self.schedule[max(0, self.current_index - 1)]) if self.current_index else 0
        x = self.states[self.current_index]
        tail_start = max(0, self.current_index + 1 - TRAIL_LENGTH)
        tail_span = self.time[self.current_index] - self.time[tail_start]
        self.status.setText(
            f'k={self.current_index:04d}/{self.n_steps}  t={self.time[self.current_index]:5.2f}{self.time_unit}  '
            f'Cₙ={self.schedule.mean():.3f}  sₖ={current_swap}  '
            f'{self.state_display[0]}={x[0]:+.2f}  {self.state_display[1]}={x[1]:+.2f}  {self.state_display[2]}={x[2]:+.2f}  '
            f'bright tail = {tail_span:.2f}{self.time_unit}'
        )
        self.slider.blockSignals(True)
        self.slider.setValue(self.current_index)
        self.slider.blockSignals(False)
        self.update_projection()


if __name__ == '__main__':
    qt_app = QtWidgets.QApplication(sys.argv)
    app.use_app('pyqt5')
    window = MainWindow()
    window.show()
    sys.exit(qt_app.exec_())
