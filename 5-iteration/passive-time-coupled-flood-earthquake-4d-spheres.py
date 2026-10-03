"""Passive-time coupled flood--earthquake visualization with three S2 spheres.

The physical model uses two separate four-dimensional Clifford sectors:

    flood:      F = (t1, x1, t2, x2)
    earthquake: S = (t1, x1, t3, x3)
    interface:  C = (t1, x1)

The first-order modal state is

    x = (eta, h*v1, h*v2, u1, u3, du1/dt1, du3/dt1),  h = 1 + eta.

It contains a bidirectional physical interface: u3 changes the x1 bed slope
seen by the flood, and the flood depth anomaly supplies d_x1(Lambda*g*h) to
the vertical seismic equation.  In addition, the deterministic coupling
schedule applies the stated explicit-Euler permutation exactly:

    x[k+1] = x[k] + dt1 * ((1-s[k])*I + s[k]*P) F(x[k]).

For graphing, each S2 is controlled by two circular phase coordinates:

    flood sphere:      (t1,x1) circle + (t2,x2) circle
    earthquake sphere: (t1,x1) circle + (t3,x3) circle
    coupled sphere:    flood-on-C circle + earthquake-on-C circle

The three S2 surfaces occupy the xyz, xyw, and xzw hyperplanes of one parent
S3 hypersphere in R4.  One shared six-plane R4 rotation and perspective map
projects the complete construction into the visible three-dimensional scene.

This is a stable one-mode visualization of the equations, not a spatially
resolved shallow-water/elastodynamic PDE solver.

Requirements: numpy, PyQt5, vispy
Run: .venv/bin/python passive-time-coupled-flood-earthquake-4d-spheres.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Line, Markers


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / 'coupled-flood-earthquake-4d-spheres.py'
SPEC = importlib.util.spec_from_file_location('passive_time_sphere_base', BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f'Cannot load shared visualization helpers from {BASE_PATH}.')
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)


ETA, P1, P2, U1, U3, W1, W3 = range(7)
STATE_NAMES = ('eta', 'hv1', 'hv2', 'u1', 'u3', 'du1/dt1', 'du3/dt1')

# A fixed involutory permutation (P^2=I) for the discrete coupling operator.
# It exchanges two complementary flood/seismic field channels while the
# physical p1 <-> u3 interface remains continuously active in F(x).
COUPLING_PERMUTATION = np.array((W1, P1, U1, P2, U3, ETA, W3))

SPHERE_NAMES = ('FLOOD  F', 'EARTHQUAKE  S', 'COUPLED  C')
SPHERE_COLORS = (
    (0.14, 0.78, 1.00, 1.0),
    (0.98, 0.58, 0.18, 1.0),
    (0.20, 0.90, 0.48, 1.0),
)
SPHERE_AXES = ((0, 1, 2), (0, 1, 3), (0, 2, 3))
SPHERE_RADIUS = base.RADIUS
TRAIL_LENGTH = 260


def deterministic_schedule(n_steps, period):
    """Return s_k with a one on each selected periodic coupling step."""
    if period <= 0:
        return np.zeros(n_steps, dtype=np.int8)
    return ((np.arange(1, n_steps + 1) % period) == 0).astype(np.int8)


def passive_time_modal_field(state):
    """Return F(x) and d_x1(Lambda*g*h) for the passive-time modal system.

    The constant Lambda mode satisfies the visualized scalar adjoint
    constraint.  Its product with the spatial depth mode still has a nonzero
    x1 derivative, which drives the vertical seismic component.
    """
    eta, p1, p2, u1, u3, w1, w3 = state
    depth = 1.0 + eta
    gravity, density = 1.0, 1.0
    k1, k2, k3 = 0.78, 0.46, 0.62
    lame_lambda, shear_mu = 0.42, 0.26
    flood_damping_1, flood_damping_2 = 0.060, 0.045
    seismic_damping = 0.055
    bed_to_flood = 0.22
    lambda_mode = 0.34

    # One real Fourier/Galerkin quadrature of the two-dimensional flood PDE.
    d_eta = -k1 * p1 - k2 * p2
    d_p1 = (
        gravity * k1 * eta
        + bed_to_flood * gravity * depth * k1 * u3
        - flood_damping_1 * p1
    )
    d_p2 = gravity * k2 * eta - flood_damping_2 * p2

    # Only t1 derivatives occur.  u1 and u3 are lifted to a first-order
    # explicit-Euler system with w_i = d_t1 u_i.
    wave_number_sq = k1**2 + k3**2
    longitudinal = lame_lambda + 2.0 * shear_mu
    stiffness_11 = longitudinal * k1**2 + shear_mu * wave_number_sq
    stiffness_33 = longitudinal * k3**2 + shear_mu * wave_number_sq
    stiffness_13 = longitudinal * k1 * k3
    flood_force = lambda_mode * gravity * k1 * eta
    acceleration_u1 = (
        -stiffness_11 * u1 - stiffness_13 * u3
    ) / density - seismic_damping * w1
    acceleration_u3 = (
        -stiffness_13 * u1 - stiffness_33 * u3 + flood_force
    ) / density - seismic_damping * w3

    field = np.array((
        d_eta, d_p1, d_p2, w1, w3, acceleration_u1, acceleration_u3,
    ), dtype=np.float64)
    return field, flood_force


def _sphere_coordinates(first_circle_angle, second_circle_angle):
    """Map two circular phase coordinates to one S2 trajectory."""
    latitude = np.arcsin(np.sin(first_circle_angle))
    longitude = np.mod(second_circle_angle, 2.0 * np.pi)
    latitude_radius = np.cos(latitude)
    return np.column_stack((
        SPHERE_RADIUS * latitude_radius * np.cos(longitude),
        SPHERE_RADIUS * latitude_radius * np.sin(longitude),
        SPHERE_RADIUS * np.sin(latitude),
    )).astype(np.float32)


def sphere_phase_paths(states):
    """Build the flood, earthquake, and interface two-circle paths."""
    eta = states[:, ETA]
    depth = 1.0 + eta
    p1, p2 = states[:, P1], states[:, P2]
    u1, u3 = states[:, U1], states[:, U3]
    w1, w3 = states[:, W1], states[:, W3]

    flood_shared = np.arctan2(p1, depth)
    flood_passive = np.arctan2(p2, depth)
    seismic_shared = np.arctan2(u1, w1)
    seismic_passive = np.arctan2(u3, w3)

    return (
        _sphere_coordinates(flood_shared, flood_passive),
        _sphere_coordinates(seismic_shared, seismic_passive),
        _sphere_coordinates(flood_shared, seismic_passive),
    )


def simulate_passive_time_system(n_steps=2400, dt=0.012, period=6):
    """Integrate the physical field and scheduled permutation in t1."""
    if n_steps < 1:
        raise ValueError('n_steps must be at least 1.')
    if dt <= 0.0:
        raise ValueError('dt must be positive.')

    schedule = deterministic_schedule(n_steps, period)
    states = np.empty((n_steps + 1, 7), dtype=np.float64)
    forcing = np.empty(n_steps + 1, dtype=np.float64)
    states[0] = (-0.28, 0.10, -0.055, 0.08, -0.22, 0.025, 0.10)

    for step, swapped in enumerate(schedule):
        field, forcing[step] = passive_time_modal_field(states[step])
        if swapped:
            field = field[COUPLING_PERMUTATION]
        states[step + 1] = states[step] + dt * field

    _, forcing[-1] = passive_time_modal_field(states[-1])
    paths = sphere_phase_paths(states)
    time = np.arange(n_steps + 1, dtype=np.float64) * dt
    return states, paths, forcing, schedule, time


def _great_circles(samples=181):
    """Return exactly two orthogonal circles used to depict each S2."""
    angle = np.linspace(0.0, 2.0 * np.pi, samples, dtype=np.float32)
    zeros = np.zeros_like(angle)
    circle_t_space = SPHERE_RADIUS * np.column_stack((np.cos(angle), np.sin(angle), zeros))
    circle_extra = SPHERE_RADIUS * np.column_stack((np.cos(angle), zeros, np.sin(angle)))
    return circle_t_space.astype(np.float32), circle_extra.astype(np.float32)


class RecentScheduleWidget(QtWidgets.QWidget):
    """Draw recent binary coupling events as distinct vertical bars."""

    def __init__(self, title, color, visible_steps=96):
        super().__init__()
        self.title = title
        self.color = QtGui.QColor(*color)
        self.visible_steps = visible_steps
        self.values = np.zeros(1, dtype=np.float64)
        self.current = 0
        self.setMinimumHeight(112)
        self.setAccessibleName('Recent binary coupling schedule')

    def set_series(self, values):
        self.values = np.asarray(values, dtype=np.float64)
        self.current = 0
        self.update()

    def set_frame(self, index):
        self.current = int(np.clip(index, 0, max(0, len(self.values) - 1)))
        self.update()

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#0b1220'))
        painter.setPen(QtGui.QColor('#e2e8f0'))
        painter.drawText(12, 20, self.title)

        left, right = 12.0, float(max(13, self.width() - 12))
        top, baseline = 30.0, float(max(31, self.height() - 24))
        painter.setPen(QtGui.QPen(QtGui.QColor('#233047'), 1))
        painter.drawLine(QtCore.QPointF(left, baseline), QtCore.QPointF(right, baseline))

        end = self.current + 1
        start = max(0, end - self.visible_steps)
        recent = self.values[start:end]
        if len(recent):
            step_width = (right - left) / max(1, len(recent))
            painter.setPen(QtGui.QPen(self.color, max(1.4, min(4.0, step_width * 0.58))))
            for offset in np.flatnonzero(recent > 0.5):
                x = left + (offset + 0.5) * step_width
                painter.drawLine(QtCore.QPointF(x, baseline), QtCore.QPointF(x, top))

        current_value = int(self.values[self.current] > 0.5) if len(self.values) else 0
        painter.setPen(QtGui.QColor('#b9c7da'))
        painter.drawText(
            12, self.height() - 6,
            f'current sₖ={current_value}     showing steps {start:04d}–{max(start, end - 1):04d}',
        )
        painter.end()


class DirectSizeGrip(QtWidgets.QWidget):
    """Bottom-right grip that resizes the top-level window without WM help."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._drag_origin = None
        self._start_size = None
        self.setCursor(QtCore.Qt.SizeFDiagCursor)
        self.setFixedSize(22, 22)
        self.setAccessibleName('Drag to resize the application window')

    def mousePressEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and not self.window().isMaximized():
            self._drag_origin = event.globalPos()
            self._start_size = self.window().size()
            self.grabMouse()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_origin is not None and self._start_size is not None:
            delta = event.globalPos() - self._drag_origin
            requested = self._start_size + QtCore.QSize(delta.x(), delta.y())
            requested = requested.expandedTo(self.window().minimumSize())
            requested = requested.boundedTo(self.window().maximumSize())
            self.window().resize(requested)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == QtCore.Qt.LeftButton and self._drag_origin is not None:
            self._drag_origin = None
            self._start_size = None
            self.releaseMouse()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.setPen(QtGui.QPen(QtGui.QColor('#93c5fd'), 1.5))
        edge = self.width() - 3
        for inset in (0, 5, 10):
            painter.drawLine(edge - inset, edge, edge, edge - inset)
        painter.end()


class PassiveTimeCoupledWindow(QtWidgets.QMainWindow):
    """Three-sphere desktop viewer for the passive-time coupled equations."""

    def __init__(self):
        super().__init__()
        self.n_steps = 2400
        self.dt = 0.012
        self.play_stride = 3
        self.current_index = 0
        self._make_ui()
        self.regenerate()

    def _make_ui(self):
        self.setWindowTitle('Passive-Time Coupled Flood–Earthquake | Three S² Surfaces in One S³')
        # Request a normal decorated window with working minimize, maximize,
        # and resize behavior.  These explicit flags avoid the fixed/dialog
        # behavior seen with some Qt window-manager defaults.
        self.setWindowFlags(
            QtCore.Qt.Window
            | QtCore.Qt.WindowTitleHint
            | QtCore.Qt.WindowSystemMenuHint
            | QtCore.Qt.WindowMinimizeButtonHint
            | QtCore.Qt.WindowMaximizeButtonHint
            | QtCore.Qt.WindowCloseButtonHint
        )
        self.setMinimumSize(640, 480)
        self.setMaximumSize(QtWidgets.QWIDGETSIZE_MAX, QtWidgets.QWIDGETSIZE_MAX)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.resize(1540, 960)

        central = QtWidgets.QWidget()
        central.setMinimumSize(1, 1)
        central.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        central.setStyleSheet('background:#020617; color:#f8fafc;')
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        heading = QtWidgets.QLabel(
            'PASSIVE-TIME COUPLED FLOOD–EARTHQUAKE  •  THREE S² SURFACES IN ONE PARENT S³  •  C = (t₁, x₁)'
        )
        heading.setStyleSheet(
            'font-size:16px; font-weight:700; letter-spacing:1px; color:#f8fafc; padding:4px 2px;'
        )
        layout.addWidget(heading)

        explainer = QtWidgets.QLabel(
            'The flood receives the moving-bed slope from u₃ at x₃=0, while the vertical earthquake mode '
            'receives +∂ₓ₁(Λgh). On scheduled steps, the fixed P exchanges selected components of F(xₖ) '
            'before the explicit-Euler t₁ update; here P swaps η↔∂ₜ₁u₁ and hv₂↔u₁.'
        )
        explainer.setWordWrap(True)
        explainer.setStyleSheet(
            'background:#0f172a; border:1px solid #334155; border-radius:4px; '
            'color:#dbeafe; padding:9px; font-size:12px;'
        )
        layout.addWidget(explainer)

        circle_legend = QtWidgets.QLabel(
            '<b style="color:#94a3b8">PARENT S³</b>: one R⁴ hypersphere'
            ' &nbsp; | &nbsp; <b style="color:#24c7ff">FLOOD F / xyz</b>: (t₁,x₁) + (t₂,x₂)'
            ' &nbsp; | &nbsp; <b style="color:#fa9430">EARTHQUAKE S / xyw</b>: (t₁,x₁) + (t₃,x₃)'
            ' &nbsp; | &nbsp; <b style="color:#33e67a">COUPLED C / xzw</b>: flood-on-C + earthquake-on-C'
        )
        circle_legend.setTextFormat(QtCore.Qt.RichText)
        circle_legend.setWordWrap(True)
        circle_legend.setStyleSheet(
            'background:#08101d; border:1px solid #334155; color:#e2e8f0; padding:7px 9px; font-size:12px;'
        )
        layout.addWidget(circle_legend)

        viewport = QtWidgets.QWidget()
        viewport.setMinimumSize(1, 1)
        viewport.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        viewport_layout = QtWidgets.QHBoxLayout(viewport)
        viewport_layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        viewport_layout.setContentsMargins(0, 0, 0, 0)
        viewport_layout.setSpacing(8)
        layout.addWidget(viewport, stretch=8)

        diagnostics = QtWidgets.QWidget()
        diagnostics.setMinimumWidth(220)
        diagnostics.setMaximumWidth(430)
        diagnostics.setSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding)
        diagnostics_layout = QtWidgets.QVBoxLayout(diagnostics)
        diagnostics_layout.setContentsMargins(0, 0, 0, 0)
        diagnostics_layout.setSpacing(8)
        self.flood_trace = base.DiagnosticWidget('FLOOD DEPTH  h', (36, 199, 255))
        self.earthquake_trace = base.DiagnosticWidget('EARTHQUAKE UPLIFT  u₃ | x₃=0', (250, 148, 48))
        self.schedule_trace = RecentScheduleWidget('COUPLING SCHEDULE  sₖ  (recent steps)', (51, 230, 122))
        for panel in (self.flood_trace, self.earthquake_trace, self.schedule_trace):
            diagnostics_layout.addWidget(panel)
        viewport_layout.addWidget(diagnostics, stretch=2)

        canvas_panel = QtWidgets.QWidget()
        canvas_panel.setMinimumSize(1, 1)
        canvas_panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        canvas_layout = QtWidgets.QVBoxLayout(canvas_panel)
        canvas_layout.setContentsMargins(0, 0, 0, 0)
        canvas_layout.setSpacing(4)
        self.hypersphere_title = QtWidgets.QLabel(
            '<b style="color:#94a3b8">ONE PARENT S³</b> &nbsp; • &nbsp; '
            '<span style="color:#24c7ff">FLOOD F</span> &nbsp; • &nbsp; '
            '<span style="color:#fa9430">EARTHQUAKE S</span> &nbsp; • &nbsp; '
            '<span style="color:#33e67a">COUPLED C</span>'
        )
        self.hypersphere_title.setTextFormat(QtCore.Qt.RichText)
        self.hypersphere_title.setAlignment(QtCore.Qt.AlignCenter)
        self.hypersphere_title.setAccessibleName(
            'One parent hypersphere containing flood, earthquake, and coupled surfaces'
        )
        self.hypersphere_title.setStyleSheet(
            'font-size:13px; font-weight:700; letter-spacing:1px; padding:3px;'
        )
        canvas_layout.addWidget(self.hypersphere_title)

        self.canvas = SceneCanvas(size=(1100, 690), bgcolor='#020617', keys='interactive')
        self.canvas.native.setParent(canvas_panel)
        self.canvas.native.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.canvas.native.setMinimumSize(1, 1)
        self.canvas.native.setMaximumSize(QtWidgets.QWIDGETSIZE_MAX, QtWidgets.QWIDGETSIZE_MAX)
        canvas_layout.addWidget(self.canvas.native, stretch=1)
        viewport_layout.addWidget(canvas_panel, stretch=7)
        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(elevation=17, azimuth=22, distance=4.3, fov=44)
        self.view.camera.set_range(x=(-1.6, 1.6), y=(-1.6, 1.6), z=(-1.6, 1.6))

        self.hypersphere_geometry = base.hypersphere_wireframe_r4()
        self.hypersphere_lines = [
            Line(color=(0.58, 0.64, 0.74, 0.20), width=0.9, connect='strip', parent=self.view.scene)
            for _ in self.hypersphere_geometry
        ]

        local_circles = _great_circles()
        self.circle_geometry = []
        self.circle_lines = []
        for sphere_id, color in enumerate(SPHERE_COLORS):
            for circle_id, points in enumerate(local_circles):
                points_r4 = base.embed_in_r4(points, SPHERE_AXES[sphere_id])
                self.circle_geometry.append((sphere_id, circle_id, points_r4))
                brightness = 1.0 if circle_id == 0 else 0.72
                circle_color = tuple(channel * brightness for channel in color[:3]) + (0.86,)
                self.circle_lines.append(
                    Line(color=circle_color, width=2.0, connect='strip', parent=self.view.scene)
                )

        self.paths = [
            Line(color=(*color[:3], 0.28), width=1.5, connect='strip', parent=self.view.scene)
            for color in SPHERE_COLORS
        ]
        self.trails = [
            Line(color=color, width=4.0, connect='strip', parent=self.view.scene)
            for color in SPHERE_COLORS
        ]
        self.starts = [Markers(parent=self.view.scene) for _ in SPHERE_COLORS]
        self.heads = [Markers(parent=self.view.scene) for _ in SPHERE_COLORS]
        self.swap_heads = [Markers(parent=self.view.scene) for _ in SPHERE_COLORS]

        controls = QtWidgets.QWidget()
        controls.setStyleSheet(
            'QPushButton, QComboBox { background:#172554; border:1px solid #3b82f6; '
            'border-radius:4px; min-height:30px; padding:3px 10px; color:#f8fafc; font-size:12px; } '
            'QPushButton:hover, QComboBox:hover { background:#1e3a8a; } '
            'QPushButton:focus, QComboBox:focus { border:2px solid #93c5fd; } '
            'QPushButton:pressed { background:#1d4ed8; }'
        )
        controls_layout = QtWidgets.QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(6)

        action_row = QtWidgets.QHBoxLayout()
        action_row.setSpacing(8)
        controls_layout.addLayout(action_row)
        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.setAccessibleName('Play or pause the coupled simulation')
        self.play_button.clicked.connect(self.toggle_play)
        action_row.addWidget(self.play_button)
        step_button = QtWidgets.QPushButton('Step')
        step_button.setAccessibleName('Advance one simulation step')
        step_button.clicked.connect(self.step_once)
        action_row.addWidget(step_button)
        reset_button = QtWidgets.QPushButton('Reset')
        reset_button.setAccessibleName('Return to the first simulation step')
        reset_button.clicked.connect(lambda: self.update_frame(0))
        action_row.addWidget(reset_button)
        self.window_size_button = QtWidgets.QPushButton('Maximize')
        self.window_size_button.setAccessibleName('Maximize or restore the application window')
        self.window_size_button.clicked.connect(self.toggle_window_size)
        action_row.addWidget(self.window_size_button)
        action_row.addSpacing(14)
        action_row.addWidget(QtWidgets.QLabel('coupling cadence:'))
        self.period_box = QtWidgets.QComboBox()
        self.period_box.setAccessibleName('Choose the deterministic coupling cadence')
        self.period_box.addItems(('never', 'every 12 steps', 'every 6 steps', 'every 3 steps', 'every step'))
        self.period_box.setCurrentIndex(2)
        action_row.addWidget(self.period_box)
        regenerate_button = QtWidgets.QPushButton('Regenerate')
        regenerate_button.setAccessibleName('Regenerate using the selected coupling cadence')
        regenerate_button.clicked.connect(self.regenerate)
        action_row.addWidget(regenerate_button)
        self.focus_box = QtWidgets.QComboBox()
        self.focus_box.setAccessibleName('Select a sphere to emphasize')
        self.focus_box.addItems(('All spheres', 'Flood sphere', 'Earthquake sphere', 'Coupled sphere'))
        self.focus_box.currentIndexChanged.connect(self.update_scene)
        action_row.addWidget(self.focus_box)
        action_row.addStretch(1)

        view_row = QtWidgets.QHBoxLayout()
        view_row.setSpacing(8)
        controls_layout.addLayout(view_row)
        view_row.addWidget(QtWidgets.QLabel('R⁴ projection:'))
        self.rotation_sliders = {}
        for label, initial in (('xw', 26), ('yw', -19), ('zw', 14)):
            view_row.addWidget(QtWidgets.QLabel(f'{label}:'))
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setAccessibleName(f'Rotate the parent hypersphere in the {label} plane')
            slider.setRange(-180, 180)
            slider.setValue(initial)
            slider.setMinimumWidth(92)
            slider.valueChanged.connect(self.update_scene)
            view_row.addWidget(slider)
            self.rotation_sliders[label] = slider
        view_row.addStretch(1)

        spatial_row = QtWidgets.QHBoxLayout()
        spatial_row.setSpacing(8)
        controls_layout.addLayout(spatial_row)
        spatial_row.addWidget(QtWidgets.QLabel('spatial rotation:'))
        for label, initial in (('xy', 0), ('xz', 0), ('yz', 0)):
            spatial_row.addWidget(QtWidgets.QLabel(f'{label}:'))
            slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
            slider.setAccessibleName(f'Rotate the parent hypersphere in the {label} plane')
            slider.setRange(-180, 180)
            slider.setValue(initial)
            slider.setMinimumWidth(92)
            slider.valueChanged.connect(self.update_scene)
            spatial_row.addWidget(slider)
            self.rotation_sliders[label] = slider
        spatial_row.addSpacing(8)
        spatial_row.addWidget(QtWidgets.QLabel('timeline:'))
        self.timeline = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.timeline.setAccessibleName('Simulation timeline')
        self.timeline.valueChanged.connect(self.on_timeline)
        spatial_row.addWidget(self.timeline, stretch=1)

        self.status = QtWidgets.QLabel()
        self.status.setStyleSheet(
            'background:#0f172a; border:1px solid #334155; color:#dbeafe; padding:6px 8px; font-size:12px;'
        )
        self.status.setAccessibleName('Current simulation values')
        status_row = QtWidgets.QHBoxLayout()
        status_row.setContentsMargins(0, 0, 0, 0)
        status_row.setSpacing(4)
        status_row.addWidget(self.status, stretch=1)
        self.size_grip = DirectSizeGrip(self)
        status_row.addWidget(self.size_grip)
        controls_layout.addLayout(status_row)
        layout.addWidget(controls)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(24)
        self.timer.timeout.connect(self.advance)

    def _period(self):
        return (0, 12, 6, 3, 1)[self.period_box.currentIndex()]

    def toggle_window_size(self):
        """Provide a window-manager-independent maximize/restore control."""
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    def changeEvent(self, event):
        super().changeEvent(event)
        if event.type() == QtCore.QEvent.WindowStateChange and hasattr(self, 'window_size_button'):
            self.window_size_button.setText('Restore' if self.isMaximized() else 'Maximize')

    def regenerate(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.states, self.local_paths, self.forcing, self.schedule, self.time = simulate_passive_time_system(
            self.n_steps, self.dt, self._period()
        )
        self.sphere_paths_r4 = tuple(
            base.embed_in_r4(points, axes)
            for points, axes in zip(self.local_paths, SPHERE_AXES)
        )
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, self.n_steps)
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.flood_trace.set_series(1.0 + self.states[:, ETA])
        self.earthquake_trace.set_series(self.states[:, U3])
        # Align state index i with the schedule entry that produced it:
        # state zero has no preceding update, while state i uses s[i-1].
        self.schedule_trace.set_series(np.r_[0, self.schedule])
        self.update_frame(0)

    def _project(self, points_r4):
        return base.project_r4_to_r3(
            points_r4,
            self.rotation_sliders['xw'].value(),
            self.rotation_sliders['yw'].value(),
            self.rotation_sliders['zw'].value(),
            self.rotation_sliders['xy'].value(),
            self.rotation_sliders['xz'].value(),
            self.rotation_sliders['yz'].value(),
        )

    def update_scene(self, *_unused):
        if not hasattr(self, 'sphere_paths_r4'):
            return
        focus = self.focus_box.currentIndex() - 1

        for line, raw in zip(self.hypersphere_lines, self.hypersphere_geometry):
            line.set_data(
                pos=self._project(raw),
                color=(0.58, 0.64, 0.74, 0.16 if focus < 0 else 0.10),
            )

        for line, (sphere_id, circle_id, raw) in zip(self.circle_lines, self.circle_geometry):
            emphasized = focus < 0 or sphere_id == focus
            color = SPHERE_COLORS[sphere_id]
            alpha = 0.90 if emphasized else 0.13
            brightness = 1.0 if circle_id == 0 else 0.68
            line.set_data(
                pos=self._project(raw),
                color=(*(channel * brightness for channel in color[:3]), alpha),
            )

        for sphere_id, (path, trail, start, head, flash, raw, color) in enumerate(zip(
            self.paths, self.trails, self.starts, self.heads, self.swap_heads,
            self.sphere_paths_r4, SPHERE_COLORS,
        )):
            emphasized = focus < 0 or sphere_id == focus
            visible = self._project(raw[:self.current_index + 1])
            path.set_data(pos=visible, color=(*color[:3], 0.30 if emphasized else 0.05))
            tail = visible[max(0, len(visible) - TRAIL_LENGTH):]
            tail_alpha = np.linspace(0.14, 1.0, len(tail), dtype=np.float32)
            tail_alpha *= 1.0 if emphasized else 0.16
            tail_colors = np.column_stack((
                np.full(len(tail), color[0]),
                np.full(len(tail), color[1]),
                np.full(len(tail), color[2]),
                tail_alpha,
            )).astype(np.float32)
            trail.set_data(pos=tail, color=tail_colors)
            head_alpha = 1.0 if emphasized else 0.25
            start.set_data(
                pos=visible[:1], face_color=(0.02, 0.04, 0.09, head_alpha),
                edge_color=(*color[:3], head_alpha), size=11, symbol='square',
            )
            head.set_data(
                pos=visible[-1:], face_color=(*color[:3], head_alpha),
                edge_color=(0.98, 0.99, 1.0, head_alpha), size=13, symbol='disc',
            )
            is_swap = self.current_index > 0 and bool(self.schedule[self.current_index - 1])
            flash.set_data(
                pos=visible[-1:],
                face_color=(1.0, 0.12, 0.15, 0.92) if is_swap and emphasized else (0, 0, 0, 0),
                size=18,
            )
        self.canvas.update()

    def update_frame(self, index):
        self.current_index = int(np.clip(index, 0, self.n_steps))
        self.flood_trace.set_frame(self.current_index)
        self.earthquake_trace.set_frame(self.current_index)
        self.schedule_trace.set_frame(self.current_index)
        self.timeline.blockSignals(True)
        self.timeline.setValue(self.current_index)
        self.timeline.blockSignals(False)

        state = self.states[self.current_index]
        current_swap = int(self.schedule[self.current_index - 1]) if self.current_index else 0
        self.status.setText(
            f'k={self.current_index:04d}/{self.n_steps}    t₁={self.time[self.current_index]:6.2f}s    '
            f'Cₙ={self.schedule.mean():.3f}    sₖ={current_swap}    '
            f'h={1.0 + state[ETA]:+.3f}    hv₁={state[P1]:+.3f}    hv₂={state[P2]:+.3f}    '
            f'u₁={state[U1]:+.3f}    u₃={state[U3]:+.3f}    '
            f'∂ₓ₁(Λgh)={self.forcing[self.current_index]:+.3f}'
        )
        self.update_scene()

    def on_timeline(self, value):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(value)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play')
            return
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


if __name__ == '__main__':
    qt_app = QtWidgets.QApplication(sys.argv)
    app.use_app('pyqt5')
    window = PassiveTimeCoupledWindow()
    window.show()
    sys.exit(qt_app.exec_())
