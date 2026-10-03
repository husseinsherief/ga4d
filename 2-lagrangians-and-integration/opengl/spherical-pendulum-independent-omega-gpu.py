import numpy as np

from clifford import Cl
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPointF, Qt, QTimer
from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Line, Markers

# ----------------------------------------------------------------------
# 1. Geometric algebra setup (3D Euclidean)
# ----------------------------------------------------------------------
layout, blades = Cl(3)
e1, e2, e3 = blades['e1'], blades['e2'], blades['e3']
PHI_ROTATION_PLANE = blades['e12']
THETA_ROTATION_PLANE = blades['e23']
OMEGA_ROTATION_PLANE = (THETA_ROTATION_PLANE + PHI_ROTATION_PLANE) / np.sqrt(2.0)
OMEGA_ROTATION_PLANE_DEGREES_FROM_THETA = 45.0

THETA_COLOR_RGB = (68, 190, 255)
PHI_COLOR_RGB = (255, 151, 74)
OMEGA_COLOR_RGB = (126, 235, 146)
FULL_COLOR_RGB = (255, 226, 92)
THETA_COLOR = (THETA_COLOR_RGB[0] / 255.0, THETA_COLOR_RGB[1] / 255.0, THETA_COLOR_RGB[2] / 255.0, 0.96)
PHI_COLOR = (PHI_COLOR_RGB[0] / 255.0, PHI_COLOR_RGB[1] / 255.0, PHI_COLOR_RGB[2] / 255.0, 0.96)
OMEGA_COLOR = (OMEGA_COLOR_RGB[0] / 255.0, OMEGA_COLOR_RGB[1] / 255.0, OMEGA_COLOR_RGB[2] / 255.0, 0.96)
FULL_COLOR = (FULL_COLOR_RGB[0] / 255.0, FULL_COLOR_RGB[1] / 255.0, FULL_COLOR_RGB[2] / 255.0, 0.96)
THETA_PLANE_COLOR = (THETA_COLOR_RGB[0] / 255.0, THETA_COLOR_RGB[1] / 255.0, THETA_COLOR_RGB[2] / 255.0, 0.62)
PHI_PLANE_COLOR = (PHI_COLOR_RGB[0] / 255.0, PHI_COLOR_RGB[1] / 255.0, PHI_COLOR_RGB[2] / 255.0, 0.62)
OMEGA_PLANE_COLOR = (OMEGA_COLOR_RGB[0] / 255.0, OMEGA_COLOR_RGB[1] / 255.0, OMEGA_COLOR_RGB[2] / 255.0, 0.62)

THETA_SPHERE_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float32)
PHI_SPHERE_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float32)
OMEGA_SPHERE_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float32)
FULL_SPHERE_CENTER = np.array([0.0, 0.0, 0.0], dtype=np.float32)


def scalar_part(mv):
    """Return scalar component for a clifford scalar or multivector."""
    if hasattr(mv, 'value'):
        return float(mv.value[0])
    return float(mv)


# ----------------------------------------------------------------------
# 2. Rotor parameterisation of a sphere
# ----------------------------------------------------------------------
R_sphere = 1.0


def position_from_angles(theta, phi):
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    return R_sphere * (st * cp * e1 + st * sp * e2 + ct * e3)


def rotor_from_plane_angle(plane, angle):
    return np.cos(0.5 * angle) - np.sin(0.5 * angle) * plane


def position_from_independent_omega(theta, phi, omega):
    theta_rotor = rotor_from_plane_angle(THETA_ROTATION_PLANE, theta)
    phi_rotor = rotor_from_plane_angle(PHI_ROTATION_PLANE, phi)
    omega_rotor = rotor_from_plane_angle(OMEGA_ROTATION_PLANE, omega)
    full_rotor = omega_rotor * phi_rotor * theta_rotor
    return R_sphere * (full_rotor * e3 * ~full_rotor)


def velocity_from_angles(theta, phi, thetadot, phidot):
    st = np.sin(theta)
    ct = np.cos(theta)
    sp = np.sin(phi)
    cp = np.cos(phi)
    e_theta = ct * cp * e1 + ct * sp * e2 - st * e3
    e_phi = -sp * e1 + cp * e2
    return R_sphere * (thetadot * e_theta + st * phidot * e_phi)


m = 1.0
g = 9.81
INITIAL_THETA = 0.8
INITIAL_PHI = 0.0
INITIAL_THETADOT = 0.0
INITIAL_PHIDOT = 1.5
OMEGA_LINEAR_SLOPE = 0.55
OMEGA_LINEAR_INTERCEPT = 0.25


def lagrangian(theta, phi, thetadot, phidot):
    v = velocity_from_angles(theta, phi, thetadot, phidot)
    kinetic = 0.5 * m * scalar_part(v | v)
    z = R_sphere * np.cos(theta)
    potential = m * g * z
    return kinetic - potential


def numerical_gradient(f, x, eps=1e-6):
    grad = np.zeros_like(x)
    for idx in range(len(x)):
        plus = x.copy()
        minus = x.copy()
        plus[idx] += eps
        minus[idx] -= eps
        grad[idx] = (f(plus) - f(minus)) / (2.0 * eps)
    return grad


def euler_lagrange_acceleration(q, qdot, eps=1e-6):
    def L_q(q_val):
        return lagrangian(q_val[0], q_val[1], qdot[0], qdot[1])

    def L_qdot(qdot_val):
        return lagrangian(q[0], q[1], qdot_val[0], qdot_val[1])

    p = numerical_gradient(L_qdot, qdot, eps)
    f = numerical_gradient(L_q, q, eps)

    def p_func(qdot_val):
        return numerical_gradient(L_qdot, qdot_val, eps)

    M = np.zeros((2, 2))
    for j in range(2):
        qdot_plus = qdot.copy()
        qdot_minus = qdot.copy()
        qdot_plus[j] += eps
        qdot_minus[j] -= eps
        M[:, j] = (p_func(qdot_plus) - p_func(qdot_minus)) / (2.0 * eps)

    def p_from_q(q_val):
        def L_fixed_qdot(qdot_val):
            return lagrangian(q_val[0], q_val[1], qdot_val[0], qdot_val[1])

        return numerical_gradient(L_fixed_qdot, qdot, eps)

    q_plus = q + eps * qdot
    q_minus = q - eps * qdot
    c_qdot = (p_from_q(q_plus) - p_from_q(q_minus)) / (2.0 * eps)

    rhs = f - c_qdot
    return np.linalg.solve(M, rhs)


def ode_rhs(state):
    theta, phi, thetadot, phidot = state
    q = np.array([theta, phi], dtype=np.float64)
    qdot = np.array([thetadot, phidot], dtype=np.float64)
    qddot = euler_lagrange_acceleration(q, qdot)
    return np.array([thetadot, phidot, qddot[0], qddot[1]])


def rk4_step(state, dt):
    k1 = ode_rhs(state)
    k2 = ode_rhs(state + 0.5 * dt * k1)
    k3 = ode_rhs(state + 0.5 * dt * k2)
    k4 = ode_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def independent_omega_from_time(time):
    time = np.asarray(time, dtype=np.float64)
    omega = OMEGA_LINEAR_SLOPE * time + OMEGA_LINEAR_INTERCEPT
    if omega.ndim == 0:
        return float(omega)
    return omega


def integrate(initial_state, dt, n_steps, progress_callback=None):
    states = [initial_state]
    s = initial_state.astype(np.float64)
    progress_interval = max(1, n_steps // 200)
    for step in range(1, n_steps + 1):
        s = rk4_step(s, dt)
        states.append(s)
        if progress_callback and (step == n_steps or step % progress_interval == 0):
            progress_callback(step)
    return np.array(states)


def multivector_to_xyz(mv):
    return np.array([
        scalar_part(mv | e1),
        scalar_part(mv | e2),
        scalar_part(mv | e3),
    ], dtype=np.float32)


def rotate_north_pole_in_plane(angle, plane):
    rotor = np.cos(0.5 * angle) - np.sin(0.5 * angle) * plane
    return multivector_to_xyz(rotor * e3 * ~rotor)


def build_theta_component_path(theta_hist):
    points = np.zeros((len(theta_hist), 3), dtype=np.float32)
    points[:, 0] = np.sin(theta_hist)
    points[:, 2] = np.cos(theta_hist)
    return points + THETA_SPHERE_CENTER


def build_phi_component_path(phi_hist, theta_reference=INITIAL_THETA):
    points = np.zeros((len(phi_hist), 3), dtype=np.float32)
    radius = np.sin(theta_reference)
    z = np.cos(theta_reference)
    points[:, 0] = radius * np.cos(phi_hist)
    points[:, 1] = radius * np.sin(phi_hist)
    points[:, 2] = z
    return points + PHI_SPHERE_CENTER


def build_omega_component_path(omega_hist):
    points = np.zeros((len(omega_hist), 3), dtype=np.float32)
    for idx, angle in enumerate(omega_hist):
        points[idx] = rotate_north_pole_in_plane(float(angle), OMEGA_ROTATION_PLANE)
    return points + OMEGA_SPHERE_CENTER


def build_full_motion_path(trajectory):
    return trajectory.astype(np.float32) + FULL_SPHERE_CENTER


def build_plane_disk_wireframe(center, offset, basis_u, basis_v, radius, n_segments=128):
    center = np.asarray(center, dtype=np.float32)
    offset = np.asarray(offset, dtype=np.float32)
    basis_u = np.asarray(basis_u, dtype=np.float32)
    basis_v = np.asarray(basis_v, dtype=np.float32)
    angles = np.linspace(0.0, 2.0 * np.pi, n_segments + 1, dtype=np.float32)
    lines = []

    for fraction in (0.35, 0.7, 1.0):
        r = radius * fraction
        ring = center + offset + r * (
            np.cos(angles)[:, None] * basis_u + np.sin(angles)[:, None] * basis_v
        )
        lines.append(ring.astype(np.float32))

    for angle in np.linspace(0.0, np.pi, 6, endpoint=False, dtype=np.float32):
        direction = np.cos(angle) * basis_u + np.sin(angle) * basis_v
        line = np.vstack([
            center + offset - radius * direction,
            center + offset + radius * direction,
        ])
        lines.append(line.astype(np.float32))

    return lines


def build_theta_plane_wireframe():
    return build_plane_disk_wireframe(
        THETA_SPHERE_CENTER,
        (0.0, 0.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        R_sphere
    )


def build_phi_plane_wireframe(theta_reference=INITIAL_THETA):
    return build_plane_disk_wireframe(
        PHI_SPHERE_CENTER,
        (0.0, 0.0, np.cos(theta_reference)),
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        np.sin(theta_reference)
    )


def build_omega_plane_wireframe():
    return build_plane_disk_wireframe(
        OMEGA_SPHERE_CENTER,
        (0.5, 0.0, 0.5),
        (-1.0 / np.sqrt(2.0), 0.0, 1.0 / np.sqrt(2.0)),
        (0.0, -1.0, 0.0),
        1.0 / np.sqrt(2.0)
    )


def translate_lines(lines, center):
    return [line + center for line in lines]


def build_sphere_wireframe(n_parallel=24, n_meridian=24):
    circles = []
    # latitude rings
    for theta in np.linspace(0.0, np.pi, n_parallel):
        ring = []
        for phi in np.linspace(0.0, 2.0 * np.pi, n_meridian + 1):
            pos = position_from_angles(theta, phi)
            ring.append([scalar_part(pos | e1), scalar_part(pos | e2), scalar_part(pos | e3)])
        circles.append(np.asarray(ring, dtype=np.float32))

    # longitude rings
    for phi in np.linspace(0.0, 2.0 * np.pi, n_parallel):
        line = []
        for theta in np.linspace(0.0, np.pi, n_meridian + 1):
            pos = position_from_angles(theta, phi)
            line.append([scalar_part(pos | e1), scalar_part(pos | e2), scalar_part(pos | e3)])
        circles.append(np.asarray(line, dtype=np.float32))

    return circles


def build_simulation(dt=0.005, n_steps=3000, progress_callback=None):
    initial_state = np.array(
        [INITIAL_THETA, INITIAL_PHI, INITIAL_THETADOT, INITIAL_PHIDOT],
        dtype=np.float64
    )

    conversion_steps = n_steps + 1
    total_work = n_steps + conversion_steps

    if progress_callback:
        progress_callback(0, total_work, 'Integrating Euler-Lagrange equations...')

    def report_integrator_progress(step):
        if progress_callback:
            progress_callback(step, total_work, 'Integrating Euler-Lagrange equations...')

    sol = integrate(initial_state, dt, n_steps, report_integrator_progress)

    theta_hist = sol[:, 0]
    phi_hist = sol[:, 1]
    time = np.arange(len(theta_hist), dtype=np.float32) * dt
    omega = independent_omega_from_time(time).astype(np.float32)

    x_hist = np.zeros(len(theta_hist), dtype=np.float32)
    y_hist = np.zeros(len(theta_hist), dtype=np.float32)
    z_hist = np.zeros(len(theta_hist), dtype=np.float32)
    speed = np.zeros(len(theta_hist), dtype=np.float32)
    energy = np.zeros(len(theta_hist), dtype=np.float32)

    conversion_interval = max(1, len(theta_hist) // 200)
    for idx, (th, ph, om, thd, phd) in enumerate(zip(theta_hist, phi_hist, omega, sol[:, 2], sol[:, 3])):
        pos = position_from_independent_omega(th, ph, om)
        vel = velocity_from_angles(th, ph, thd, phd)
        x_hist[idx] = scalar_part(pos | e1)
        y_hist[idx] = scalar_part(pos | e2)
        z_hist[idx] = scalar_part(pos | e3)
        speed[idx] = float(np.sqrt(max(scalar_part(vel | vel), 0.0)))
        potential = m * g * R_sphere * np.cos(th)
        kinetic = 0.5 * m * scalar_part(vel | vel)
        energy[idx] = kinetic + potential
        if progress_callback and (idx == len(theta_hist) - 1 or idx % conversion_interval == 0):
            progress_callback(
                n_steps + idx + 1,
                total_work,
                'Finalizing trajectory arrays...'
            )

    traj = np.column_stack([x_hist, y_hist, z_hist]).astype(np.float32)
    return sol, traj, speed, energy, time, omega


class AngleTraceWidget(QtWidgets.QWidget):
    def __init__(self, title, angle_name, line_color, parent=None):
        super().__init__(parent)
        self.title = title
        self.angle_name = angle_name
        self.line_color = QtGui.QColor(*line_color)
        self.time = np.array([], dtype=np.float32)
        self.values = np.array([], dtype=np.float64)
        self.current_index = 0
        self.setMinimumHeight(138)
        self.setStyleSheet('background:#090d14;')

    def set_data(self, time, values):
        self.time = np.asarray(time, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        self.current_index = 0
        self.update()

    def set_frame(self, index):
        if len(self.values) == 0:
            return
        self.current_index = int(np.clip(index, 0, len(self.values) - 1))
        self.update()

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor('#090d14'))

        if len(self.time) < 2 or len(self.values) < 2:
            painter.end()
            return

        left, top, right, bottom = 78, 28, 18, 36
        plot = self.rect().adjusted(left, top, -right, -bottom)
        if plot.width() <= 8 or plot.height() <= 8:
            painter.end()
            return

        vmin = float(np.min(self.values))
        vmax = float(np.max(self.values))
        if abs(vmax - vmin) < 1e-9:
            vmin -= 1.0
            vmax += 1.0
        pad = 0.08 * (vmax - vmin)
        vmin -= pad
        vmax += pad
        tmin = float(self.time[0])
        tmax = float(self.time[-1])

        def map_x(t):
            return plot.left() + (float(t) - tmin) / (tmax - tmin) * plot.width()

        def map_y(v):
            return plot.bottom() - (float(v) - vmin) / (vmax - vmin) * plot.height()

        painter.setPen(QtGui.QPen(QtGui.QColor('#2b3444'), 1))
        painter.drawRect(plot)
        for frac in (0.25, 0.5, 0.75):
            x = plot.left() + frac * plot.width()
            y = plot.top() + frac * plot.height()
            painter.drawLine(int(x), plot.top(), int(x), plot.bottom())
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))

        painter.setPen(QtGui.QPen(QtGui.QColor('#d7e2f1'), 1))
        painter.setFont(QtGui.QFont('Sans Serif', 9))
        painter.drawText(10, 18, self.title)
        painter.drawText(10, plot.top() + 16, f'{self.angle_name} angle')
        painter.drawText(plot.center().x() - 30, self.height() - 10, 'time (s)')

        painter.setPen(QtGui.QPen(QtGui.QColor('#92a0b3'), 1))
        painter.drawText(10, plot.top() + 4, f'{vmax:.2f} rad')
        painter.drawText(10, plot.bottom(), f'{vmin:.2f} rad')
        painter.drawText(plot.left(), self.height() - 10, f'{tmin:.1f}')
        painter.drawText(plot.right() - 34, self.height() - 10, f'{tmax:.1f}')

        points = [QPointF(map_x(t), map_y(v)) for t, v in zip(self.time, self.values)]
        painter.setPen(QtGui.QPen(self.line_color, 2))
        painter.drawPolyline(QtGui.QPolygonF(points))

        idx = self.current_index
        cx = map_x(self.time[idx])
        cy = map_y(self.values[idx])
        painter.setPen(QtGui.QPen(QtGui.QColor('#ffffff'), 1))
        painter.drawLine(int(cx), plot.top(), int(cx), plot.bottom())
        painter.setBrush(QtGui.QBrush(QtGui.QColor('#ffd24a')))
        painter.drawEllipse(QPointF(cx, cy), 4.5, 4.5)

        value = float(self.values[idx])
        painter.setPen(QtGui.QPen(QtGui.QColor('#ffffff'), 1))
        painter.drawText(
            plot.left() + 8,
            plot.top() + 18,
            f't={self.time[idx]:.2f}s   {self.angle_name}={value:.4f} rad ({np.degrees(value):.1f} deg)'
        )
        painter.end()


class MainWindow(QtWidgets.QMainWindow):
    def _add_sphere_panel(self, grid, row, column, title, color, center, plane_lines=None):
        panel = QtWidgets.QWidget()
        panel.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.setSpacing(3)

        title_label = QtWidgets.QLabel(title)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            f'color: rgb({int(color[0] * 255)}, {int(color[1] * 255)}, {int(color[2] * 255)}); '
            'background:#090d14; padding: 4px; font-size: 10pt; font-weight: 600;'
        )
        panel_layout.addWidget(title_label)

        canvas = SceneCanvas(title=title, size=(620, 330), bgcolor='black')
        canvas.native.setParent(panel)
        canvas.native.setMinimumSize(220, 180)
        canvas.native.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        panel_layout.addWidget(canvas.native, stretch=1)

        view = canvas.central_widget.add_view()
        view.camera = scene.TurntableCamera(elevation=20, azimuth=38, distance=3.4)
        center = np.asarray(center, dtype=np.float32)
        view.camera.set_range(
            x=(float(center[0] - 1.18), float(center[0] + 1.18)),
            y=(float(center[1] - 1.18), float(center[1] + 1.18)),
            z=(-1.15, 1.18)
        )

        for sphere_line in translate_lines(self.base_wireframe, center):
            Line(pos=sphere_line, color=(0.5, 0.55, 0.6, 0.42), width=1.0, connect='strip', parent=view.scene)

        for plane_line in plane_lines or ():
            Line(pos=plane_line, color=color[:3] + (0.62,), width=2.6, connect='strip', parent=view.scene)

        north_marker = Markers(parent=view.scene)
        north_marker.set_data(
            pos=np.array([[center[0], center[1], center[2] + R_sphere]], dtype=np.float32),
            face_color=(1.0, 1.0, 1.0, 0.95),
            edge_color=(0.0, 0.0, 0.0, 1.0),
            size=7
        )

        path_line = Line(color=color, width=3.0, connect='strip', parent=view.scene)
        head = Markers(parent=view.scene)

        grid.addWidget(panel, row, column)
        return canvas, path_line, head

    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spherical Pendulum With Independent Omega')
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowMinimizeButtonHint
            | Qt.WindowMaximizeButtonHint
            | Qt.WindowCloseButtonHint
        )
        self.setMinimumSize(980, 680)
        self.resize(1420, 1040)

        self.dt = 0.005
        self.n_steps = 3000
        self.play_stride = 1.0
        self.speed_multiplier = 1.0
        self.frame_cursor = 0.0
        self.current_index = 0

        central = QtWidgets.QWidget()
        central.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setSpacing(6)

        instruction = QtWidgets.QLabel(
            'Numerical Euler-Lagrange pipeline in GA for a spherical pendulum: '
            'build L(q, qdot), compute d/dt(∂L/∂qdot)=∂L/∂q at each step, integrate with RK4, '
            f'then prescribe an independent omega(t)={OMEGA_LINEAR_SLOPE:.2f}t+{OMEGA_LINEAR_INTERCEPT:.2f}. '
            'The full-motion sphere uses the composed rotor R_omega R_phi R_theta, so omega changes the plotted position.'
        )
        instruction.setWordWrap(True)
        instruction.setStyleSheet('color:#e6eefb; background:#0f1520; padding: 7px; font-size: 10pt;')
        layout.addWidget(instruction)

        scene_title = QtWidgets.QLabel('Independent sphere views: theta | phi | full motion with independent omega')
        scene_title.setAlignment(Qt.AlignCenter)
        scene_title.setStyleSheet('color:#e6eefb; background:#090d14; padding:4px; font-size:11pt;')
        layout.addWidget(scene_title)

        sphere_area = QtWidgets.QWidget()
        sphere_grid = QtWidgets.QGridLayout(sphere_area)
        sphere_grid.setContentsMargins(0, 0, 0, 0)
        sphere_grid.setHorizontalSpacing(6)
        sphere_grid.setVerticalSpacing(0)
        for column in range(3):
            sphere_grid.setColumnStretch(column, 1)
        layout.addWidget(sphere_area, stretch=2)

        self.base_wireframe = build_sphere_wireframe()
        self.theta_canvas, self.theta_line, self.theta_head = self._add_sphere_panel(
            sphere_grid,
            0,
            0,
            'theta only',
            THETA_COLOR,
            THETA_SPHERE_CENTER,
            build_theta_plane_wireframe()
        )
        self.phi_canvas, self.phi_line, self.phi_head = self._add_sphere_panel(
            sphere_grid,
            0,
            1,
            'phi only',
            PHI_COLOR,
            PHI_SPHERE_CENTER,
            build_phi_plane_wireframe()
        )
        self.full_canvas, self.full_line, self.full_head = self._add_sphere_panel(
            sphere_grid,
            0,
            2,
            'full motion: x(theta, phi, omega)',
            FULL_COLOR,
            FULL_SPHERE_CENTER
        )

        diagnostics = QtWidgets.QWidget()
        diagnostics.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        diagnostics_layout = QtWidgets.QHBoxLayout(diagnostics)
        diagnostics_layout.setContentsMargins(0, 0, 0, 0)
        diagnostics_layout.setSpacing(8)
        layout.addWidget(diagnostics, stretch=1)

        angle_panel = QtWidgets.QWidget()
        angle_layout = QtWidgets.QVBoxLayout(angle_panel)
        angle_layout.setContentsMargins(0, 0, 0, 0)
        angle_layout.setSpacing(6)
        self.theta_trace = AngleTraceWidget(
            'Theta angular movement: polar swing away from vertical',
            'theta',
            (68, 190, 255)
        )
        self.phi_trace = AngleTraceWidget(
            'Phi angular movement: azimuth around the vertical axis',
            'phi',
            (255, 151, 74)
        )
        angle_layout.addWidget(self.theta_trace)
        angle_layout.addWidget(self.phi_trace)
        diagnostics_layout.addWidget(angle_panel, stretch=3)

        omega_panel = QtWidgets.QWidget()
        omega_layout = QtWidgets.QVBoxLayout(omega_panel)
        omega_layout.setContentsMargins(0, 0, 0, 0)
        self.omega_trace = AngleTraceWidget(
            f'Independent omega angular movement: omega(t)={OMEGA_LINEAR_SLOPE:.2f}t+{OMEGA_LINEAR_INTERCEPT:.2f}',
            'omega',
            (126, 235, 146)
        )
        self.omega_trace.setMinimumHeight(138)
        omega_layout.addWidget(self.omega_trace)
        diagnostics_layout.addWidget(omega_panel, stretch=2)

        controls = QtWidgets.QWidget()
        control_layout = QtWidgets.QHBoxLayout(controls)
        layout.addWidget(controls)

        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.clicked.connect(self.toggle_play)
        control_layout.addWidget(self.play_button)

        step_button = QtWidgets.QPushButton('Step')
        step_button.clicked.connect(self.step_once)
        control_layout.addWidget(step_button)

        reset_button = QtWidgets.QPushButton('Reset')
        reset_button.clicked.connect(self.reset_animation)
        control_layout.addWidget(reset_button)

        control_layout.addWidget(QtWidgets.QLabel('Speed'))
        self.speed_input = QtWidgets.QDoubleSpinBox()
        self.speed_input.setDecimals(2)
        self.speed_input.setRange(0.05, 20.0)
        self.speed_input.setSingleStep(0.1)
        self.speed_input.setValue(self.speed_multiplier)
        self.speed_input.setSuffix('x')
        self.speed_input.setFixedWidth(78)
        self.speed_input.valueChanged.connect(self.on_speed_changed)
        control_layout.addWidget(self.speed_input)

        self.frame_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        control_layout.addWidget(self.frame_slider, stretch=1)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setMinimumWidth(460)
        control_layout.addWidget(self.status_label)

        self.timer = QTimer()
        self.timer.setInterval(16)
        self.timer.timeout.connect(self.advance_frame)

        self.status_bar = self.statusBar()
        self.simulate()
        self.update_frame(0)

    def simulate(self):
        progress = QtWidgets.QProgressDialog(
            'Preparing Euler-Lagrange trajectory...',
            '',
            0,
            self.n_steps,
            self
        )
        progress.setWindowTitle('Generating trajectory')
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setAutoClose(False)
        progress.setMinimumWidth(360)
        progress.show()
        QtWidgets.QApplication.processEvents()

        def report_progress(value, total, label):
            progress.setMaximum(total)
            progress.setLabelText(label)
            progress.setValue(value)
            QtWidgets.QApplication.processEvents()

        try:
            self.sol, self.trajectory, self.speed, self.energy, self.time, self.omega = build_simulation(
                self.dt,
                self.n_steps,
                report_progress
            )
            progress.setMaximum(3)
            progress.setLabelText('Building component rotation spheres...')
            progress.setValue(0)
            QtWidgets.QApplication.processEvents()
            self.theta_path = build_theta_component_path(self.sol[:, 0])
            progress.setValue(1)
            QtWidgets.QApplication.processEvents()
            self.phi_path = build_phi_component_path(self.sol[:, 1])
            progress.setValue(2)
            QtWidgets.QApplication.processEvents()
            self.full_path = build_full_motion_path(self.trajectory)
            progress.setValue(3)
            QtWidgets.QApplication.processEvents()
        finally:
            progress.setValue(progress.maximum())
            progress.close()

        self.path_len = len(self.theta_path)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(self.path_len - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.theta_trace.set_data(self.time, self.sol[:, 0])
        self.phi_trace.set_data(self.time, self.sol[:, 1])
        self.omega_trace.set_data(self.time, self.omega)

        e0 = float(self.energy[0])
        em = float(np.max(np.abs(self.energy - e0)))
        self.status_bar.showMessage(
            f'dt={self.dt:.4f} | steps={self.n_steps} | total runtime={self.time[-1]:.2f}s | '
            f'base theta/phi energy={e0:.6f} | base max energy drift={em:.3e} | '
            f'omega(t)={OMEGA_LINEAR_SLOPE:.2f}t+{OMEGA_LINEAR_INTERCEPT:.2f}'
        )

    def on_slider_changed(self, value):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(value)

    def on_speed_changed(self, value):
        self.speed_multiplier = float(value)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play')
            return

        if self.current_index >= self.path_len - 1:
            self.update_frame(0)
        self.frame_cursor = float(self.current_index)
        self.timer.start()
        self.play_button.setText('Pause')

    def step_once(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(min(self.current_index + 1, self.path_len - 1))

    def reset_animation(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(0)

    def advance_frame(self):
        self.frame_cursor = min(
            self.frame_cursor + self.play_stride * self.speed_multiplier,
            float(self.path_len - 1)
        )
        next_index = int(self.frame_cursor)
        self.update_frame(next_index, sync_cursor=False)
        if next_index >= self.path_len - 1:
            self.timer.stop()
            self.play_button.setText('Play')

    def update_frame(self, index, sync_cursor=True):
        self.current_index = int(np.clip(index, 0, self.path_len - 1))
        if sync_cursor:
            self.frame_cursor = float(self.current_index)
        theta_head = self.theta_path[: self.current_index + 1]
        phi_head = self.phi_path[: self.current_index + 1]
        full_head = self.full_path[: self.current_index + 1]
        self.theta_line.set_data(pos=theta_head)
        self.phi_line.set_data(pos=phi_head)
        self.full_line.set_data(pos=full_head)

        self.theta_head.set_data(
            pos=np.array([theta_head[-1]], dtype=np.float32),
            face_color=THETA_COLOR,
            edge_color=(0.0, 0.0, 0.0, 1.0),
            size=11
        )
        self.phi_head.set_data(
            pos=np.array([phi_head[-1]], dtype=np.float32),
            face_color=PHI_COLOR,
            edge_color=(0.0, 0.0, 0.0, 1.0),
            size=11
        )
        self.full_head.set_data(
            pos=np.array([full_head[-1]], dtype=np.float32),
            face_color=FULL_COLOR,
            edge_color=(0.0, 0.0, 0.0, 1.0),
            size=11
        )

        theta, phi, thetadot, phidot = self.sol[self.current_index]
        v = velocity_from_angles(theta, phi, thetadot, phidot)
        v_scalar = float(np.linalg.norm([
            scalar_part(v | e1), scalar_part(v | e2), scalar_part(v | e3)
        ]))
        self.status_label.setText(
            f'frame {self.current_index}/{self.path_len - 1}   '
            f't={self.time[self.current_index]:.2f}s   '
            f'theta={theta:.3f}   phi={phi:.3f}   '
            f'omega={self.omega[self.current_index]:.3f}   '
            f'base |v|={v_scalar:.4f}   base E={self.energy[self.current_index]:.6f}'
        )

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_index)
        self.frame_slider.blockSignals(False)

        self.theta_trace.set_frame(self.current_index)
        self.phi_trace.set_frame(self.current_index)
        self.omega_trace.set_frame(self.current_index)

        self.theta_canvas.update()
        self.phi_canvas.update()
        self.full_canvas.update()


if __name__ == '__main__':
    qt_app = QtWidgets.QApplication([])
    app.use_app('pyqt5')
    window = MainWindow()
    window.show()
    qt_app.exec_()
