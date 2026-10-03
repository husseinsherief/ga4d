"""
The code uses a **classical fourth-order Runge‑Kutta (RK4) integrator** to advance the state of the pendulum forward in time.

==========================================
1. What is being integrated?
==========================================

The system is formulated as a first‑order ODE:

```
d/dt [θ, φ, θ̇, φ̇] = [θ̇, φ̇, θ̈, φ̈]
```

where the accelerations `θ̈, φ̈` are obtained from the Euler–Lagrange equations (solved numerically at each evaluation).  
So the right‑hand side function `ode_rhs(state)` takes the current state vector and returns the derivative vector.

==========================================
2. The RK4 step
==========================================

The function `rk4_step(state, dt)` performs **one time step** of size `dt`.  
It computes four “slopes”:

- `k1 = f(s)`           – slope at the beginning of the step  
- `k2 = f(s + 0.5·dt·k1)` – slope at the midpoint, estimated using `k1`  
- `k3 = f(s + 0.5·dt·k2)` – midpoint slope, refined using `k2`  
- `k4 = f(s + dt·k3)`   – slope at the end, using `k3`

Then it combines them with weights:

```
s_new = s + (dt/6) · (k1 + 2·k2 + 2·k3 + k4)
```

This is the standard RK4 formula that gives **fourth‑order global accuracy** (the error per step is O(Δt⁵), and total error over a fixed interval is O(Δt⁴)).

==========================================
3. The RK4 table
==========================================

The table at the bottom right shows the **four slope vectors** computed by the **classical fourth‑order Runge‑Kutta (RK4) integrator** during the current time step.

Each row corresponds to one RK4 stage:

- **k₁** – slope at the beginning of the step (`f(state)`).  
- **k₂** – slope at the midpoint, estimated using `k₁` (`f(state + ½·dt·k₁)`).  
- **k₃** – refined slope at the midpoint, using `k₂` (`f(state + ½·dt·k₂)`).  
- **k₄** – slope at the end of the step, using `k₃` (`f(state + dt·k₃)`).

The columns are the components of these slope vectors, in the order:

- `dθ/dt`  → time derivative of θ (angular velocity in polar angle)  
- `dφ/dt`  → time derivative of φ (angular velocity in azimuth)  
- `dθ̇/dt` → time derivative of θ̇ (angular acceleration in θ)  
- `dφ̇/dt` → time derivative of φ̇ (angular acceleration in φ)

In other words, for the state vector `s = [θ, φ, θ̇, φ̇]`, the right‑hand side function `f(s)` returns `[θ̇, φ̇, θ̈, φ̈]`. The table lists the four evaluations of `f` at different points inside the step.

These slopes are then combined with weights to produce the next state:

```
s_new = s + (dt/6)·(k₁ + 2·k₂ + 2·k₃ + k₄)
```

The RK4 values are calculated for the column variables simultaneously by placing their initial state inside of an array.
"""

import numpy as np

from clifford import Cl
from PyQt5 import QtGui
from PyQt5 import QtWidgets
from PyQt5.QtCore import QPointF, Qt, QTimer
from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Line, Markers, Text

# ----------------------------------------------------------------------
# 1. Geometric algebra setup (3D Euclidean)
# ----------------------------------------------------------------------
layout, blades = Cl(3)
e1, e2, e3 = blades['e1'], blades['e2'], blades['e3']


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


def rk4_step_with_slopes(state, dt):
    k1 = ode_rhs(state)
    k2 = ode_rhs(state + 0.5 * dt * k1)
    k3 = ode_rhs(state + 0.5 * dt * k2)
    k4 = ode_rhs(state + dt * k3)
    next_state = state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    return next_state, np.stack([k1, k2, k3, k4]).astype(np.float64)


def integrate(initial_state, dt, n_steps, progress_callback=None):
    states = [initial_state]
    rk4_slopes = np.zeros((n_steps, 4, 4), dtype=np.float64)
    s = initial_state.astype(np.float64)
    progress_interval = max(1, n_steps // 200)
    for step in range(1, n_steps + 1):
        s, slopes = rk4_step_with_slopes(s, dt)
        states.append(s)
        rk4_slopes[step - 1] = slopes
        if progress_callback and (step == n_steps or step % progress_interval == 0):
            progress_callback(step)
    return np.array(states), rk4_slopes


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
    theta0 = 0.8
    phi0 = 0.0
    thetadot0 = 0.0
    phidot0 = 1.5
    initial_state = np.array([theta0, phi0, thetadot0, phidot0], dtype=np.float64)

    conversion_steps = n_steps + 1
    total_work = n_steps + conversion_steps

    if progress_callback:
        progress_callback(0, total_work, 'Integrating Euler-Lagrange equations...')

    def report_integrator_progress(step):
        if progress_callback:
            progress_callback(step, total_work, 'Integrating Euler-Lagrange equations...')

    sol, rk4_slopes = integrate(initial_state, dt, n_steps, report_integrator_progress)

    theta_hist = sol[:, 0]
    phi_hist = sol[:, 1]

    x_hist = np.zeros(len(theta_hist), dtype=np.float32)
    y_hist = np.zeros(len(theta_hist), dtype=np.float32)
    z_hist = np.zeros(len(theta_hist), dtype=np.float32)
    speed = np.zeros(len(theta_hist), dtype=np.float32)
    energy = np.zeros(len(theta_hist), dtype=np.float32)

    conversion_interval = max(1, len(theta_hist) // 200)
    for idx, (th, ph, thd, phd) in enumerate(zip(theta_hist, phi_hist, sol[:, 2], sol[:, 3])):
        pos = position_from_angles(th, ph)
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
    time = np.arange(len(traj), dtype=np.float32) * dt
    return sol, traj, speed, energy, time, rk4_slopes


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
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spherical Pendulum RK4 Angle Diagnostics')
        self.resize(1420, 1040)

        self.dt = 0.005
        self.n_steps = 3000
        self.play_stride = 2
        self.current_index = 0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)
        layout.setSpacing(6)

        instruction = QtWidgets.QLabel(
            'Numerical Euler-Lagrange pipeline in GA for a spherical pendulum: '
            'build L(q, qdot), compute d/dt(∂L/∂qdot)=∂L/∂q at each step, integrate with RK4, '
            'then draw the accumulated trajectory on the sphere. The angle traces show theta and phi over time; '
            'the RK4 table shows k1-k4 for the current step.'
        )
        instruction.setWordWrap(True)
        instruction.setStyleSheet('color:#e6eefb; background:#0f1520; padding: 7px; font-size: 10pt;')
        layout.addWidget(instruction)

        self.canvas = SceneCanvas(title='Spherical Pendulum (GA + Euler-Lagrange)', size=(1280, 760), bgcolor='black')
        self.canvas.native.setParent(central)
        layout.addWidget(self.canvas.native)

        self.view = self.canvas.central_widget.add_view()
        self.view.camera = scene.TurntableCamera(elevation=20, azimuth=45, distance=4.0)
        self.view.camera.set_range(x=(-1.2, 1.2), y=(-1.2, 1.2), z=(-1.2, 1.2))

        self.title = Text(
            'Spherical pendulum trajectory (GA Lagrangian dynamics)',
            color='white',
            font_size=14,
            anchor_x='center',
            anchor_y='top',
            parent=self.view.scene
        )
        self.title.pos = (0.0, 0.0, 1.28)

        self.sphere_lines = [
            Line(pos=line, color=(0.5, 0.55, 0.6, 0.45), width=1.0, connect='strip', parent=self.view.scene)
            for line in build_sphere_wireframe()
        ]
        self.orbit_line = Line(
            color=(0.15, 0.78, 1.0, 0.95), width=3.0, connect='strip', parent=self.view.scene
        )
        self.orbit_head = Markers(parent=self.view.scene)
        self.anchor = Markers(parent=self.view.scene)
        self.anchor.set_data(
            pos=np.array([[0.0, 0.0, R_sphere]], dtype=np.float32),
            face_color=(1.0, 0.75, 0.15, 0.95),
            edge_color=(0, 0, 0, 1),
            size=10
        )

        diagnostics = QtWidgets.QWidget()
        diagnostics_layout = QtWidgets.QHBoxLayout(diagnostics)
        diagnostics_layout.setContentsMargins(0, 0, 0, 0)
        diagnostics_layout.setSpacing(8)
        layout.addWidget(diagnostics)

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

        rk4_panel = QtWidgets.QGroupBox('RK4 slopes for the current step')
        rk4_panel.setStyleSheet(
            'QGroupBox { color:#e6eefb; border:1px solid #2b3444; margin-top:8px; padding:6px; } '
            'QGroupBox::title { subcontrol-origin: margin; left:8px; padding:0 4px; }'
        )
        rk4_layout = QtWidgets.QVBoxLayout(rk4_panel)
        self.rk4_context = QtWidgets.QLabel()
        self.rk4_context.setStyleSheet('color:#d7e2f1; font-size:9pt;')
        rk4_layout.addWidget(self.rk4_context)

        self.rk4_table = QtWidgets.QTableWidget(4, 4)
        self.rk4_table.setHorizontalHeaderLabels(['dtheta/dt', 'dphi/dt', 'dthetadot/dt', 'dphidot/dt'])
        self.rk4_table.setVerticalHeaderLabels(['k1', 'k2', 'k3', 'k4'])
        self.rk4_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.rk4_table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        self.rk4_table.setMinimumWidth(540)
        self.rk4_table.setMinimumHeight(172)
        self.rk4_table.setFont(QtGui.QFont('Monospace', 9))
        self.rk4_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.rk4_table.verticalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.rk4_table.setStyleSheet(
            'QTableWidget { background:#090d14; color:#e6eefb; gridline-color:#2b3444; } '
            'QHeaderView::section { background:#141b27; color:#d7e2f1; border:1px solid #2b3444; padding:4px; }'
        )
        rk4_layout.addWidget(self.rk4_table)
        diagnostics_layout.addWidget(rk4_panel, stretch=2)

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
            self.sol, self.trajectory, self.speed, self.energy, self.time, self.rk4_slopes = build_simulation(
                self.dt,
                self.n_steps,
                report_progress
            )
        finally:
            progress.setValue(progress.maximum())
            progress.close()

        self.path_len = len(self.trajectory)
        self.path = self.trajectory.astype(np.float32)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(self.path_len - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)
        self.theta_trace.set_data(self.time, self.sol[:, 0])
        self.phi_trace.set_data(self.time, self.sol[:, 1])

        e0 = float(self.energy[0])
        em = float(np.max(np.abs(self.energy - e0)))
        self.status_bar.showMessage(
            f'dt={self.dt:.4f} | steps={self.n_steps} | total runtime={self.time[-1]:.2f}s | '
            f'initial energy={e0:.6f} | max energy drift={em:.3e}'
        )

    def on_slider_changed(self, value):
        self.timer.stop()
        self.play_button.setText('Play')
        self.update_frame(value)

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play')
            return

        if self.current_index >= self.path_len - 1:
            self.update_frame(0)
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
        next_index = min(self.current_index + self.play_stride, self.path_len - 1)
        self.update_frame(next_index)
        if next_index >= self.path_len - 1:
            self.timer.stop()
            self.play_button.setText('Play')

    def update_frame(self, index):
        self.current_index = int(np.clip(index, 0, self.path_len - 1))
        head = self.path[: self.current_index + 1]
        self.orbit_line.set_data(pos=head)
        head_pos = head[-1]

        self.orbit_head.set_data(
            pos=np.array([head_pos], dtype=np.float32),
            face_color=(1.0, 0.95, 0.2, 1.0),
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
            f'|v|={v_scalar:.4f}   E={self.energy[self.current_index]:.6f}'
        )

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_index)
        self.frame_slider.blockSignals(False)

        self.theta_trace.set_frame(self.current_index)
        self.phi_trace.set_frame(self.current_index)
        self.update_rk4_table()

        self.title.pos = (0.0, 0.0, 1.28)
        self.canvas.update()

    def update_rk4_table(self):
        if not hasattr(self, 'rk4_slopes') or len(self.rk4_slopes) == 0:
            return

        step_index = min(self.current_index, len(self.rk4_slopes) - 1)
        slopes = self.rk4_slopes[step_index]
        theta, phi, thetadot, phidot = self.sol[step_index]
        self.rk4_context.setText(
            f'step {step_index} -> {step_index + 1}   '
            f's=[theta {theta:.4f}, phi {phi:.4f}, thetadot {thetadot:.4f}, phidot {phidot:.4f}]'
        )

        for row in range(4):
            for col in range(4):
                item = QtWidgets.QTableWidgetItem(f'{slopes[row, col]: .6f}')
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
                self.rk4_table.setItem(row, col, item)


if __name__ == '__main__':
    qt_app = QtWidgets.QApplication([])
    app.use_app('pyqt5')
    window = MainWindow()
    window.show()
    qt_app.exec_()
