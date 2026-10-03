import numpy as np

from clifford import Cl
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer
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


def integrate(initial_state, dt, n_steps):
    states = [initial_state]
    s = initial_state.astype(np.float64)
    for _ in range(n_steps):
        s = rk4_step(s, dt)
        states.append(s)
    return np.array(states)


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


def build_simulation(dt=0.005, n_steps=3000):
    theta0 = 0.8
    phi0 = 0.0
    thetadot0 = 0.0
    phidot0 = 1.5
    initial_state = np.array([theta0, phi0, thetadot0, phidot0], dtype=np.float64)
    sol = integrate(initial_state, dt, n_steps)

    theta_hist = sol[:, 0]
    phi_hist = sol[:, 1]

    x_hist = np.zeros(len(theta_hist), dtype=np.float32)
    y_hist = np.zeros(len(theta_hist), dtype=np.float32)
    z_hist = np.zeros(len(theta_hist), dtype=np.float32)
    speed = np.zeros(len(theta_hist), dtype=np.float32)
    energy = np.zeros(len(theta_hist), dtype=np.float32)

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

    traj = np.column_stack([x_hist, y_hist, z_hist]).astype(np.float32)
    time = np.arange(len(traj), dtype=np.float32) * dt
    return sol, traj, speed, energy, time


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Spherical Pendulum with Numerical Euler-Lagrange')
        self.resize(1250, 880)

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
            'then draw the accumulated trajectory on the sphere. Play/Pause animates; the scrubber lets you inspect any frame.'
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
        self.sol, self.trajectory, self.speed, self.energy, self.time = build_simulation(self.dt, self.n_steps)
        self.path_len = len(self.trajectory)
        self.path = self.trajectory.astype(np.float32)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(self.path_len - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

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
        vel = scalar_part(v | e1)
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

        self.title.pos = (0.0, 0.0, 1.28)
        self.canvas.update()


if __name__ == '__main__':
    qt_app = QtWidgets.QApplication([])
    app.use_app('pyqt5')
    window = MainWindow()
    window.show()
    qt_app.exec_()
