"""Four synchronized spherical-pendulum views with input-swapped coupling.

The valid C=0 Euler-Lagrange trajectory is integrated once.  On a scheduled
coupling sample, a display copies that sample, reorders the numerical-gradient
input from ``(theta, phi)`` to ``(phi, theta)``, and evaluates the ordinary
coordinate gradient at that reordered point:

    C=0.00: always differentiate at (theta, phi)
    C=0.25: differentiate at (phi, theta) every 8th sample
    C=0.50: differentiate at (phi, theta) every 4th sample
    C=1.00: differentiate at (phi, theta) every 2nd sample

The central-difference algorithm and the order of the two partial-derivative
outputs are unchanged.  Coupled display samples are not fed back into the
source integration, so all schedules remain finite and C=0 remains unchanged.
"""

from dataclasses import dataclass
import time

import numpy as np

from clifford import Cl
from PyQt5 import QtGui, QtWidgets
from PyQt5.QtCore import QPointF, Qt, QTimer
from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Line, Markers


# ----------------------------------------------------------------------
# 1. Coupling configuration
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class CouplingConfig:
    coefficient: float
    period: int
    description: str
    color: tuple[int, int, int]

    @property
    def short_label(self):
        return f"C={self.coefficient:g}"


COUPLING_CONFIGS = (
    CouplingConfig(0.0, 0, "normal gradient inputs", (45, 190, 255)),
    CouplingConfig(
        0.25,
        8,
        "swap gradient inputs every 8th sample",
        (255, 166, 77),
    ),
    CouplingConfig(
        0.5,
        4,
        "swap gradient inputs every 4th sample",
        (92, 220, 132),
    ),
    CouplingConfig(
        1.0,
        2,
        "swap gradient inputs every 2nd sample",
        (220, 105, 255),
    ),
)

# Rendering is deliberately decoupled from integration. Four independent
# OpenGL canvases are smooth at 30 visual updates per second, while wall-clock
# playback still advances through the correct number of simulation samples.
RENDER_INTERVAL_MS = 33
PATH_UPLOAD_CHUNK_SIZE = 64
TRAIL_DURATION_SECONDS = 0.9
MAX_TRAIL_SAMPLES = 256
GRAPH_PATH_BRIGHTNESS = 0.45
GRAPH_PATH_ALPHA = 0.72
TRAIL_OLDEST_BRIGHTNESS = 0.16
TRAIL_OLDEST_ALPHA = 0.04


@dataclass
class CoupledDisplayData:
    config: CouplingConfig
    angles: np.ndarray
    coordinate_gradients: np.ndarray
    positions: np.ndarray
    swap_mask: np.ndarray
    actual_swap_fraction: float


# ----------------------------------------------------------------------
# 2. Geometric algebra setup (3D Euclidean)
# ----------------------------------------------------------------------
layout, blades = Cl(3)
e1, e2, e3 = blades["e1"], blades["e2"], blades["e3"]


def scalar_part(multivector):
    """Return the scalar component of a Clifford scalar or multivector."""
    if hasattr(multivector, "value"):
        return float(multivector.value[0])
    return float(multivector)


# ----------------------------------------------------------------------
# 3. Spherical-pendulum model
# ----------------------------------------------------------------------
R_SPHERE = 1.0
MASS = 1.0
GRAVITY = 9.81


def position_from_angles(theta, phi):
    """Return the point on the unit sphere represented by theta and phi."""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    return R_SPHERE * (
        sin_theta * cos_phi * e1
        + sin_theta * sin_phi * e2
        + cos_theta * e3
    )


def velocity_from_angles(theta, phi, theta_dot, phi_dot):
    """Return the Cartesian tangent velocity as a GA vector."""
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    e_theta = (
        cos_theta * cos_phi * e1
        + cos_theta * sin_phi * e2
        - sin_theta * e3
    )
    e_phi = -sin_phi * e1 + cos_phi * e2
    return R_SPHERE * (
        theta_dot * e_theta + sin_theta * phi_dot * e_phi
    )


def lagrangian(theta, phi, theta_dot, phi_dot):
    """Return kinetic energy minus gravitational potential energy."""
    velocity = velocity_from_angles(theta, phi, theta_dot, phi_dot)
    kinetic = 0.5 * MASS * scalar_part(velocity | velocity)
    potential = MASS * GRAVITY * R_SPHERE * np.cos(theta)
    return kinetic - potential


def numerical_gradient(function, parameters, eps=1e-6):
    """Return independent central-difference partials as a NumPy vector."""
    point = np.asarray(parameters, dtype=np.float64)
    if point.ndim != 1 or point.size == 0:
        raise ValueError("parameters must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(point)):
        raise ValueError("parameters must contain only finite values")
    if not np.isfinite(eps) or eps <= 0.0:
        raise ValueError("eps must be positive and finite")

    gradient = np.empty(point.shape, dtype=np.float64)
    for index in range(len(point)):
        plus = point.copy()
        minus = point.copy()
        plus[index] += eps
        minus[index] -= eps
        gradient[index] = (
            function(plus) - function(minus)
        ) / (2.0 * eps)
    return gradient


def euler_lagrange_acceleration(q, q_dot, eps=1e-6):
    """Numerically solve the two Euler-Lagrange equations for q double-dot."""
    def lagrangian_at_q(q_value):
        return lagrangian(
            q_value[0],
            q_value[1],
            q_dot[0],
            q_dot[1],
        )

    def lagrangian_at_q_dot(q_dot_value):
        return lagrangian(
            q[0],
            q[1],
            q_dot_value[0],
            q_dot_value[1],
        )

    force = numerical_gradient(lagrangian_at_q, q, eps)

    def momentum_at_q_dot(q_dot_value):
        return numerical_gradient(lagrangian_at_q_dot, q_dot_value, eps)

    mass_matrix = np.zeros((2, 2), dtype=np.float64)
    for column in range(2):
        q_dot_plus = q_dot.copy()
        q_dot_minus = q_dot.copy()
        q_dot_plus[column] += eps
        q_dot_minus[column] -= eps
        mass_matrix[:, column] = (
            momentum_at_q_dot(q_dot_plus)
            - momentum_at_q_dot(q_dot_minus)
        ) / (2.0 * eps)

    def momentum_at_q(q_value):
        def lagrangian_with_fixed_q(q_dot_value):
            return lagrangian(
                q_value[0],
                q_value[1],
                q_dot_value[0],
                q_dot_value[1],
            )

        return numerical_gradient(lagrangian_with_fixed_q, q_dot, eps)

    q_plus = q + eps * q_dot
    q_minus = q - eps * q_dot
    convective_term = (
        momentum_at_q(q_plus) - momentum_at_q(q_minus)
    ) / (2.0 * eps)

    right_hand_side = force - convective_term
    return np.linalg.solve(mass_matrix, right_hand_side)


def ode_rhs(state):
    """Return the first-order form of the valid C=0 equations of motion."""
    theta, phi, theta_dot, phi_dot = state
    q = np.array([theta, phi], dtype=np.float64)
    q_dot = np.array([theta_dot, phi_dot], dtype=np.float64)
    q_double_dot = euler_lagrange_acceleration(q, q_dot)
    return np.array(
        [theta_dot, phi_dot, q_double_dot[0], q_double_dot[1]],
        dtype=np.float64,
    )


def rk4_step(state, dt):
    """Advance the valid C=0 state by one classical RK4 step."""
    k1 = ode_rhs(state)
    k2 = ode_rhs(state + 0.5 * dt * k1)
    k3 = ode_rhs(state + 0.5 * dt * k2)
    k4 = ode_rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate(initial_state, dt, number_of_steps, progress_callback=None):
    """Integrate the valid C=0 trajectory."""
    states = np.empty((number_of_steps + 1, 4), dtype=np.float64)
    states[0] = np.asarray(initial_state, dtype=np.float64)
    progress_interval = max(1, number_of_steps // 200)

    for step in range(1, number_of_steps + 1):
        states[step] = rk4_step(states[step - 1], dt)
        if not np.all(np.isfinite(states[step])):
            raise FloatingPointError(
                f"non-finite C=0 state encountered at integration step {step}"
            )
        if progress_callback and (
            step == number_of_steps or step % progress_interval == 0
        ):
            progress_callback(step)
    return states


# ----------------------------------------------------------------------
# 4. Derivative-input coupling schedules and trajectory conversion
# ----------------------------------------------------------------------
def periodic_swap_mask(number_of_samples, period):
    """Return a mask for one-based integration steps divisible by period."""
    if isinstance(period, (bool, np.bool_)) or not isinstance(
        period, (int, np.integer)
    ):
        raise TypeError("period must be a non-negative integer")
    period = int(period)
    if period < 0:
        raise ValueError("period must be a non-negative integer")

    mask = np.zeros(number_of_samples, dtype=bool)
    if period == 0 or number_of_samples <= 1:
        return mask

    # Index zero is the initial condition; indices 1..N are integration steps.
    step_indices = np.arange(1, number_of_samples, dtype=np.int64)
    mask[1:] = (step_indices % period) == 0
    return mask


def coordinate_gradient(angle_input, angle_rates, eps=1e-6):
    """Differentiate the Lagrangian normally at one ordered angle input.

    The caller decides whether ``angle_input`` is ``(theta, phi)`` or
    ``(phi, theta)``.  This function always returns the two central-difference
    partials in the same order as the supplied input; it never exchanges the
    calculated derivative components.
    """
    angles = np.asarray(angle_input, dtype=np.float64)
    rates = np.asarray(angle_rates, dtype=np.float64)
    if angles.shape != (2,) or rates.shape != (2,):
        raise ValueError("angle_input and angle_rates must each have shape (2,)")
    if not np.all(np.isfinite(angles)) or not np.all(np.isfinite(rates)):
        raise ValueError("angle inputs and rates must contain finite values")

    def lagrangian_at_angles(angle_values):
        return lagrangian(
            angle_values[0],
            angle_values[1],
            rates[0],
            rates[1],
        )

    return numerical_gradient(lagrangian_at_angles, angles, eps)


def trajectory_from_angle_history(angles):
    """Convert an entire theta/phi history to Cartesian sphere positions."""
    theta = np.asarray(angles[:, 0], dtype=np.float64)
    phi = np.asarray(angles[:, 1], dtype=np.float64)
    sin_theta = np.sin(theta)
    positions = np.column_stack(
        [
            R_SPHERE * sin_theta * np.cos(phi),
            R_SPHERE * sin_theta * np.sin(phi),
            R_SPHERE * np.cos(theta),
        ]
    )
    return positions.astype(np.float32)


def build_sphere_wireframe(number_of_parallels=20, number_of_meridians=28):
    """Return reusable latitude and longitude arrays for each sphere view."""
    circles = []

    for theta in np.linspace(0.0, np.pi, number_of_parallels):
        ring = []
        for phi in np.linspace(0.0, 2.0 * np.pi, number_of_meridians + 1):
            position = position_from_angles(theta, phi)
            ring.append(
                [
                    scalar_part(position | e1),
                    scalar_part(position | e2),
                    scalar_part(position | e3),
                ]
            )
        circles.append(np.asarray(ring, dtype=np.float32))

    for phi in np.linspace(0.0, 2.0 * np.pi, number_of_parallels):
        meridian = []
        for theta in np.linspace(0.0, np.pi, number_of_meridians + 1):
            position = position_from_angles(theta, phi)
            meridian.append(
                [
                    scalar_part(position | e1),
                    scalar_part(position | e2),
                    scalar_part(position | e3),
                ]
            )
        circles.append(np.asarray(meridian, dtype=np.float32))
    return circles


def line_strips_as_segments(line_strips):
    """Combine many wireframe strips into one GL ``segments`` vertex array.

    A sphere formerly used one VisPy visual per latitude and longitude. This
    representation preserves every segment while reducing each sphere's
    wireframe from roughly 40 draw calls to one.
    """
    segment_arrays = []
    for line_strip in line_strips:
        strip = np.asarray(line_strip, dtype=np.float32)
        if strip.ndim != 2 or strip.shape[1] != 3:
            raise ValueError("each line strip must have shape (N, 3)")
        if len(strip) < 2:
            continue

        segments = np.empty((2 * (len(strip) - 1), 3), dtype=np.float32)
        segments[0::2] = strip[:-1]
        segments[1::2] = strip[1:]
        segment_arrays.append(segments)

    if not segment_arrays:
        return np.zeros((0, 3), dtype=np.float32)
    return np.concatenate(segment_arrays, axis=0)


def build_trail_color_ramp(rgb_color, number_of_points):
    """Return RGBA colors fading from a dim tail to a bright head."""
    if isinstance(number_of_points, (bool, np.bool_)) or not isinstance(
        number_of_points,
        (int, np.integer),
    ):
        raise TypeError("number_of_points must be a positive integer")
    number_of_points = int(number_of_points)
    if number_of_points <= 0:
        raise ValueError("number_of_points must be a positive integer")

    rgb = np.asarray(rgb_color, dtype=np.float32)
    if rgb.shape != (3,) or not np.all(np.isfinite(rgb)):
        raise ValueError("rgb_color must contain three finite components")

    if number_of_points == 1:
        progress = np.ones(1, dtype=np.float32)
    else:
        progress = np.linspace(
            0.0,
            1.0,
            number_of_points,
            dtype=np.float32,
        )

    brightness = TRAIL_OLDEST_BRIGHTNESS + (
        1.0 - TRAIL_OLDEST_BRIGHTNESS
    ) * progress**0.75
    alpha = TRAIL_OLDEST_ALPHA + (
        1.0 - TRAIL_OLDEST_ALPHA
    ) * progress**1.6

    colors = np.empty((number_of_points, 4), dtype=np.float32)
    colors[:, :3] = rgb[np.newaxis, :] * brightness[:, np.newaxis]
    colors[:, 3] = alpha
    return colors


def build_simulation(dt=0.005, number_of_steps=3000, progress_callback=None):
    """Integrate C=0, then evaluate gradients at scheduled input orderings."""
    initial_state = np.array([0.8, 0.0, 0.0, 1.5], dtype=np.float64)
    number_of_samples = number_of_steps + 1
    swapped_gradient_work = sum(
        number_of_steps // config.period
        for config in COUPLING_CONFIGS
        if config.period > 0
    )
    total_work = (
        number_of_steps
        + number_of_samples
        + swapped_gradient_work
    )

    if progress_callback:
        progress_callback(
            0,
            total_work,
            "Integrating the valid C=0 Euler-Lagrange trajectory...",
        )

    def report_integrator_progress(step):
        if progress_callback:
            progress_callback(
                step,
                total_work,
                "Integrating the valid C=0 Euler-Lagrange trajectory...",
            )

    base_solution = integrate(
        initial_state,
        dt,
        number_of_steps,
        progress_callback=report_integrator_progress,
    )

    energy = np.empty(len(base_solution), dtype=np.float64)
    base_coordinate_gradients = np.empty(
        (len(base_solution), 2),
        dtype=np.float64,
    )
    conversion_interval = max(1, len(base_solution) // 200)
    for index, (theta, phi, theta_dot, phi_dot) in enumerate(base_solution):
        velocity = velocity_from_angles(theta, phi, theta_dot, phi_dot)
        velocity_squared = scalar_part(velocity | velocity)
        kinetic = 0.5 * MASS * velocity_squared
        potential = MASS * GRAVITY * R_SPHERE * np.cos(theta)
        energy[index] = kinetic + potential
        base_coordinate_gradients[index] = coordinate_gradient(
            (theta, phi),
            (theta_dot, phi_dot),
        )

        if progress_callback and (
            index == len(base_solution) - 1
            or index % conversion_interval == 0
        ):
            progress_callback(
                number_of_steps + index + 1,
                total_work,
                "Evaluating the normal C=0 coordinate gradients...",
            )

    variants = []
    completed_work = number_of_steps + number_of_samples
    base_angles = base_solution[:, :2]
    base_rates = base_solution[:, 2:]
    for config in COUPLING_CONFIGS:
        swap_mask = periodic_swap_mask(number_of_samples, config.period)
        angles = base_angles.copy()
        coordinate_gradients = base_coordinate_gradients.copy()

        swapped_indices = np.flatnonzero(swap_mask)
        if len(swapped_indices):
            # Swap the input values first. Each gradient is then calculated
            # normally at (phi, theta); its output tuple is never reversed.
            angles[swapped_indices] = angles[swapped_indices, ::-1]
            progress_interval = max(1, len(swapped_indices) // 100)
            for local_index, sample_index in enumerate(
                swapped_indices,
                start=1,
            ):
                coordinate_gradients[sample_index] = coordinate_gradient(
                    angles[sample_index],
                    base_rates[sample_index],
                )
                if progress_callback and (
                    local_index == len(swapped_indices)
                    or local_index % progress_interval == 0
                ):
                    progress_callback(
                        completed_work + local_index,
                        total_work,
                        f"Evaluating {config.short_label} gradients at "
                        "scheduled (phi, theta) inputs...",
                    )

        completed_work += len(swapped_indices)
        variants.append(
            CoupledDisplayData(
                config=config,
                angles=angles,
                coordinate_gradients=coordinate_gradients,
                positions=trajectory_from_angle_history(angles),
                swap_mask=swap_mask,
                actual_swap_fraction=float(
                    np.count_nonzero(swap_mask) / number_of_steps
                ),
            )
        )

    times = np.arange(len(base_solution), dtype=np.float64) * dt
    return base_solution, variants, energy, times


# ----------------------------------------------------------------------
# 5. Original single-series theta/phi trace widget
# ----------------------------------------------------------------------
class AngleTraceWidget(QtWidgets.QWidget):
    """Draw one of the original C=0 angle traces and its current frame."""

    def __init__(self, title, angle_name, line_color, parent=None):
        super().__init__(parent)
        self.title = title
        self.angle_name = angle_name
        self.line_color = QtGui.QColor(*line_color)
        self.time = np.array([], dtype=np.float64)
        self.values = np.array([], dtype=np.float64)
        self.current_index = 0
        self._polyline_cache = None
        self._polyline_cache_key = None
        self.setMinimumHeight(110)
        self.setStyleSheet("background:#090d14;")

    def set_data(self, time, values):
        self.time = np.asarray(time, dtype=np.float64)
        self.values = np.asarray(values, dtype=np.float64)
        if self.time.ndim != 1 or self.values.ndim != 1:
            raise ValueError("time and values must be one-dimensional arrays")
        if len(self.time) != len(self.values):
            raise ValueError("time and values must have equal lengths")
        self.current_index = 0
        self._polyline_cache = None
        self._polyline_cache_key = None
        self.update()

    def set_frame(self, index):
        if len(self.values) == 0:
            return
        self.current_index = int(np.clip(index, 0, len(self.values) - 1))
        self.update()

    def paintEvent(self, event):
        del event
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor("#090d14"))

        if len(self.time) < 2 or len(self.values) < 2:
            painter.end()
            return

        left, top, right, bottom = 78, 28, 18, 36
        plot = self.rect().adjusted(left, top, -right, -bottom)
        if plot.width() <= 8 or plot.height() <= 8:
            painter.end()
            return

        minimum = float(np.min(self.values))
        maximum = float(np.max(self.values))
        if abs(maximum - minimum) < 1e-9:
            minimum -= 1.0
            maximum += 1.0
        padding = 0.08 * (maximum - minimum)
        minimum -= padding
        maximum += padding
        time_minimum = float(self.time[0])
        time_maximum = float(self.time[-1])

        def map_x(time_value):
            return plot.left() + (
                (float(time_value) - time_minimum)
                / (time_maximum - time_minimum)
                * plot.width()
            )

        def map_y(value):
            return plot.bottom() - (
                (float(value) - minimum) / (maximum - minimum) * plot.height()
            )

        painter.setPen(QtGui.QPen(QtGui.QColor("#2b3444"), 1))
        painter.drawRect(plot)
        for fraction in (0.25, 0.5, 0.75):
            x = plot.left() + fraction * plot.width()
            y = plot.top() + fraction * plot.height()
            painter.drawLine(int(x), plot.top(), int(x), plot.bottom())
            painter.drawLine(plot.left(), int(y), plot.right(), int(y))

        painter.setFont(QtGui.QFont("Sans Serif", 9))
        painter.setPen(QtGui.QPen(QtGui.QColor("#d7e2f1"), 1))
        painter.drawText(10, 18, self.title)
        painter.drawText(10, plot.top() + 16, f"{self.angle_name} angle")

        painter.setPen(QtGui.QPen(QtGui.QColor("#92a0b3"), 1))
        painter.drawText(10, plot.top() + 4, f"{maximum:.2f} rad")
        painter.drawText(10, plot.bottom(), f"{minimum:.2f} rad")
        painter.drawText(plot.left(), self.height() - 10, f"{time_minimum:.1f}")
        painter.drawText(plot.right() - 34, self.height() - 10, f"{time_maximum:.1f}")
        painter.drawText(plot.center().x() - 30, self.height() - 10, "time (s)")

        # The curve is static throughout playback. Cache its mapped Qt points
        # and rebuild them only when the widget geometry or data range changes.
        cache_key = (
            plot.left(),
            plot.top(),
            plot.width(),
            plot.height(),
            minimum,
            maximum,
            time_minimum,
            time_maximum,
            len(self.values),
        )
        if self._polyline_cache_key != cache_key:
            points = [
                QPointF(map_x(time_value), map_y(value))
                for time_value, value in zip(self.time, self.values)
            ]
            self._polyline_cache = QtGui.QPolygonF(points)
            self._polyline_cache_key = cache_key
        painter.setPen(QtGui.QPen(self.line_color, 2))
        painter.drawPolyline(self._polyline_cache)

        current = self.current_index
        current_x = map_x(self.time[current])
        current_y = map_y(self.values[current])
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
        painter.drawLine(int(current_x), plot.top(), int(current_x), plot.bottom())
        painter.setBrush(QtGui.QBrush(QtGui.QColor("#ffd24a")))
        painter.drawEllipse(QPointF(current_x, current_y), 4.5, 4.5)

        value = float(self.values[current])
        painter.setPen(QtGui.QPen(QtGui.QColor("#ffffff"), 1))
        painter.drawText(
            plot.left() + 8,
            plot.top() + 18,
            f"t={self.time[current]:.2f}s   "
            f"{self.angle_name}={value:.4f} rad "
            f"({np.degrees(value):.1f} deg)",
        )
        painter.end()


# ----------------------------------------------------------------------
# 6. Four-sphere application
# ----------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def _add_sphere_panel(
        self,
        grid,
        row,
        column,
        config,
        wireframe_segments,
    ):
        """Add one independent Qt/VisPy sphere subwindow to the panel grid."""
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        panel.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        panel.setStyleSheet(
            "QFrame { background:#090d14; border:1px solid #263142; }"
            "QLabel { border:none; }"
        )
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(2, 2, 2, 2)
        panel_layout.setSpacing(2)

        red, green, blue = config.color
        title_label = QtWidgets.QLabel(
            f"{config.short_label} | {config.description}\n"
            "gradient input: (theta, phi)"
        )
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setWordWrap(True)
        title_label.setMinimumWidth(0)
        title_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        title_label.setStyleSheet(
            f"color:rgb({red}, {green}, {blue}); "
            "background:#090d14; padding:4px; font-size:10pt; "
            "font-weight:600; border:none;"
        )
        panel_layout.addWidget(title_label)

        # Each panel owns a distinct SceneCanvas, viewport, and camera.
        canvas = SceneCanvas(
            title=f"{config.short_label} spherical-pendulum view",
            size=(360, 430),
            bgcolor="black",
        )
        canvas.native.setParent(panel)
        canvas.native.setMinimumSize(160, 150)
        canvas.native.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        panel_layout.addWidget(canvas.native, stretch=1)

        view = canvas.central_widget.add_view()
        view.camera = scene.TurntableCamera(
            elevation=20,
            azimuth=45,
            distance=4.0,
        )
        view.camera.set_range(
            x=(-1.2, 1.2),
            y=(-1.2, 1.2),
            z=(-1.2, 1.2),
        )

        wireframe = Line(
            pos=wireframe_segments,
            color=(0.5, 0.55, 0.6, 0.38),
            width=1.0,
            connect="segments",
            method="gl",
            parent=view.scene,
        )

        normalized_color = tuple(
            component / 255.0 for component in config.color
        )
        graph_color = tuple(
            component * GRAPH_PATH_BRIGHTNESS
            for component in normalized_color
        ) + (GRAPH_PATH_ALPHA,)
        # The accumulated path is split into a rarely updated committed prefix
        # and a short active chunk. This avoids uploading an ever-growing VBO
        # to all four OpenGL contexts on every animation tick. It is kept dark
        # so the brighter fading motion trail is easy to follow.
        committed_line = Line(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=graph_color,
            width=2.5,
            connect="strip",
            method="gl",
            parent=view.scene,
        )
        active_line = Line(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=graph_color,
            width=2.5,
            connect="strip",
            method="gl",
            parent=view.scene,
        )
        trail_line = Line(
            pos=np.zeros((1, 3), dtype=np.float32),
            color=build_trail_color_ramp(normalized_color, 1),
            width=4.0,
            connect="strip",
            method="gl",
            parent=view.scene,
        )
        orbit_head = Markers(parent=view.scene)
        anchor = Markers(parent=view.scene)
        anchor.set_data(
            pos=np.array([[0.0, 0.0, R_SPHERE]], dtype=np.float32),
            face_color=(1.0, 0.75, 0.15, 0.95),
            edge_color=(0.0, 0.0, 0.0, 1.0),
            size=8,
        )

        grid.addWidget(panel, row, column)
        return {
            "config": config,
            "panel": panel,
            "canvas": canvas,
            "view": view,
            "title_label": title_label,
            "title_prefix": f"{config.short_label} | {config.description}",
            "displayed_state": "gradient input: (theta, phi)",
            "wireframe": wireframe,
            "committed_line": committed_line,
            "active_line": active_line,
            "committed_chunk_start": None,
            "trail_line": trail_line,
            "trail_length_samples": self.trail_length_samples,
            "trail_rgb": np.asarray(normalized_color, dtype=np.float32),
            "trail_color_cache": {},
            "trail_color_count": None,
            "orbit_head": orbit_head,
            "head_position_buffer": np.zeros((1, 3), dtype=np.float32),
            "normal_marker_color": normalized_color + (1.0,),
            "swap_marker_color": (1.0, 0.95, 0.2, 1.0),
            "anchor": anchor,
        }

    def __init__(self, dt=0.005, number_of_steps=3000):
        super().__init__()
        self.setWindowTitle(
            "Spherical Pendulum: Four Input-Swapped Derivative Views"
        )
        self.setWindowFlag(Qt.Window, True)
        self.setWindowFlag(Qt.WindowTitleHint, True)
        self.setWindowFlag(Qt.WindowSystemMenuHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        self.setMinimumSize(800, 600)
        self.setMaximumSize(16777215, 16777215)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.resize(1420, 900)

        if not np.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be positive and finite")
        if isinstance(number_of_steps, (bool, np.bool_)) or not isinstance(
            number_of_steps,
            (int, np.integer),
        ):
            raise TypeError("number_of_steps must be a positive integer")
        if number_of_steps <= 0:
            raise ValueError("number_of_steps must be a positive integer")

        self.dt = float(dt)
        self.number_of_steps = int(number_of_steps)
        self.trail_length_samples = min(
            MAX_TRAIL_SAMPLES,
            max(2, int(round(TRAIL_DURATION_SECONDS / self.dt))),
        )
        self.speed_multiplier = 1.0
        self.frame_cursor = 0.0
        self._last_playback_time = None
        self.current_index = 0
        self.variants = []
        self.sphere_panels = []

        central = QtWidgets.QWidget()
        central.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        self.setCentralWidget(central)
        outer_layout = QtWidgets.QVBoxLayout(central)
        outer_layout.setSizeConstraint(QtWidgets.QLayout.SetNoConstraint)
        outer_layout.setSpacing(6)

        instruction = QtWidgets.QLabel(
            "The valid C=0 Euler-Lagrange trajectory is integrated once. On "
            "a scheduled sample, a coupled view reorders the numerical-"
            "gradient input from (theta, phi) to (phi, theta), then runs the "
            "ordinary central difference normally; the returned partials "
            "are never swapped or fed back into C=0. C=0.25, C=0.5, and C=1 "
            "reorder inputs every 8th, 4th, and 2nd sample. The full path "
            "stays dark while recent points fade toward a bright moving head."
        )
        instruction.setWordWrap(True)
        instruction.setStyleSheet(
            "color:#e6eefb; background:#0f1520; padding:7px; font-size:10pt;"
        )
        outer_layout.addWidget(instruction)

        scene_title = QtWidgets.QLabel(
            "Independent sphere subwindows: C=0 | C=0.25 | C=0.5 | C=1"
        )
        scene_title.setAlignment(Qt.AlignCenter)
        scene_title.setStyleSheet(
            "color:#e6eefb; background:#090d14; padding:4px; font-size:11pt;"
        )
        outer_layout.addWidget(scene_title)

        sphere_area = QtWidgets.QWidget()
        sphere_area.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        sphere_grid = QtWidgets.QGridLayout(sphere_area)
        sphere_grid.setContentsMargins(0, 0, 0, 0)
        sphere_grid.setHorizontalSpacing(6)
        sphere_grid.setVerticalSpacing(0)
        for column in range(len(COUPLING_CONFIGS)):
            sphere_grid.setColumnMinimumWidth(column, 0)
            sphere_grid.setColumnStretch(column, 1)
        outer_layout.addWidget(sphere_area, stretch=5)

        wireframe_segments = line_strips_as_segments(build_sphere_wireframe())
        for column, config in enumerate(COUPLING_CONFIGS):
            self.sphere_panels.append(
                self._add_sphere_panel(
                    sphere_grid,
                    0,
                    column,
                    config,
                    wireframe_segments,
                )
            )

        diagnostics = QtWidgets.QWidget()
        diagnostics.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        diagnostics_layout = QtWidgets.QHBoxLayout(diagnostics)
        diagnostics_layout.setContentsMargins(0, 0, 0, 0)
        diagnostics_layout.setSpacing(7)
        outer_layout.addWidget(diagnostics, stretch=2)

        angle_panel = QtWidgets.QWidget()
        angle_layout = QtWidgets.QVBoxLayout(angle_panel)
        angle_layout.setContentsMargins(0, 0, 0, 0)
        angle_layout.setSpacing(6)

        self.theta_trace = AngleTraceWidget(
            "Theta angular movement: polar swing away from vertical",
            "theta",
            (68, 190, 255),
        )
        self.phi_trace = AngleTraceWidget(
            "Phi angular movement: azimuth around the vertical axis",
            "phi",
            (255, 151, 74),
        )
        angle_layout.addWidget(self.theta_trace)
        angle_layout.addWidget(self.phi_trace)
        diagnostics_layout.addWidget(angle_panel, stretch=1)

        controls = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls)
        outer_layout.addWidget(controls)

        self.play_button = QtWidgets.QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        step_button = QtWidgets.QPushButton("Step")
        step_button.clicked.connect(self.step_once)
        controls_layout.addWidget(step_button)

        reset_button = QtWidgets.QPushButton("Reset")
        reset_button.clicked.connect(self.reset_animation)
        controls_layout.addWidget(reset_button)

        controls_layout.addWidget(QtWidgets.QLabel("Speed"))
        self.speed_input = QtWidgets.QDoubleSpinBox()
        self.speed_input.setDecimals(2)
        self.speed_input.setRange(0.01, 20.0)
        self.speed_input.setSingleStep(0.1)
        self.speed_input.setValue(self.speed_multiplier)
        self.speed_input.setSuffix("x")
        self.speed_input.setFixedWidth(78)
        self.speed_input.setToolTip(
            "Playback multiplier: 1.00x plays simulation time in real time."
        )
        self.speed_input.valueChanged.connect(self.on_speed_changed)
        controls_layout.addWidget(self.speed_input)

        self.frame_slider = QtWidgets.QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.valueChanged.connect(self.on_slider_changed)
        controls_layout.addWidget(self.frame_slider, stretch=1)

        self.status_label = QtWidgets.QLabel()
        self.status_label.setMinimumWidth(0)
        self.status_label.setSizePolicy(
            QtWidgets.QSizePolicy.Ignored,
            QtWidgets.QSizePolicy.Preferred,
        )
        controls_layout.addWidget(self.status_label)

        self.timer = QTimer(self)
        self.timer.setInterval(RENDER_INTERVAL_MS)
        self.timer.setTimerType(Qt.PreciseTimer)
        self.timer.timeout.connect(self.advance_frame)

        self.status_bar = self.statusBar()
        self.simulate()
        self.update_frame(0)

    def simulate(self):
        progress = QtWidgets.QProgressDialog(
            "Preparing the C=0 trajectory and coupled gradient inputs...",
            "",
            0,
            self.number_of_steps,
            self,
        )
        progress.setWindowTitle("Generating four sphere histories")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setAutoClose(False)
        progress.setMinimumWidth(420)
        progress.show()
        QtWidgets.QApplication.processEvents()

        def report_progress(value, total, label):
            progress.setMaximum(total)
            progress.setLabelText(label)
            progress.setValue(value)
            QtWidgets.QApplication.processEvents()

        try:
            (
                self.base_solution,
                self.variants,
                self.energy,
                self.time,
            ) = build_simulation(
                self.dt,
                self.number_of_steps,
                report_progress,
            )
        finally:
            progress.setValue(progress.maximum())
            progress.close()

        self.path_length = len(self.base_solution)
        self.frame_slider.blockSignals(True)
        self.frame_slider.setMaximum(self.path_length - 1)
        self.frame_slider.setValue(0)
        self.frame_slider.blockSignals(False)

        self.theta_trace.set_data(
            self.time,
            self.base_solution[:, 0],
        )
        self.phi_trace.set_data(
            self.time,
            self.base_solution[:, 1],
        )

        initial_energy = float(self.energy[0])
        maximum_energy_drift = float(
            np.max(np.abs(self.energy - initial_energy))
        )
        fractions = " | ".join(
            f"{variant.config.short_label}:input-swap="
            f"{variant.actual_swap_fraction:.3f}"
            for variant in self.variants
        )
        self.status_bar.showMessage(
            f"C=0 source integration | dt={self.dt:.4f} | "
            f"steps={self.number_of_steps} | runtime={self.time[-1]:.2f}s | "
            f"max C=0 energy drift={maximum_energy_drift:.3e} | {fractions}"
        )

    def on_slider_changed(self, value):
        self.timer.stop()
        self._last_playback_time = None
        self.play_button.setText("Play")
        self.update_frame(value)

    def on_speed_changed(self, value):
        """Change playback speed without changing dt or recomputing data."""
        self.speed_multiplier = float(value)
        if self.timer.isActive():
            # Apply the new multiplier only to time elapsed after this change.
            self._last_playback_time = time.perf_counter()

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self._last_playback_time = None
            self.play_button.setText("Play")
            return

        if self.current_index >= self.path_length - 1:
            self.update_frame(0)
        self.frame_cursor = float(self.current_index)
        self._last_playback_time = time.perf_counter()
        self.timer.start()
        self.play_button.setText("Pause")

    def step_once(self):
        self.timer.stop()
        self._last_playback_time = None
        self.play_button.setText("Play")
        self.update_frame(min(self.current_index + 1, self.path_length - 1))

    def reset_animation(self):
        self.timer.stop()
        self._last_playback_time = None
        self.play_button.setText("Play")
        self.update_frame(0)

    def advance_frame(self, elapsed_seconds=None):
        """Advance according to elapsed wall time, even after a slow redraw."""
        if elapsed_seconds is None:
            current_time = time.perf_counter()
            if self._last_playback_time is None:
                elapsed_seconds = self.timer.interval() / 1000.0
            else:
                elapsed_seconds = current_time - self._last_playback_time
            self._last_playback_time = current_time
        elif not np.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")

        # Avoid an enormous jump after the process was suspended, while still
        # compensating for render stalls of up to one full second.
        elapsed_seconds = min(float(elapsed_seconds), 1.0)
        frames_elapsed = elapsed_seconds * self.speed_multiplier / self.dt
        self.frame_cursor = min(
            self.frame_cursor + frames_elapsed,
            float(self.path_length - 1),
        )
        next_index = int(self.frame_cursor)
        if next_index != self.current_index:
            self.update_frame(next_index, sync_cursor=False)
        if next_index >= self.path_length - 1:
            self.timer.stop()
            self._last_playback_time = None
            self.play_button.setText("Play")

    @staticmethod
    def _update_path_visual(panel, positions, current_index):
        """Update only a short active path chunk on routine redraws."""
        chunk_start = (
            current_index // PATH_UPLOAD_CHUNK_SIZE
        ) * PATH_UPLOAD_CHUNK_SIZE

        if panel["committed_chunk_start"] != chunk_start:
            panel["committed_line"].set_data(
                pos=positions[: chunk_start + 1]
            )
            panel["committed_chunk_start"] = chunk_start

        panel["active_line"].set_data(
            pos=positions[chunk_start : current_index + 1]
        )

        trail_start = max(
            0,
            current_index - panel["trail_length_samples"] + 1,
        )
        trail_positions = positions[trail_start : current_index + 1]
        trail_count = len(trail_positions)
        trail_colors = panel["trail_color_cache"].get(trail_count)
        if trail_colors is None:
            trail_colors = build_trail_color_ramp(
                panel["trail_rgb"],
                trail_count,
            )
            panel["trail_color_cache"][trail_count] = trail_colors
        if panel["trail_color_count"] == trail_count:
            panel["trail_line"].set_data(pos=trail_positions)
        else:
            panel["trail_line"].set_data(
                pos=trail_positions,
                color=trail_colors,
            )
            panel["trail_color_count"] = trail_count

    def update_frame(self, index, sync_cursor=True):
        self.current_index = int(np.clip(index, 0, self.path_length - 1))
        if sync_cursor:
            self.frame_cursor = float(self.current_index)

        active_input_swaps = []
        active_gradient_values = []
        for panel, variant in zip(self.sphere_panels, self.variants):
            self._update_path_visual(
                panel,
                variant.positions,
                self.current_index,
            )

            is_swapped = bool(variant.swap_mask[self.current_index])
            marker_color = (
                panel["swap_marker_color"]
                if is_swapped
                else panel["normal_marker_color"]
            )
            panel["head_position_buffer"][0] = variant.positions[
                self.current_index
            ]
            panel["orbit_head"].set_data(
                pos=panel["head_position_buffer"],
                face_color=marker_color,
                edge_color=(0.0, 0.0, 0.0, 1.0),
                size=11 if is_swapped else 9,
            )

            state_label = (
                "gradient input: (phi, theta)"
                if is_swapped
                else "gradient input: (theta, phi)"
            )
            if panel["displayed_state"] != state_label:
                panel["title_label"].setText(
                    f"{panel['title_prefix']}\n{state_label}"
                )
                panel["displayed_state"] = state_label
            if is_swapped:
                active_input_swaps.append(variant.config.short_label)
                gradient = variant.coordinate_gradients[self.current_index]
                active_gradient_values.append(
                    f"{variant.config.short_label}=({gradient[0]:.2f},"
                    f"{gradient[1]:.2f})"
                )

        base_theta, base_phi = self.base_solution[self.current_index, :2]
        swap_text = (
            ", ".join(active_input_swaps)
            if active_input_swaps
            else "none"
        )
        if active_gradient_values:
            gradient_text = "; ".join(active_gradient_values)
        else:
            gradient = self.variants[0].coordinate_gradients[
                self.current_index
            ]
            gradient_text = (
                f"C=0=({gradient[0]:.2f},{gradient[1]:.2f})"
            )
        self.status_label.setText(
            f"frame {self.current_index}/{self.path_length - 1}   "
            f"t={self.time[self.current_index]:.2f}s   "
            f"base theta={base_theta:.3f}   base phi={base_phi:.3f}   "
            f"swapped gradient inputs: {swap_text}   "
            f"normal gradient outputs: {gradient_text}"
        )

        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_index)
        self.frame_slider.blockSignals(False)

        self.theta_trace.set_frame(self.current_index)
        self.phi_trace.set_frame(self.current_index)


if __name__ == "__main__":
    app.use_app("pyqt5")
    qt_application = QtWidgets.QApplication.instance()
    if qt_application is None:
        qt_application = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    qt_application.exec_()
