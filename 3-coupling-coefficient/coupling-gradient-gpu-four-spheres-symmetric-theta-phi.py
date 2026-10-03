"""Run the four-sphere display with phi following theta at twice the speed.

This is a separate variant of ``coupling-gradient-gpu-four-spheres.py``.
The requested trajectory relation is

    phi(t) = theta(2 t).

For every displayed sample ``t_i = i * dt``, the program therefore uses

    theta_i     = theta_source(t_i)
    phi_i       = theta_source(2 t_i)
    theta_dot_i = theta_source_dot(t_i)
    phi_dot_i   = 2 theta_source_dot(2 t_i).

The authoritative theta source is integrated internally through twice the
visible duration.  Sampling that one source at indices ``i`` and ``2*i``
enforces the relation directly instead of hoping that two independent
floating-point integrations remain synchronized.

If theta obeys ``theta_ddot = F(theta)``, the chain rule gives

    phi_ddot(t) = 4 F(phi(t)).

Accordingly, phi's potential/force term is scaled by ``2**2 = 4`` while its
kinetic term retains the standard form.  The coupling rule is unchanged:

* C=0 uses the normal input ``(theta, phi)``.
* Scheduled samples first reorder the input to ``(phi, theta)``.
* The ordinary central-difference gradient is then calculated.
* The two returned derivative components are never manually exchanged.
* Coupled display samples are not fed back into the valid C=0 integration.

The point is still drawn with the standard spherical-coordinate map: theta is
the polar display coordinate and phi is the azimuth display coordinate.

This entry point intentionally reuses the optimized four-canvas GUI,
resizable window, theta/phi traces, playback speed control, and fading trails
from the sibling program.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from PyQt5 import QtWidgets
from vispy import app


def _load_base_program():
    """Load the sibling GUI under a private module name."""
    source_path = Path(__file__).with_name(
        "coupling-gradient-gpu-four-spheres.py"
    )
    module_name = "_coupling_gradient_four_spheres_base"
    specification = importlib.util.spec_from_file_location(
        module_name,
        source_path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load base program from {source_path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


BASE = _load_base_program()

PHI_TIME_SCALE = 8
PHI_FORCE_SCALE = PHI_TIME_SCALE**2
THETA_SOURCE_INITIAL_STATE = np.array(
    [0.8, 0.8, 0.0, 0.0],
    dtype=np.float64,
)


def theta_lagrangian(angle, angular_velocity):
    """Return the one-coordinate Lagrangian that defines theta(t)."""
    kinetic = (
        0.5
        * BASE.MASS
        * BASE.R_SPHERE**2
        * float(angular_velocity) ** 2
    )
    potential = (
        BASE.MASS
        * BASE.GRAVITY
        * BASE.R_SPHERE
        * np.cos(float(angle))
    )
    return float(kinetic - potential)


def phi_lagrangian(angle, angular_velocity):
    """Return the Lagrangian whose solution follows theta at twice its speed.

    Time compression by two multiplies the required acceleration—and
    therefore the coordinate-force term—by four.
    """
    kinetic = (
        0.5
        * BASE.MASS
        * BASE.R_SPHERE**2
        * float(angular_velocity) ** 2
    )
    potential = (
        PHI_FORCE_SCALE
        * BASE.MASS
        * BASE.GRAVITY
        * BASE.R_SPHERE
        * np.cos(float(angle))
    )
    return float(kinetic - potential)


def lagrangian(theta, phi, theta_dot, phi_dot):
    """Return theta's dynamics plus the chain-rule-scaled phi dynamics."""
    return (
        theta_lagrangian(theta, theta_dot)
        + phi_lagrangian(phi, phi_dot)
    )


def total_energy(theta, phi, theta_dot, phi_dot):
    """Return energy consistent with the twice-speed phi Lagrangian."""
    kinetic = (
        0.5
        * BASE.MASS
        * BASE.R_SPHERE**2
        * (float(theta_dot) ** 2 + float(phi_dot) ** 2)
    )
    potential = (
        BASE.MASS
        * BASE.GRAVITY
        * BASE.R_SPHERE
        * (
            np.cos(float(theta))
            + PHI_FORCE_SCALE * np.cos(float(phi))
        )
    )
    return float(kinetic + potential)


def _project_helper_state_to_theta(state):
    """Keep the unused helper coordinates equal during source integration.

    Only columns zero and two form the authoritative theta source.  The base
    RK4 routine expects a four-component state, so the helper phi columns are
    reset after each internal step.  The visible phi trajectory is constructed
    later by time-scaled sampling and does not use these helper columns.
    """
    selected_state = np.asarray(state, dtype=np.float64)
    if selected_state.shape != (4,):
        raise ValueError("state must have shape (4,)")
    projected = selected_state.copy()
    projected[1] = projected[0]
    projected[3] = projected[2]
    return projected


# Functions defined in the base module look up ``lagrangian`` in that module's
# global namespace when called. Installing this version makes its unchanged
# numerical-gradient pipeline use theta's normal law and phi's 4x force law.
BASE.lagrangian = lagrangian

# Do not require the sibling program to support custom initial-state or
# projection keyword arguments.  This local integrator preserves the original
# four-positional-argument calling convention used by older copies of
# coupling-gradient-gpu-four-spheres.py.
def integrate_theta_with_double_speed_phi(
    _base_initial_state,
    dt,
    number_of_steps,
    progress_callback=None,
    **_compatibility_options,
):
    """Return states satisfying phi(t_i) == theta(2*t_i) exactly.

    ``_base_initial_state`` and optional compatibility keywords are accepted
    because different revisions of the sibling program call ``integrate``
    differently.  This variant deliberately uses one authoritative theta
    source and integrates it for twice the requested visible duration.
    """
    del _base_initial_state, _compatibility_options

    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")
    if isinstance(number_of_steps, (bool, np.bool_)) or not isinstance(
        number_of_steps,
        (int, np.integer),
    ):
        raise TypeError("number_of_steps must be a positive integer")
    number_of_steps = int(number_of_steps)
    if number_of_steps <= 0:
        raise ValueError("number_of_steps must be a positive integer")

    source_steps = PHI_TIME_SCALE * number_of_steps
    source_states = np.empty((source_steps + 1, 4), dtype=np.float64)
    source_states[0] = _project_helper_state_to_theta(
        THETA_SOURCE_INITIAL_STATE
    )
    progress_interval = max(1, number_of_steps // 200)

    for source_step in range(1, source_steps + 1):
        source_states[source_step] = _project_helper_state_to_theta(
            BASE.rk4_step(source_states[source_step - 1], dt)
        )
        if not np.all(np.isfinite(source_states[source_step])):
            raise FloatingPointError(
                "non-finite theta source encountered at internal step "
                f"{source_step}"
            )

        if source_step % PHI_TIME_SCALE == 0:
            visible_step = source_step // PHI_TIME_SCALE
            if progress_callback and (
                visible_step == number_of_steps
                or visible_step % progress_interval == 0
            ):
                progress_callback(visible_step)

    visible_indices = np.arange(number_of_steps + 1, dtype=np.int64)
    phi_source_indices = PHI_TIME_SCALE * visible_indices
    states = np.empty((number_of_steps + 1, 4), dtype=np.float64)
    states[:, 0] = source_states[visible_indices, 0]
    states[:, 1] = source_states[phi_source_indices, 0]
    states[:, 2] = source_states[visible_indices, 2]
    states[:, 3] = (
        PHI_TIME_SCALE * source_states[phi_source_indices, 2]
    )

    return states


# The base build function resolves ``integrate`` through its module globals.
# Installing this adapter lets both old and new sibling revisions use the
# time-scaled invariant without requiring new build_simulation parameters.
BASE.integrate = integrate_theta_with_double_speed_phi

_BASE_BUILD_SIMULATION = BASE.build_simulation


def build_simulation(
    dt=0.005,
    number_of_steps=3000,
    progress_callback=None,
):
    """Build theta(t), phi(t)=theta(2t), and their energy diagnostic."""
    base_solution, variants, _old_energy, times = _BASE_BUILD_SIMULATION(
        dt,
        number_of_steps,
        progress_callback,
    )
    del _old_energy

    energy = np.fromiter(
        (
            total_energy(theta, phi, theta_dot, phi_dot)
            for theta, phi, theta_dot, phi_dot in base_solution
        ),
        dtype=np.float64,
        count=len(base_solution),
    )
    return base_solution, variants, energy, times


# MainWindow.simulate resolves build_simulation through the base module.
BASE.build_simulation = build_simulation


class MainWindow(BASE.MainWindow):
    """Four-sphere window specialized for phi(t) = theta(2t)."""

    def __init__(self, dt=0.005, number_of_steps=3000):
        super().__init__(dt=dt, number_of_steps=number_of_steps)
        self.setWindowTitle(
            "phi(t) = theta(2t): Four Input-Swapped Gradient Views"
        )

        # The first widget in the inherited outer layout is its explanatory
        # label. Replace only its text; all sizing and styling remain intact.
        outer_layout = self.centralWidget().layout()
        instruction_item = outer_layout.itemAt(0)
        instruction_label = (
            instruction_item.widget()
            if instruction_item is not None
            else None
        )
        if isinstance(instruction_label, QtWidgets.QLabel):
            instruction_label.setText(
                "Phi follows the authoritative theta trajectory at twice the "
                "speed: phi(t)=theta(2t) and phi_dot(t)=2 theta_dot(2t). "
                "Theta is computed internally through twice the visible "
                "duration, then sampled at i for theta and 2i for phi. By "
                "the chain rule, phi's force term is four times theta's. "
                "Scheduled (theta,phi) to (phi,theta) gradient-input "
                "reordering remains display-only; derivative outputs are "
                "never manually exchanged."
            )


# Convenient imports for notebooks or other Python programs.
numerical_gradient = BASE.numerical_gradient
coordinate_gradient = BASE.coordinate_gradient
periodic_swap_mask = BASE.periodic_swap_mask
COUPLING_CONFIGS = BASE.COUPLING_CONFIGS

__all__ = [
    "COUPLING_CONFIGS",
    "PHI_FORCE_SCALE",
    "PHI_TIME_SCALE",
    "THETA_SOURCE_INITIAL_STATE",
    "MainWindow",
    "build_simulation",
    "coordinate_gradient",
    "integrate_theta_with_double_speed_phi",
    "lagrangian",
    "numerical_gradient",
    "periodic_swap_mask",
    "phi_lagrangian",
    "theta_lagrangian",
    "total_energy",
]


if __name__ == "__main__":
    app.use_app("pyqt5")
    qt_application = QtWidgets.QApplication.instance()
    if qt_application is None:
        qt_application = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    qt_application.exec_()
