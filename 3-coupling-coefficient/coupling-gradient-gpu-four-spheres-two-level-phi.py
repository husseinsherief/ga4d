"""Four-sphere display with a two-level, twice-speed phi trajectory.

Theta follows the same one-angle numerical Euler-Lagrange trajectory used by
the related programs.  Phi is a derived two-level copy:

    phi(t) = Q(theta(2 t))

where

    Q(a) = 0.43 rad   when wrapped a is below pi
    Q(a) = 5.86 rad   when wrapped a is at or above pi.

Thus phi contains only the two requested values while its transition pattern
follows the approximate shape of theta's angular movement at twice the speed.
The authoritative theta trajectory is calculated through twice the visible
duration, so every displayed phi sample uses an actually calculated theta
sample at time ``2*t``.

The scheduled C=0, C=0.25, C=0.5, and C=1 gradient-input reordering remains
display-only.  Numerical-gradient outputs are never manually exchanged.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np
from PyQt5 import QtWidgets
from vispy import app


def _load_base_program():
    """Load the optimized sibling GUI under a private module name."""
    source_path = Path(__file__).with_name(
        "coupling-gradient-gpu-four-spheres.py"
    )
    module_name = "_coupling_gradient_two_level_phi_base"
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

PHI_TIME_SCALE = 2
PHI_LOW_LEVEL = 0.43
PHI_HIGH_LEVEL = 5.86
PHI_THRESHOLD = np.pi
TWO_PI = 2.0 * np.pi

# Only theta's value and rate are authoritative.  The helper phi entries let
# the inherited four-component RK4 implementation evaluate theta normally.
THETA_SOURCE_INITIAL_STATE = np.array(
    [0.8, 0.8, 0.0, 0.0],
    dtype=np.float64,
)


def angle_lagrangian(angle, angular_velocity):
    """Return the shared one-angle function used to calculate theta."""
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


def lagrangian(theta, phi, theta_dot, phi_dot):
    """Return two separable angle components for numerical differentiation."""
    return (
        angle_lagrangian(theta, theta_dot)
        + angle_lagrangian(phi, phi_dot)
    )


def two_level_phi(theta_values):
    """Map theta values to exactly 0.43 or 5.86 radians.

    Inputs are wrapped to ``[0, 2*pi)`` before comparison with ``pi``.  Scalars
    return a float; arrays return a float64 array with the same shape.
    """
    theta_array = np.asarray(theta_values, dtype=np.float64)
    if not np.all(np.isfinite(theta_array)):
        raise ValueError("theta_values must contain only finite values")

    wrapped_theta = np.mod(theta_array, TWO_PI)
    result = np.where(
        wrapped_theta < PHI_THRESHOLD,
        PHI_LOW_LEVEL,
        PHI_HIGH_LEVEL,
    )
    if result.ndim == 0:
        return float(result)
    return result.astype(np.float64, copy=False)


def _project_helper_state_to_theta(state):
    """Reset the inherited RK4 helper coordinates to theta's state."""
    selected_state = np.asarray(state, dtype=np.float64)
    if selected_state.shape != (4,):
        raise ValueError("state must have shape (4,)")
    projected = selected_state.copy()
    projected[1] = projected[0]
    projected[3] = projected[2]
    return projected


# The inherited derivative functions resolve this name through their module
# globals, so theta continues to use the ordinary numerical-gradient pipeline.
BASE.lagrangian = lagrangian


def integrate_theta_with_two_level_phi(
    _base_initial_state,
    dt,
    number_of_steps,
    progress_callback=None,
    **_compatibility_options,
):
    """Build theta(t) and the exact sampled relation phi(t)=Q(theta(2t)).

    The broad signature supports both older and newer revisions of the sibling
    GUI without requiring changes to that file.
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
    twice_speed_indices = PHI_TIME_SCALE * visible_indices

    states = np.empty((number_of_steps + 1, 4), dtype=np.float64)
    states[:, 0] = source_states[visible_indices, 0]
    states[:, 1] = two_level_phi(
        source_states[twice_speed_indices, 0]
    )
    states[:, 2] = source_states[visible_indices, 2]

    # Each level is constant between transitions.  At a discontinuous
    # transition there is no finite classical derivative, so the stored phi
    # rate is defined as zero rather than introducing an artificial spike.
    states[:, 3] = 0.0
    return states


# The base build function resolves ``integrate`` through its module globals.
BASE.integrate = integrate_theta_with_two_level_phi
_BASE_BUILD_SIMULATION = BASE.build_simulation


def theta_source_energy(theta, theta_dot):
    """Return the conserved-energy diagnostic for authoritative theta."""
    kinetic = (
        0.5
        * BASE.MASS
        * BASE.R_SPHERE**2
        * float(theta_dot) ** 2
    )
    potential = (
        BASE.MASS
        * BASE.GRAVITY
        * BASE.R_SPHERE
        * np.cos(float(theta))
    )
    return float(kinetic + potential)


def build_simulation(
    dt=0.005,
    number_of_steps=3000,
    progress_callback=None,
):
    """Build the four displays and use theta's source-energy diagnostic."""
    base_solution, variants, _old_energy, times = _BASE_BUILD_SIMULATION(
        dt,
        number_of_steps,
        progress_callback,
    )
    del _old_energy

    energy = np.fromiter(
        (
            theta_source_energy(theta, theta_dot)
            for theta, theta_dot in base_solution[:, (0, 2)]
        ),
        dtype=np.float64,
        count=len(base_solution),
    )
    return base_solution, variants, energy, times


# MainWindow.simulate resolves build_simulation through the base module.
BASE.build_simulation = build_simulation


class MainWindow(BASE.MainWindow):
    """Four-sphere window with a two-level, twice-speed phi function."""

    def __init__(self, dt=0.005, number_of_steps=3000):
        super().__init__(dt=dt, number_of_steps=number_of_steps)
        self.setWindowTitle(
            "Two-Level phi(t) from theta(2t): Four Coupling Views"
        )

        outer_layout = self.centralWidget().layout()
        instruction_item = outer_layout.itemAt(0)
        instruction_label = (
            instruction_item.widget()
            if instruction_item is not None
            else None
        )
        if isinstance(instruction_label, QtWidgets.QLabel):
            instruction_label.setText(
                "Phi is the two-level function Q(theta(2t)). It is exactly "
                "0.43 rad while wrapped theta(2t) is below pi and exactly "
                "5.86 rad while it is at or above pi. Theta is calculated "
                "internally through twice the visible duration, making phi's "
                "transitions follow theta at twice the speed. Scheduled "
                "(theta,phi) to (phi,theta) gradient-input reordering remains "
                "display-only; derivative outputs are never exchanged."
            )


numerical_gradient = BASE.numerical_gradient
coordinate_gradient = BASE.coordinate_gradient
periodic_swap_mask = BASE.periodic_swap_mask
COUPLING_CONFIGS = BASE.COUPLING_CONFIGS

__all__ = [
    "COUPLING_CONFIGS",
    "MainWindow",
    "PHI_HIGH_LEVEL",
    "PHI_LOW_LEVEL",
    "PHI_THRESHOLD",
    "PHI_TIME_SCALE",
    "THETA_SOURCE_INITIAL_STATE",
    "angle_lagrangian",
    "build_simulation",
    "coordinate_gradient",
    "integrate_theta_with_two_level_phi",
    "lagrangian",
    "numerical_gradient",
    "periodic_swap_mask",
    "theta_source_energy",
    "two_level_phi",
]


if __name__ == "__main__":
    app.use_app("pyqt5")
    qt_application = QtWidgets.QApplication.instance()
    if qt_application is None:
        qt_application = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    qt_application.exec_()
