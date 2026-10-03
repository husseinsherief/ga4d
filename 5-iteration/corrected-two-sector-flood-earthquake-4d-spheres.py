"""Corrected two-sector flood--earthquake 4D phase visualization.

This is a separate version of the sphere viewer for the corrected model:

  Flood sector F:   (t1, x1, t2, x2), with ∂t2(h, v1, v2) = 0
  Seismic sector S: (t1, x1, t3, x3)
  Shared surface C: (t1, x1)

The plotted modal state is X = (h, h v1, h v2, u3|C).  It is a stable,
single-spatial-mode visualization of the stated PDEs, not a mesh-resolved
numerical solution.  In particular, u3 is driven through the corrected
shared-surface term ∂x1(Λ g h), while the flood momentum receives the
moving-bed source from z_b = z_b0 - u3|x3=0.  On the selected binary
schedule, the earlier coupling-frequency construction additionally applies

    ΔX_coupled[k] = ((1-s[k]) I + s[k] P) ΔX[k]

to the completed modal increment.  Thus the corrected physical feedback and
the selectable swap coupling are both present and separately visible.

Requirements: numpy, PyQt5, vispy
Run: python corrected-two-sector-flood-earthquake-4d-spheres.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtWidgets
from vispy import app


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / 'coupled-flood-earthquake-4d-spheres.py'
SPEC = importlib.util.spec_from_file_location('corrected_sphere_base', BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f'Cannot load shared visualization code from {BASE_PATH}.')
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

# The three S² surfaces remain embedded in one parent S³, but their captions
# now describe the corrected four-coordinate modal state.
base.SPHERES = (
    ('S²₁  flood / interface planes', (0, 5), 'θ(h,hv₁) latitude  •  θ(hv₂,u₃) longitude', (0, 1, 2)),
    ('S²₂  shared-surface planes', (1, 4), 'θ(h,hv₂) latitude  •  θ(hv₁,u₃) longitude', (0, 1, 3)),
    ('S²₃  cross-sector planes', (2, 3), 'θ(h,u₃) latitude  •  θ(hv₁,hv₂) longitude', (0, 2, 3)),
)


def corrected_permutation_for(choice):
    """Coordinate transpositions for the corrected state X=(h,hv1,hv2,u3)."""
    return {
        0: np.array((1, 0, 2, 3)),  # h <-> hv1 (within flood sector)
        1: np.array((0, 3, 2, 1)),  # hv1 <-> u3 (shared interface)
        2: np.array((0, 1, 3, 2)),  # hv2 <-> u3 (cross-sector)
    }[choice]


def corrected_two_sector_modal_run(n_steps=2400, dt=0.012, period=6, permutation_choice=1):
    """Evolve a one-mode reduction of the corrected shared-surface system.

    h, p1, p2 are flood variables (p_i = h v_i) evolving in t1 only.
    u3 is the vertical seismic displacement on x3=0.  The scalar Λ is an
    imposed adjoint mode satisfying the visualized counterpart of the
    constraint; its x1 gradient provides the stated vertical forcing.
    The second-order seismic equation is integrated with a centred update,
    while the flood part uses explicit Euler.  Once those uncoupled candidate
    increments have been formed, the binary schedule applies the earlier
    coupling-frequency permutation to the whole increment vector.
    """
    gravity, rho = 1.0, 1.0
    k1, k2 = 0.78, 0.46
    lame_lambda, shear_mu = 0.42, 0.26
    seismic_omega, seismic_damping = 1.08, 0.055
    states = np.empty((n_steps + 1, 4), dtype=np.float64)
    forcing = np.empty(n_steps + 1, dtype=np.float64)
    schedule = base.deterministic_schedule(n_steps, period)
    permutation = corrected_permutation_for(permutation_choice)
    # h > 0 is maintained because it is the depth factor in h v_i.
    states[0] = (0.72, 0.10, -0.055, -0.22)
    previous_u3 = states[0, 3] - dt * 0.10

    for step, swapped in enumerate(schedule):
        h, p1, p2, u3 = states[step]
        phase = step * dt
        # One Fourier component of Λ on C=(t1,x1).  Its spatial derivative
        # supplies +∂x1(Λ g h) in the corrected u3 equation.
        lambda_gradient = 0.34 * np.cos(0.37 * phase)
        flood_force = lambda_gradient * gravity * h
        forcing[step] = flood_force

        # z_b = z_b0 - u3, so -g h ∂x1 z_b contributes +g h k1 u3
        # for this shared-surface Fourier mode.
        dh = -k1 * p1 - k2 * p2
        # Linear shallow-water pressure gradients act on the depth anomaly
        # about the unit mean depth, not on the absolute depth itself.
        eta = h - 1.0
        # The real modal quadrature uses p_i with the opposite phase to the
        # spatial derivative, giving the restoring +g k_i eta terms here.
        dp1 = (gravity * k1) * eta + 0.22 * gravity * h * k1 * u3 - 0.06 * p1
        dp2 = (gravity * k2) * eta - 0.045 * p2

        # A single vertical seismic mode of
        # rho (∂t1 + ∂t3)^2 u3 - elastic terms = ∂x1(Λ g h).
        elastic_frequency_sq = seismic_omega ** 2 + (lame_lambda + 2.0 * shear_mu) * k1 ** 2 / rho
        acceleration_u3 = flood_force / rho - elastic_frequency_sq * u3 - seismic_damping * (u3 - previous_u3) / dt
        next_u3 = 2.0 * u3 - previous_u3 + dt * dt * acceleration_u3

        # The flood variables are first order and u3 is second order, so form
        # their completed increments before applying the discrete coupling
        # operator. At s[k]=1, P exchanges the selected increments exactly as
        # in the earlier coupling-frequency model; at s[k]=0 this is the
        # corrected physical modal update unchanged.
        increment = np.array((dt * dh, dt * dp1, dt * dp2, next_u3 - u3))
        if swapped:
            increment = increment[permutation]
        states[step + 1] = states[step] + increment
        states[step + 1, 0] = max(0.08, states[step + 1, 0])
        previous_u3 = u3

    forcing[-1] = forcing[-2]
    angles = np.column_stack([np.arctan2(states[:, j], states[:, i]) for i, j in base.PAIRS])
    return states, angles, forcing, schedule, np.arange(n_steps + 1, dtype=np.float64) * dt


class CorrectedTwoSectorWindow(base.MainWindow):
    def _make_ui(self):
        super()._make_ui()
        self.setWindowTitle('Corrected Two-Sector Flood–Earthquake | 4D Phase Spheres')
        layout = self.centralWidget().layout()
        layout.itemAt(0).widget().setText(
            'CORRECTED FLOOD–EARTHQUAKE • TWO 4D SECTORS SHARING C = (t₁, x₁)'
        )
        layout.itemAt(1).widget().setText(
 C           'Flood sector F=(t₁,x₁,t₂,x₂): h, v₁, v₂ are independent of t₂.  Seismic sector '
            'S=(t₁,x₁,t₃,x₃): u=u₁γₓ₁+u₃γₓ₃.  The shared interface uses z_b=z_b₀−u₃|ₓ₃₌₀; '
            'the vertical seismic mode is forced by +∂ₓ₁(Λgh).  At scheduled steps, the selected '
            'coordinate increments are exchanged by ΔXᶜ=((1−sₖ)I+sₖP)ΔX.'
        )
        layout.itemAt(2).widget().setText(
            '<b style="color:#94a3b8">S³ — one parent hypersphere</b> &nbsp; | &nbsp; '
            '<b style="color:#24c7ff">S²₁ — flood/interface phase</b> &nbsp; | &nbsp; '
            '<b style="color:#fa853a">S²₂ — shared-surface phase</b> &nbsp; | &nbsp; '
            '<b style="color:#66e88d">S²₃ — cross-sector phase</b> '
            '&nbsp; • &nbsp; square = start; bright tail = recent direction; white-edged disc = current state'
        )
        for label, text in zip(self.axis_labels, ('X₁ = h', 'X₂ = hv₁', 'X₃ = hv₂', 'X₄ = u₃|𝒞')):
            label.text = text
        self.data_box.clear()
        self.data_box.addItem('Corrected two-sector one-mode reduction')
        self.data_box.setEnabled(False)
        self.period_box.setEnabled(True)
        self.permutation_box.clear()
        self.permutation_box.addItems((
            'h ↔ hv₁  (flood)',
            'hv₁ ↔ u₃  (shared interface)',
            'hv₂ ↔ u₃  (cross-sector)',
        ))
        self.permutation_box.setCurrentIndex(1)
        self.permutation_box.setEnabled(True)
        self.flood_trace.title = 'FLOOD DEPTH  h  (t₁ evolution)'
        self.bed_trace.title = 'FLOOD MOMENTUM  h v₁  (t₁ evolution)'
        self.schedule_trace.title = 'BINARY COUPLING SCHEDULE  sₖ  (red flash = swap)'

    def regenerate(self):
        self.timer.stop()
        self.play_button.setText('Play')
        self.states, self.angles, self.forcing, self.schedule, self.time = corrected_two_sector_modal_run(
            self.n_steps, self.dt, self._period(), self.permutation_box.currentIndex()
        )
        self.time_unit = 's'
        self.state_display = ('h', 'hv₁', 'hv₂', 'u₃')
        self.sphere_paths_r4 = [
            base.embed_in_r4(base.sphere_coordinates(self.angles[:, pair[0]], self.angles[:, pair[1]]), axes)
            for (_name, pair, _description, axes) in base.SPHERES
        ]
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.n_steps)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.flood_trace.set_series(self.states[:, 0])
        self.bed_trace.set_series(self.states[:, 1])
        self.schedule_trace.set_series(np.r_[self.schedule, self.schedule[-1]])
        self.update_frame(0)

    def update_frame(self, index):
        if not hasattr(self, 'states'):
            return
        super().update_frame(index)
        x = self.states[self.current_index]
        current_swap = int(self.schedule[self.current_index - 1]) if self.current_index else 0
        self.status.setText(
            f'k={self.current_index:04d}/{self.n_steps}  t₁={self.time[self.current_index]:5.2f}s  '
            f'Cₙ={self.schedule.mean():.3f}  sₖ={current_swap}  '
            f'h={x[0]:+.3f}  hv₁={x[1]:+.3f}  hv₂={x[2]:+.3f}  u₃|𝒞={x[3]:+.3f}  '
            f'∂ₓ₁(Λgh)={self.forcing[self.current_index]:+.3f}'
        )


if __name__ == '__main__':
    qt_app = QtWidgets.QApplication(sys.argv)
    app.use_app('pyqt5')
    window = CorrectedTwoSectorWindow()
    window.show()
    sys.exit(qt_app.exec_())
