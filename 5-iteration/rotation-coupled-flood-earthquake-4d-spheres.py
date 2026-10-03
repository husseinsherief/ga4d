"""Rotation-coupled flood--earthquake fields on three S2 spheres in one S3.

This program keeps the window, projection, and hypersphere construction from
``passive-time-coupled-flood-earthquake-4d-spheres.py`` while replacing its
shared-surface/binary-schedule coupling with the continuous rotation model

    Phi' = R(theta) Phi,

where Phi = (u1, u3, eta, v1, v2).  The displayed depth is h = 1 + eta.  The
constant generator J mixes (u1, eta) and (u3, v1), and annihilates v2.

The modal equations include the Euler, Coriolis-like, and centrifugal terms
from the supplied expanded component equations.  theta and theta_dot are
integrated as dynamical variables.  The theta equation uses the stated local
quotient with a small inertia floor so that it remains defined when the modal
amplitude passes close to zero.

Each displayed S2 is parameterized by two circles:

    flood sphere:      (t1, x1) and (t2, x2)
    earthquake sphere: (t1, x1) and (t3, x3)
    coupled sphere:    the two planes mixed by R(theta)

Phase of each circle is shown as a bead + trail on that great circle (six
on-ring trajectories).  A chord joins each sphere's two beads.

All three S2 surfaces are embedded in the xyz, xyw, and xzw hyperplanes of a
single parent S3 hypersphere in R4.  This is a stable one-mode visualization,
not a spatially resolved PDE solver.

Requirements: numpy, PyQt5, vispy
Run: .venv/bin/python rotation-coupled-flood-earthquake-4d-spheres.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from vispy import app


HERE = Path(__file__).resolve().parent
VIEWER_PATH = HERE / 'passive-time-coupled-flood-earthquake-4d-spheres.py'
SPEC = importlib.util.spec_from_file_location('rotation_coupled_viewer_base', VIEWER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f'Cannot load the shared hypersphere viewer from {VIEWER_PATH}.')
viewer_base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(viewer_base)


ETA, V1, V2, U1, U3, W1, W3, THETA, OMEGA = range(9)
STATE_NAMES = (
    'eta', 'v1', 'v2', 'u1', 'u3', 'du1/dt1', 'du3/dt1', 'theta', 'dtheta/dt1',
)

SPHERE_AXES = viewer_base.SPHERE_AXES
ROTATION_GENERATOR = np.array((
    (0.0, 0.0, -1.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, -1.0, 0.0),
    (1.0, 0.0, 0.0, 0.0, 0.0),
    (0.0, 1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 0.0, 0.0),
), dtype=np.float64)

# Four useful initial rotor states.  omega is measured in radians per t1 unit.
ROTATION_PRESETS = (
    ('aligned / slow', 0.08, 0.06),
    ('gentle rotation', 0.24, 0.12),
    ('balanced rotation', np.pi / 4.0, 0.18),
    ('strong rotation', 1.08, 0.25),
)


def rotation_matrix(theta):
    """Return the supplied five-component flood/seismic mixing rotation."""
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))
    return np.array((
        (cosine, 0.0, -sine, 0.0, 0.0),
        (0.0, cosine, 0.0, -sine, 0.0),
        (sine, 0.0, cosine, 0.0, 0.0),
        (0.0, sine, 0.0, cosine, 0.0),
        (0.0, 0.0, 0.0, 0.0, 1.0),
    ), dtype=np.float64)


def rotation_matrix_derivative(theta):
    """Return dR/dtheta for :func:`rotation_matrix`."""
    cosine = float(np.cos(theta))
    sine = float(np.sin(theta))
    return np.array((
        (-sine, 0.0, -cosine, 0.0, 0.0),
        (0.0, -sine, 0.0, -cosine, 0.0),
        (cosine, 0.0, -sine, 0.0, 0.0),
        (0.0, cosine, 0.0, -sine, 0.0),
        (0.0, 0.0, 0.0, 0.0, 0.0),
    ), dtype=np.float64)


def combined_state(state):
    """Return Phi = (u1, u3, eta, v1, v2) for the modal perturbations."""
    return np.array(
        (state[U1], state[U3], state[ETA], state[V1], state[V2]),
        dtype=np.float64,
    )


def _physical_modal_terms(state):
    """Evaluate the uncoupled modal PDE terms using the rotated bed/depth.

    The second row of the stated rotation is the rotated vertical-displacement
    channel: u3' = cos(theta) u3 - sin(theta) v1.  This is the component used
    for the moving-bed contribution.  The rotated eta channel supplies the
    flood-to-earthquake forcing.
    """
    eta, v1, v2, u1, u3, w1, w3, theta = state[:8]
    depth = 1.0 + eta
    rotated = rotation_matrix(theta) @ combined_state(state)
    rotated_u3 = rotated[1]
    rotated_eta = rotated[2]

    gravity, density = 1.0, 1.0
    k1, k2, k3 = 0.78, 0.46, 0.62
    lame_lambda, shear_mu = 0.42, 0.26
    flood_damping_1, flood_damping_2 = 0.060, 0.045
    seismic_damping = 0.055
    bed_to_flood = 0.22
    lambda_mode = 0.34

    d_eta = -k1 * v1 - k2 * v2
    d_v1 = (
        gravity * k1 * eta
        + bed_to_flood * gravity * depth * k1 * rotated_u3
        - flood_damping_1 * v1
    )
    d_v2 = gravity * k2 * eta - flood_damping_2 * v2

    wave_number_sq = k1**2 + k3**2
    longitudinal = lame_lambda + 2.0 * shear_mu
    stiffness_11 = longitudinal * k1**2 + shear_mu * wave_number_sq
    stiffness_33 = longitudinal * k3**2 + shear_mu * wave_number_sq
    stiffness_13 = longitudinal * k1 * k3
    flood_force = lambda_mode * gravity * k1 * rotated_eta
    acceleration_u1 = (
        -stiffness_11 * u1 - stiffness_13 * u3
    ) / density - seismic_damping * w1
    acceleration_u3 = (
        -stiffness_13 * u1 - stiffness_33 * u3 + flood_force
    ) / density - seismic_damping * w3

    return np.array((
        d_eta, d_v1, d_v2, w1, w3, acceleration_u1, acceleration_u3,
    ), dtype=np.float64), flood_force


def _rotation_acceleration(state, base_field):
    """Evaluate the supplied theta equation with a local directional second derivative."""
    phi = combined_state(state)
    phi_dot = np.array((
        base_field[3], base_field[4], base_field[0], base_field[1], base_field[2],
    ))

    # Directionally differentiate the physical Phi_dot along its own local
    # trajectory.  This evaluates the Phi_ddot term in the stated theta
    # equation without introducing another stored derivative level.
    probe_dt = 1.0e-5
    probe = state.copy()
    probe[:7] += probe_dt * base_field
    probe[THETA] += probe_dt * state[OMEGA]
    probe_field, _ = _physical_modal_terms(probe)
    probe_phi_dot = np.array((
        probe_field[3], probe_field[4], probe_field[0],
        probe_field[1], probe_field[2],
    ))
    phi_ddot = (probe_phi_dot - phi_dot) / probe_dt

    omega = state[OMEGA]
    inertia_floor = 0.28
    denominator = float(phi @ phi) + inertia_floor
    numerator = 2.0 * omega * float(phi @ phi_dot)
    numerator += float(phi_ddot @ ROTATION_GENERATOR @ phi)
    return -numerator / denominator


def rotation_coupled_modal_field(state):
    """Return the nine-component first-order rotation-coupled field."""
    base_field, flood_force = _physical_modal_terms(state)
    alpha = _rotation_acceleration(state, base_field)
    eta, v1, _v2, u1, u3, w1, w3, _theta, omega = state

    # A modal mass ratio controls how strongly the five-component rotor acts
    # on this normalized one-mode reduction.  It is not a binary blend: every
    # time step evaluates all three rotation-force terms continuously.
    modal_mass_ratio = 0.085
    d_eta = base_field[0] - modal_mass_ratio * (
        alpha * u1 + 2.0 * omega * w1 - omega**2 * eta
    )
    d_v1 = base_field[1] - modal_mass_ratio * (
        alpha * u3 + 2.0 * omega * w3 - omega**2 * v1
    )
    d_v2 = base_field[2]
    acceleration_u1 = base_field[5] + modal_mass_ratio * (
        alpha * eta + 2.0 * omega * d_eta - omega**2 * u1
    )
    acceleration_u3 = base_field[6] + modal_mass_ratio * (
        alpha * v1 + 2.0 * omega * d_v1 - omega**2 * u3
    )

    field = np.array((
        d_eta, d_v1, d_v2, w1, w3,
        acceleration_u1, acceleration_u3, omega, alpha,
    ), dtype=np.float64)
    return field, flood_force, alpha


RING_TRAIL_LENGTH = 180

# Local great-circle bases in each sphere's (u,v,w) chart:
#   0 = uv-plane, 1 = uw-plane, 2 = vw-plane
# Pairing avoids shared R4 2-planes across the three embeddings:
#   flood xyz:      xy + yz
#   earthquake xyw: xw + yw
#   coupled xzw:    xz + zw
SPHERE_CIRCLE_BASES = ((0, 2), (1, 2), (0, 2))
# Concentric radii so stacked origins still separate the three S2 rings.
SPHERE_RADII = (
    float(viewer_base.SPHERE_RADIUS),
    float(viewer_base.SPHERE_RADIUS) * 0.82,
    float(viewer_base.SPHERE_RADIUS) * 0.66,
)


def circle_phase_series(states):
    """Return shape (3, 2, n) phases: sphere × (azimuthal, longitude) × time."""
    eta = states[:, ETA]
    depth = 1.0 + eta
    v1, v2 = states[:, V1], states[:, V2]
    u1, u3 = states[:, U1], states[:, U3]
    w1, w3 = states[:, W1], states[:, W3]

    flood_shared = np.arctan2(v1, depth)
    flood_higher = np.arctan2(v2, depth)
    seismic_shared = np.arctan2(u1, w1)
    seismic_higher = np.arctan2(u3, w3)

    phi = np.column_stack((u1, u3, eta, v1, v2))
    rotated = np.empty_like(phi)
    for index, (angle, vector) in enumerate(zip(states[:, THETA], phi)):
        rotated[index] = rotation_matrix(angle) @ vector
    mixed_u1_eta = np.arctan2(rotated[:, 2], rotated[:, 0])
    mixed_u3_v1 = np.arctan2(rotated[:, 3], rotated[:, 1])

    return np.stack((
        np.stack((flood_shared, flood_higher)),
        np.stack((seismic_shared, seismic_higher)),
        np.stack((mixed_u1_eta, mixed_u3_v1)),
    ), axis=0).astype(np.float64)


def local_circle_points(phases, basis_id, radius):
    """Sample a local great circle: basis 0=uv, 1=uw, 2=vw."""
    phases = np.asarray(phases, dtype=np.float64)
    cosine = np.cos(phases)
    sine = np.sin(phases)
    zeros = np.zeros_like(cosine)
    radius = float(radius)
    basis = int(basis_id)
    if basis == 0:
        local = np.column_stack((radius * cosine, radius * sine, zeros))
    elif basis == 1:
        local = np.column_stack((radius * cosine, zeros, radius * sine))
    elif basis == 2:
        local = np.column_stack((zeros, radius * cosine, radius * sine))
    else:
        raise ValueError('basis_id must be 0, 1, or 2')
    return local.astype(np.float32)


def point_on_circle(phase, basis_id, radius=None):
    """Map a scalar phase onto a local great-circle basis."""
    if radius is None:
        radius = float(viewer_base.SPHERE_RADIUS)
    return local_circle_points(np.asarray([phase], dtype=np.float64), basis_id, radius)[0]


def points_on_circle(phases, basis_id, radius=None):
    """Vectorized `point_on_circle` for a 1-D phase array."""
    if radius is None:
        radius = float(viewer_base.SPHERE_RADIUS)
    return local_circle_points(phases, basis_id, radius)


def sphere_circle_basis(sphere_id, circle_id):
    return SPHERE_CIRCLE_BASES[int(sphere_id)][int(circle_id)]


def sphere_radius(sphere_id):
    return SPHERE_RADII[int(sphere_id)]


def build_sphere_circle_geometry(samples=181):
    """Return six distinct R4 rings: two non-overlapping planes per S2."""
    angle = np.linspace(0.0, 2.0 * np.pi, samples, dtype=np.float64)
    geometry = []
    for sphere_id, axes in enumerate(SPHERE_AXES):
        radius = sphere_radius(sphere_id)
        for circle_id in (0, 1):
            basis = sphere_circle_basis(sphere_id, circle_id)
            local = local_circle_points(angle, basis, radius)
            points_r4 = viewer_base.base.embed_in_r4(local, axes)
            geometry.append((sphere_id, circle_id, points_r4))
    return geometry


def r4_plane_key(points_r4, tol=1.0e-6):
    """Return which R4 coordinate pair a ring spans (for uniqueness checks)."""
    spans = np.ptp(points_r4, axis=0)
    active = tuple(int(index) for index, span in enumerate(spans) if span > tol)
    return active


def chord_alpha(sphere_id, coupling_c, emphasized):
    """Return chord opacity per the approved ring-bead design."""
    if not emphasized:
        return 0.08
    if int(sphere_id) == 2:
        return float(0.15 + 0.75 * np.clip(coupling_c, 0.0, 1.0))
    return 0.35


def circle_rgb(color, circle_id):
    """Paired sphere hues: circle 0 full, circle 1 lifted toward white for visibility."""
    rgb = tuple(float(channel) for channel in color[:3])
    if int(circle_id) == 0:
        return rgb
    # Dark backgrounds hide darkened siblings; lift longitude rings instead.
    return tuple(channel * 0.78 + 0.22 for channel in rgb)


def circle_line_width(circle_id):
    """Keep both rings bold; circle 0 is only slightly thicker."""
    return 2.8 if int(circle_id) == 0 else 2.5


def circle_ring_alpha(circle_id, emphasized):
    if not emphasized:
        return 0.16
    return 0.95 if int(circle_id) == 0 else 0.92


def sphere_phase_paths(states):
    """Build the flood, earthquake, and rotation-mixed two-circle paths."""
    phases = circle_phase_series(states)
    return (
        viewer_base._sphere_coordinates(phases[0, 0], phases[0, 1]),
        viewer_base._sphere_coordinates(phases[1, 0], phases[1, 1]),
        viewer_base._sphere_coordinates(phases[2, 0], phases[2, 1]),
    )


def simulate_rotation_coupled_system(n_steps=2400, dt=0.012, preset=2):
    """Integrate the continuous rotation-coupled modal system in t1."""
    if n_steps < 1:
        raise ValueError('n_steps must be at least 1.')
    if dt <= 0.0:
        raise ValueError('dt must be positive.')
    if not 0 <= preset < len(ROTATION_PRESETS):
        raise ValueError(f'preset must be between 0 and {len(ROTATION_PRESETS) - 1}.')

    _name, theta_initial, omega_initial = ROTATION_PRESETS[preset]
    states = np.empty((n_steps + 1, 9), dtype=np.float64)
    forcing = np.empty(n_steps + 1, dtype=np.float64)
    angular_acceleration = np.empty(n_steps + 1, dtype=np.float64)
    states[0] = (
        -0.28, 0.10, -0.055, 0.08, -0.22, 0.025, 0.10,
        theta_initial, omega_initial,
    )

    for step in range(n_steps):
        field, forcing[step], angular_acceleration[step] = rotation_coupled_modal_field(
            states[step]
        )
        states[step + 1] = states[step] + dt * field

    _, forcing[-1], angular_acceleration[-1] = rotation_coupled_modal_field(states[-1])
    time = np.arange(n_steps + 1, dtype=np.float64) * dt
    coupling = np.sin(states[:, THETA]) ** 2
    coupling_flux = np.sin(2.0 * states[:, THETA]) * states[:, OMEGA]
    paths = sphere_phase_paths(states)
    phases = circle_phase_series(states)
    return states, paths, forcing, angular_acceleration, coupling, coupling_flux, time, phases


class RotationCoupledWindow(viewer_base.PassiveTimeCoupledWindow):
    """Resizable three-sphere viewer for continuous rotation-based coupling."""

    def _make_ui(self):
        super()._make_ui()
        self.setWindowTitle(
            'Rotation-Coupled Flood–Earthquake | Three S² Surfaces in One S³'
        )

        root_layout = self.centralWidget().layout()
        heading = root_layout.itemAt(0).widget()
        heading.setText(
            'ROTATION-COUPLED FLOOD–EARTHQUAKE  •  THREE S² SURFACES IN ONE PARENT S³'
        )
        explainer = root_layout.itemAt(1).widget()
        explainer.setText(
            'The sectors couple through the dynamical higher-order-axis rotation '
            'Φ′ = R(θ)Φ. Every update evaluates the rotated bed/depth plus the Euler, '
            'Coriolis-like, and centrifugal terms; θ and θ̇ evolve with the fields. '
            'Each S² shows two defining rings; phase beads travel on those rings.'
        )
        legend = root_layout.itemAt(2).widget()
        legend.setText(
            '<b style="color:#94a3b8">PARENT S³</b>: one R⁴ hypersphere'
            ' &nbsp; | &nbsp; <b style="color:#24c7ff">FLOOD F / xyz</b>: '
            'azimuthal (t₁,x₁) + longitude (t₂,x₂)'
            ' &nbsp; | &nbsp; <b style="color:#fa9430">EARTHQUAKE S / xyw</b>: '
            'azimuthal (t₁,x₁) + longitude (t₃,x₃)'
            ' &nbsp; | &nbsp; <b style="color:#33e67a">COUPLED R(θ)Φ / xzw</b>: '
            '(u₁,η) mix + (u₃,v₁) mix'
        )
        self.hypersphere_title.setText(
            '<b style="color:#94a3b8">ONE PARENT S³</b> &nbsp; • &nbsp; '
            '<span style="color:#24c7ff">FLOOD F · 2 rings</span> &nbsp; • &nbsp; '
            '<span style="color:#fa9430">EARTHQUAKE S · 2 rings</span> &nbsp; • &nbsp; '
            '<span style="color:#33e67a">R(θ)Φ · 2 rings</span>'
        )

        for label in self.centralWidget().findChildren(QtWidgets.QLabel):
            if label.text() == 'coupling cadence:':
                label.setText('initial rotor:')
                break
        self.period_box.clear()
        self.period_box.addItems(tuple(item[0] for item in ROTATION_PRESETS))
        self.period_box.setCurrentIndex(2)
        self.period_box.setAccessibleName('Choose the initial rotation state')

        old_trace = self.schedule_trace
        trace_layout = old_trace.parentWidget().layout()
        trace_index = trace_layout.indexOf(old_trace)
        trace_layout.removeWidget(old_trace)
        old_trace.hide()
        old_trace.deleteLater()
        self.coupling_trace = viewer_base.base.DiagnosticWidget(
            'ROTATION COUPLING  C = sin²θ', (51, 230, 122)
        )
        trace_layout.insertWidget(trace_index, self.coupling_trace)
        self.schedule_trace = self.coupling_trace

        self.play_button.setAccessibleName('Play or pause the rotation-coupled simulation')

        from vispy.scene.visuals import Line, Markers

        # Hide legacy free-space S2 path visuals (v1 ring-bead view).
        for visual in (*self.paths, *self.trails, *self.starts, *self.heads, *self.swap_heads):
            visual.visible = False

        # Replace the inherited shared-plane rings with six distinct R4 circles.
        self.circle_geometry = build_sphere_circle_geometry()
        for geom_index, line in enumerate(self.circle_lines):
            sphere_id, circle_id, points_r4 = self.circle_geometry[geom_index]
            rgb = circle_rgb(viewer_base.SPHERE_COLORS[sphere_id], circle_id)
            line.set_data(
                pos=self._project(points_r4),
                color=(*rgb, 0.92),
                width=circle_line_width(circle_id),
            )

        self.ring_trails = []
        self.ring_beads = []
        for sphere_id, color in enumerate(viewer_base.SPHERE_COLORS):
            for circle_id in (0, 1):
                rgb = circle_rgb(color, circle_id)
                self.ring_trails.append(
                    Line(
                        color=(*rgb, 0.70),
                        width=3.4 if circle_id == 0 else 3.0,
                        connect='strip',
                        parent=self.view.scene,
                    )
                )
                self.ring_beads.append(Markers(parent=self.view.scene))

        self.coupling_chords = [
            Line(color=(*color[:3], 0.35), width=1.6, connect='strip', parent=self.view.scene)
            for color in viewer_base.SPHERE_COLORS
        ]

    def regenerate(self):
        self.timer.stop()
        self.play_button.setText('Play')
        application = QtWidgets.QApplication.instance()
        if application is not None:
            application.setOverrideCursor(QtCore.Qt.WaitCursor)
            application.processEvents()
        try:
            (
                self.states,
                self.local_paths,
                self.forcing,
                self.angular_acceleration,
                self.coupling,
                self.coupling_flux,
                self.time,
                self.circle_phases,
            ) = simulate_rotation_coupled_system(
                self.n_steps, self.dt, self.period_box.currentIndex()
            )
        finally:
            if application is not None:
                application.restoreOverrideCursor()

        # The inherited scene uses schedule only for optional binary-event
        # flashes.  Rotation coupling is continuous, so there are no flashes.
        self.schedule = np.zeros(self.n_steps, dtype=np.int8)
        self.sphere_paths_r4 = tuple(
            viewer_base.base.embed_in_r4(points, axes)
            for points, axes in zip(self.local_paths, SPHERE_AXES)
        )
        self.timeline.blockSignals(True)
        self.timeline.setRange(0, self.n_steps)
        self.timeline.setValue(0)
        self.timeline.blockSignals(False)
        self.flood_trace.set_series(1.0 + self.states[:, ETA])
        self.earthquake_trace.set_series(self.states[:, U3])
        self.coupling_trace.set_series(self.coupling)
        self.update_frame(0)

    def update_scene(self, *_unused):
        if not hasattr(self, 'circle_phases'):
            return
        focus = self.focus_box.currentIndex() - 1
        index = self.current_index

        for line, raw in zip(self.hypersphere_lines, self.hypersphere_geometry):
            line.set_data(
                pos=self._project(raw),
                color=(0.58, 0.64, 0.74, 0.16 if focus < 0 else 0.10),
            )

        trail_i = 0
        for sphere_id, color in enumerate(viewer_base.SPHERE_COLORS):
            emphasized = focus < 0 or sphere_id == focus
            axes = SPHERE_AXES[sphere_id]
            radius = sphere_radius(sphere_id)
            pair_r3 = []
            for circle_id in (0, 1):
                basis = sphere_circle_basis(sphere_id, circle_id)
                rgb = circle_rgb(color, circle_id)
                ring_alpha = circle_ring_alpha(circle_id, emphasized)
                geom_index = sphere_id * 2 + circle_id
                line = self.circle_lines[geom_index]
                _sid, _cid, raw = self.circle_geometry[geom_index]
                line.set_data(
                    pos=self._project(raw),
                    color=(*rgb, ring_alpha),
                    width=circle_line_width(circle_id),
                )

                phases = self.circle_phases[sphere_id, circle_id, : index + 1]
                start = max(0, len(phases) - RING_TRAIL_LENGTH)
                trail_phases = phases[start:]
                local = points_on_circle(trail_phases, basis, radius)
                trail_r4 = viewer_base.base.embed_in_r4(local, axes)
                visible = self._project(trail_r4)
                if len(visible) == 0:
                    visible = self._project(
                        viewer_base.base.embed_in_r4(
                            point_on_circle(0.0, basis, radius).reshape(1, 3), axes
                        )
                    )
                tail_alpha = np.linspace(0.12, 1.0, len(visible), dtype=np.float32)
                tail_alpha *= 1.0 if emphasized else 0.16
                trail_colors = np.column_stack((
                    np.full(len(visible), rgb[0]),
                    np.full(len(visible), rgb[1]),
                    np.full(len(visible), rgb[2]),
                    tail_alpha,
                )).astype(np.float32)
                self.ring_trails[trail_i].set_data(pos=visible, color=trail_colors)

                bead_local = point_on_circle(float(phases[-1]), basis, radius)
                bead_r4 = viewer_base.base.embed_in_r4(bead_local.reshape(1, 3), axes)
                bead_r3 = self._project(bead_r4)
                pair_r3.append(bead_r3[0])
                head_alpha = 1.0 if emphasized else 0.22
                self.ring_beads[trail_i].set_data(
                    pos=bead_r3,
                    face_color=(*rgb, head_alpha),
                    edge_color=(0.98, 0.99, 1.0, head_alpha),
                    size=12 if circle_id == 0 else 11,
                    symbol='disc',
                )
                trail_i += 1

            chord_pts = np.vstack(pair_r3).astype(np.float32)
            c_value = float(self.coupling[index])
            alpha = chord_alpha(sphere_id, c_value, emphasized)
            self.coupling_chords[sphere_id].set_data(
                pos=chord_pts,
                color=(*color[:3], alpha),
            )

        for visual in (*self.paths, *self.trails, *self.starts, *self.heads, *self.swap_heads):
            visual.visible = False

    def update_frame(self, index):
        self.current_index = int(np.clip(index, 0, self.n_steps))
        self.flood_trace.set_frame(self.current_index)
        self.earthquake_trace.set_frame(self.current_index)
        self.coupling_trace.set_frame(self.current_index)
        self.timeline.blockSignals(True)
        self.timeline.setValue(self.current_index)
        self.timeline.blockSignals(False)

        state = self.states[self.current_index]
        rotated = rotation_matrix(state[THETA]) @ combined_state(state)
        self.status.setText(
            f'k={self.current_index:04d}/{self.n_steps}    '
            f't₁={self.time[self.current_index]:6.2f}s    '
            f'θ={state[THETA]:+.3f} rad    θ̇={state[OMEGA]:+.3f}    '
            f'C=sin²θ={self.coupling[self.current_index]:.3f}    '
            f'ḊC={self.coupling_flux[self.current_index]:+.3f}    '
            f'h={1.0 + state[ETA]:+.3f}    v₁={state[V1]:+.3f}    '
            f'u₃={state[U3]:+.3f}    u₃′={rotated[1]:+.3f}    '
            f'θ̈={self.angular_acceleration[self.current_index]:+.3f}'
        )
        self.update_scene()

if __name__ == '__main__':
    qt_app = QtWidgets.QApplication(sys.argv)
    app.use_app('pyqt5')
    window = RotationCoupledWindow()
    window.show()
    sys.exit(qt_app.exec_())
