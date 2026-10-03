"""Completed two-level-phi graphs viewed from the two sphere poles.

The trajectory data is the same data produced by
``coupling-gradient-gpu-four-spheres-two-level-phi.py``.  At frame 3000, the
entire accumulated graph for each coupling is drawn twice:

* the top row places the north pole at the center of the viewer;
* the bottom row places the south pole at the center of the viewer.

These are ordinary views of the complete graph after rotating the camera to
elevation +90 or -90 degrees.  No trajectory points are selected, separated,
projected into bins, or reconnected to manufacture geometric outlines.  The
perspective, azimuth, sphere wireframe, accumulated-path color, recent trail,
and final moving-point marker match the source four-sphere viewer.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt
from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Line, Markers


DT = 0.005
FINAL_FRAME = 3000
NORTH_ROW = 0
SOUTH_ROW = 1
POLE_AZIMUTH = 45.0
NORTH_ELEVATION = 90.0
SOUTH_ELEVATION = -90.0
CAMERA_DISTANCE = 4.0
CAMERA_FOV = 45.0


def _load_two_level_program():
    """Load the sibling trajectory program under a private module name."""
    source_path = Path(__file__).with_name(
        "coupling-gradient-gpu-four-spheres-two-level-phi.py"
    )
    module_name = "_coupling_gradient_two_level_phi_pole_view_source"
    specification = importlib.util.spec_from_file_location(
        module_name,
        source_path,
    )
    if specification is None or specification.loader is None:
        raise ImportError(f"could not load trajectory source from {source_path}")

    module = importlib.util.module_from_spec(specification)
    sys.modules[module_name] = module
    specification.loader.exec_module(module)
    return module


MODEL = _load_two_level_program()
BASE = MODEL.BASE


@dataclass(frozen=True)
class PoleView:
    """One fixed camera orientation and its viewer-facing sphere pole."""

    name: str
    row: int
    elevation: float
    pole_z: float

    @property
    def pole_position(self):
        return np.array(
            [[0.0, 0.0, self.pole_z * BASE.R_SPHERE]],
            dtype=np.float32,
        )


POLE_VIEWS = (
    PoleView(
        name="NORTH POLE VIEW",
        row=NORTH_ROW,
        elevation=NORTH_ELEVATION,
        pole_z=1.0,
    ),
    PoleView(
        name="SOUTH POLE VIEW",
        row=SOUTH_ROW,
        elevation=SOUTH_ELEVATION,
        pole_z=-1.0,
    ),
)


@dataclass(frozen=True)
class FinalTrajectoryData:
    """The complete final-frame graph for one coupling configuration."""

    config: object
    positions: np.ndarray
    final_input_is_swapped: bool


# Keep the old public type name usable for callers of the first envelope
# revision.  Its meaning is now the requested full trajectory, not an outline.
EnvelopeData = FinalTrajectoryData


def _positive_integer(value, name):
    """Validate and return one non-boolean positive integer."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a positive integer")
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def build_final_trajectories(
    final_frame=FINAL_FRAME,
    *,
    dt=DT,
    progress_callback=None,
):
    """Calculate and return each complete graph through ``final_frame``."""
    final_frame = _positive_integer(final_frame, "final_frame")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")

    _base_solution, variants, _energy, _times = MODEL.build_simulation(
        dt=float(dt),
        number_of_steps=final_frame,
        progress_callback=progress_callback,
    )

    trajectories = []
    expected_shape = (final_frame + 1, 3)
    for variant in variants:
        positions = np.asarray(
            variant.positions[: final_frame + 1],
            dtype=np.float32,
        )
        if positions.shape != expected_shape:
            raise ValueError(
                f"{variant.config.short_label} positions have shape "
                f"{positions.shape}; expected {expected_shape}"
            )
        if not np.all(np.isfinite(positions)):
            raise FloatingPointError(
                f"{variant.config.short_label} positions contain "
                "non-finite values"
            )

        swap_mask = np.asarray(variant.swap_mask, dtype=bool)
        if swap_mask.shape != (final_frame + 1,):
            raise ValueError(
                f"{variant.config.short_label} swap mask has shape "
                f"{swap_mask.shape}; expected {(final_frame + 1,)}"
            )

        trajectories.append(
            FinalTrajectoryData(
                config=variant.config,
                positions=positions.copy(),
                final_input_is_swapped=bool(swap_mask[final_frame]),
            )
        )
    return trajectories


def extract_final_envelopes(progress_callback=None):
    """Compatibility wrapper returning the requested full final graphs."""
    return build_final_trajectories(
        FINAL_FRAME,
        dt=DT,
        progress_callback=progress_callback,
    )


def normalized_rgb(rgb):
    """Convert an integer RGB tuple to a float32 vector."""
    result = np.asarray(rgb, dtype=np.float32) / 255.0
    if result.shape != (3,):
        raise ValueError("rgb must contain exactly three values")
    return result


def accumulated_graph_color(rgb):
    """Return the dim RGBA color used by the source accumulated graph."""
    base_rgb = normalized_rgb(rgb)
    return tuple(
        (base_rgb * BASE.GRAPH_PATH_BRIGHTNESS).tolist()
    ) + (BASE.GRAPH_PATH_ALPHA,)


def recent_trail(positions, dt=DT):
    """Return the source viewer's bright recent tail at the final frame."""
    points = np.asarray(positions, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("positions must have shape (number_of_samples, 3)")
    if len(points) == 0:
        raise ValueError("positions cannot be empty")
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be positive and finite")

    trail_length = min(
        BASE.MAX_TRAIL_SAMPLES,
        max(2, int(round(BASE.TRAIL_DURATION_SECONDS / float(dt)))),
    )
    return points[max(0, len(points) - trail_length) :]


class EnvelopeWindow(QtWidgets.QMainWindow):
    """Static 2-by-4 comparison of the complete north/south pole views."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(
            "Frame 3000: Two-Level Phi Graphs from North and South Poles"
        )
        self.setWindowFlag(Qt.Window, True)
        self.setWindowFlag(Qt.WindowTitleHint, True)
        self.setWindowFlag(Qt.WindowSystemMenuHint, True)
        self.setWindowFlag(Qt.WindowMinimizeButtonHint, True)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, True)
        self.setWindowFlag(Qt.WindowCloseButtonHint, True)
        self.setMinimumSize(900, 650)
        self.resize(1520, 980)
        self.canvases = []
        self.trajectories = []

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer_layout = QtWidgets.QVBoxLayout(central)
        outer_layout.setContentsMargins(7, 7, 7, 7)
        outer_layout.setSpacing(6)

        description = QtWidgets.QLabel(
            "The same complete frame-3000 graphs as the two-level phi "
            "viewer are shown from two fixed camera rotations. Top: camera "
            "at +90° elevation, with the north pole centered from the "
            "viewer's perspective. Bottom: camera at −90° elevation, with "
            "the south pole centered. Each gold center marker identifies "
            "the viewer-facing pole; no points are extracted or rebuilt."
        )
        description.setWordWrap(True)
        description.setStyleSheet(
            "color:#e6eefb; background:#0f1520; padding:8px; font-size:10pt;"
        )
        outer_layout.addWidget(description)

        progress = QtWidgets.QProgressDialog(
            "Calculating the complete frame-3000 graphs...",
            "",
            0,
            FINAL_FRAME,
            self,
        )
        progress.setWindowTitle("Building pole-view graphs")
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)
        progress.setCancelButton(None)
        progress.setAutoClose(False)
        progress.setMinimumWidth(430)
        progress.show()
        QtWidgets.QApplication.processEvents()

        def report_progress(value, total, label):
            progress.setMaximum(total)
            progress.setLabelText(label)
            progress.setValue(value)
            QtWidgets.QApplication.processEvents()

        try:
            self.trajectories = build_final_trajectories(
                FINAL_FRAME,
                dt=DT,
                progress_callback=report_progress,
            )
        finally:
            progress.setValue(progress.maximum())
            progress.close()

        plot_area = QtWidgets.QWidget()
        plot_grid = QtWidgets.QGridLayout(plot_area)
        plot_grid.setContentsMargins(0, 0, 0, 0)
        plot_grid.setHorizontalSpacing(6)
        plot_grid.setVerticalSpacing(6)
        for column in range(len(self.trajectories)):
            plot_grid.setColumnStretch(column, 1)
        for pole_view in POLE_VIEWS:
            plot_grid.setRowStretch(pole_view.row, 1)
        outer_layout.addWidget(plot_area, stretch=1)

        wireframe_segments = BASE.line_strips_as_segments(
            BASE.build_sphere_wireframe()
        )
        for pole_view in POLE_VIEWS:
            for column, trajectory in enumerate(self.trajectories):
                self._add_trajectory_panel(
                    plot_grid,
                    pole_view,
                    column,
                    trajectory,
                    wireframe_segments,
                )

        self.statusBar().showMessage(
            "Frame 3000 complete graphs | north elevation +90° | "
            "south elevation −90° | azimuth 45° | perspective FOV 45°"
        )

    def _add_trajectory_panel(
        self,
        grid,
        pole_view,
        column,
        trajectory,
        wireframe_segments,
    ):
        """Add one fixed pole-facing view of one complete trajectory."""
        panel = QtWidgets.QFrame()
        panel.setFrameShape(QtWidgets.QFrame.StyledPanel)
        panel.setStyleSheet(
            "QFrame { background:#090d14; border:1px solid #263142; }"
            "QLabel { border:none; }"
        )
        panel_layout = QtWidgets.QVBoxLayout(panel)
        panel_layout.setContentsMargins(2, 2, 2, 2)
        panel_layout.setSpacing(2)

        red, green, blue = trajectory.config.color
        input_state = (
            "(phi, theta)"
            if trajectory.final_input_is_swapped
            else "(theta, phi)"
        )
        title = QtWidgets.QLabel(
            f"{pole_view.name} | {trajectory.config.short_label}\n"
            f"complete 0–{FINAL_FRAME} graph | final input: {input_state}"
        )
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet(
            f"color:rgb({red}, {green}, {blue}); "
            "background:#090d14; padding:3px; "
            "font-size:9.5pt; font-weight:600;"
        )
        panel_layout.addWidget(title)

        canvas = SceneCanvas(
            title=(
                f"{pole_view.name} {trajectory.config.short_label}"
            ),
            size=(360, 360),
            bgcolor="black",
        )
        canvas.native.setParent(panel)
        canvas.native.setMinimumSize(150, 150)
        canvas.native.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding,
            QtWidgets.QSizePolicy.Expanding,
        )
        panel_layout.addWidget(canvas.native, stretch=1)
        self.canvases.append(canvas)

        view = canvas.central_widget.add_view()
        camera = scene.TurntableCamera(
            fov=CAMERA_FOV,
            elevation=pole_view.elevation,
            azimuth=POLE_AZIMUTH,
            distance=CAMERA_DISTANCE,
        )
        camera.set_range(
            x=(-1.2, 1.2),
            y=(-1.2, 1.2),
            z=(-1.2, 1.2),
        )
        camera.interactive = False
        view.camera = camera

        Line(
            pos=wireframe_segments,
            color=(0.5, 0.55, 0.6, 0.38),
            width=1.0,
            connect="segments",
            method="gl",
            parent=view.scene,
        )

        points = trajectory.positions
        Line(
            pos=points,
            color=accumulated_graph_color(trajectory.config.color),
            width=2.5,
            connect="strip",
            method="gl",
            parent=view.scene,
        )

        trail_points = recent_trail(points, DT)
        trail_rgb = normalized_rgb(trajectory.config.color)
        Line(
            pos=trail_points,
            color=BASE.build_trail_color_ramp(
                trail_rgb,
                len(trail_points),
            ),
            width=4.0,
            connect="strip",
            method="gl",
            parent=view.scene,
        )

        head_color = (
            (1.0, 0.95, 0.2, 1.0)
            if trajectory.final_input_is_swapped
            else tuple(trail_rgb.tolist()) + (1.0,)
        )
        orbit_head = Markers(parent=view.scene)
        orbit_head.set_data(
            pos=points[-1:],
            face_color=head_color,
            edge_color=(0.0, 0.0, 0.0, 1.0),
            size=11 if trajectory.final_input_is_swapped else 9,
        )

        focused_pole = Markers(parent=view.scene)
        focused_pole.set_data(
            pos=pole_view.pole_position,
            face_color=(1.0, 0.75, 0.15, 0.95),
            edge_color=(0.0, 0.0, 0.0, 1.0),
            size=8,
        )

        grid.addWidget(panel, pole_view.row, column)


__all__ = [
    "CAMERA_DISTANCE",
    "CAMERA_FOV",
    "DT",
    "EnvelopeData",
    "EnvelopeWindow",
    "FINAL_FRAME",
    "FinalTrajectoryData",
    "NORTH_ELEVATION",
    "POLE_AZIMUTH",
    "POLE_VIEWS",
    "SOUTH_ELEVATION",
    "PoleView",
    "accumulated_graph_color",
    "build_final_trajectories",
    "extract_final_envelopes",
    "normalized_rgb",
    "recent_trail",
]


if __name__ == "__main__":
    app.use_app("pyqt5")
    qt_application = QtWidgets.QApplication.instance()
    if qt_application is None:
        qt_application = QtWidgets.QApplication([])
    window = EnvelopeWindow()
    window.show()
    qt_application.exec_()
