"""Real-data 4D phase visualization: 2011 Tohoku earthquake + tsunami.

This is a companion to ``coupled-flood-earthquake-4d-spheres.py``.  It keeps
the same one-parent-S³ / three-great-S² display, but its four coordinates are
actual observations sampled onto one UTC timeline:

    X = (IU.MAJO.00.BH1, IU.MAJO.00.BH2, IU.MAJO.00.BHZ, DART 21418 residual).

The MAJO channels are public broadband seismic velocity records from the
EarthScope FDSN service; the DART channel is NOAA bottom-pressure residual
water height.  The S² paths are a phase embedding of those observations, not
a calibrated tsunami-source inversion or a claim that DART directly measures
ground displacement.

No account is required.  Requirements: numpy, PyQt5, vispy.
Run: python tohoku-real-earthquake-tsunami-4d-spheres.py
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtWidgets
from vispy import app


HERE = Path(__file__).resolve().parent
BASE_PATH = HERE / 'coupled-flood-earthquake-4d-spheres.py'
SPEC = importlib.util.spec_from_file_location('phase_sphere_base', BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f'Cannot load shared visualization code from {BASE_PATH}.')
base = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(base)

# Compact, immutable event snapshot: 2,401 common UTC samples (15 minutes
# before to 60 minutes after the origin). It ships beside this program, so
# opening the application never waits on the network.
EVENT_SNAPSHOT = HERE / 'tohoku-2011-majo-dart-75min.npz'


def load_real_tohoku(n_steps):
    """Load the embedded public MAJO + DART event snapshot without networking."""
    try:
        with np.load(EVENT_SNAPSHOT) as event:
            source_seconds = event['seconds'].astype(np.float64)
            source_values = event['values'].astype(np.float64)
    except (ValueError, OSError) as exc:
        raise RuntimeError(f'Local event snapshot could not be read: {exc}') from exc
    duration = source_seconds[-1]
    target_seconds = np.linspace(0.0, duration, n_steps + 1)
    values = np.column_stack([
        np.interp(target_seconds, source_seconds, source_values[:, channel])
        for channel in range(source_values.shape[1])
    ])
    physical = dict(zip(('BH1', 'BH2', 'BHZ', 'DART'), values.T))
    states = np.column_stack([
        base._robust_scale(values[:, channel]) for channel in range(values.shape[1])
    ])
    angles = np.column_stack([np.arctan2(states[:, j], states[:, i]) for i, j in base.PAIRS])
    return states, angles, target_seconds, physical


class RealTohokuWindow(base.MainWindow):
    """Reuse the proven R⁴ scene while replacing the data pipeline and labels."""

    def _make_ui(self):
        super()._make_ui()
        self.setWindowTitle('2011 Tohoku | MAJO earthquake + DART tsunami | 4D Phase Spheres')
        layout = self.centralWidget().layout()
        layout.itemAt(0).widget().setText(
            '2011 TŌHOKU — REAL MAJO SEISMOGRAMS + NOAA DART 21418 WATER RESIDUAL '
            '• ONE PARENT S³ CONTAINING THREE GREAT S² SPHERES'
        )
        layout.itemAt(1).widget().setText(
            'Observed phase embedding on one UTC timeline: X = (MAJO BH1, MAJO BH2, MAJO BHZ seismic velocity, '
            'DART 21418 water-height residual).  All four channels are centred and robust-normalized only for the 4D '
            'angular mapping; the left plots retain physical units.  The neutral wire envelope is one S³; its three '
            'coloured great S² surfaces share one origin in R⁴.'
        )
        self.data_box.clear()
        self.data_box.addItem('2011 Tōhoku — EarthScope MAJO + NOAA DART (public)')
        self.data_box.setEnabled(False)
        self.period_box.setEnabled(False)
        self.permutation_box.setEnabled(False)
        self.flood_trace.title = 'MAJO BH1  •  seismic velocity (m/s)'
        self.bed_trace.title = 'MAJO BH2  •  seismic velocity (m/s)'
        self.schedule_trace.title = 'NOAA DART 21418  •  water residual (m)'
        self._loaded_observation = None

    def regenerate(self):
        self.timer.stop()
        self.play_button.setText('Play')
        try:
            if self._loaded_observation is None:
                self._loaded_observation = load_real_tohoku(self.n_steps)
            self.states, self.angles, seconds, self.physical = self._loaded_observation
        except RuntimeError as exc:
            QtWidgets.QMessageBox.critical(self, 'Public observational data unavailable', str(exc))
            return
        self.schedule = np.zeros(self.n_steps, dtype=np.int8)
        self.time, self.time_unit = seconds / 60.0, 'min'
        self.state_display = ('BH1', 'BH2', 'BHZ', 'DART')
        self.sphere_paths_r4 = [
            base.embed_in_r4(base.sphere_coordinates(self.angles[:, pair[0]], self.angles[:, pair[1]]), axes)
            for (_name, pair, _description, axes) in base.SPHERES
        ]
        self.slider.blockSignals(True)
        self.slider.setRange(0, self.n_steps)
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.flood_trace.set_series(self.physical['BH1'])
        self.bed_trace.set_series(self.physical['BH2'])
        self.schedule_trace.set_series(self.physical['DART'])
        self.update_frame(0)

    def update_frame(self, index):
        if not hasattr(self, 'states'):
            return
        super().update_frame(index)
        x = self.states[self.current_index]
        self.status.setText(
            f'UTC +{self.time[self.current_index]:5.2f} min  •  k={self.current_index:04d}/{self.n_steps}  '
            f'BH1={x[0]:+.2f}  BH2={x[1]:+.2f}  BHZ={x[2]:+.2f}  DART={x[3]:+.2f} '
            ' (normalized only in S³)'
        )


if __name__ == '__main__':
    qt_app = QtWidgets.QApplication(sys.argv)
    app.use_app('pyqt5')
    window = RealTohokuWindow()
    window.show()
    sys.exit(qt_app.exec_())
