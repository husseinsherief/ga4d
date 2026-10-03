import numpy as np
from clifford.g4 import layout

from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Line, Text
from vispy.scene.widgets.label import Label
from vispy.visuals import Visual
from vispy.gloo import Program, VertexBuffer
from vispy.visuals.transforms import TransformSystem

from PyQt5 import QtWidgets
from PyQt5.QtCore import Qt, QTimer


# ----------------------------------------------------------------------
# 4D rotation matrix
# ----------------------------------------------------------------------
def rotation_matrix_4d(angle, plane):
    R = np.eye(4, dtype=np.float32)
    i, j = plane
    c = np.cos(angle)
    s = np.sin(angle)
    R[i, i] = c
    R[i, j] = -s
    R[j, i] = s
    R[j, j] = c
    return R


# ----------------------------------------------------------------------
# Data builder
# ----------------------------------------------------------------------
def build_data(geom_scale, n_theta=8, n_phi=12):
    e1, e2, e3, e4 = layout.basis_vectors.values()

    theta1_vals = np.linspace(0.01, np.pi - 0.01, n_theta)
    theta2_vals = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    verts_4d = []
    rotors_4d = []
    idx_map = {}

    for i, t1 in enumerate(theta1_vals):
        for j, t2 in enumerate(theta2_vals):
            for k, phi in enumerate(phi_vals):
                x1 = np.cos(t1)
                x2 = np.sin(t1) * np.cos(t2)
                x3 = np.sin(t1) * np.sin(t2) * np.cos(phi)
                x4 = np.sin(t1) * np.sin(t2) * np.sin(phi)

                v_4d = geom_scale * np.array([x1, x2, x3, x4], dtype=np.float32)
                verts_4d.append(v_4d)

                v_mv = geom_scale * (x1 * e1 + x2 * e2 + x3 * e3 + x4 * e4)
                R = (1 + v_mv * e1) / abs(1 + v_mv * e1)

                idx12 = layout.bladeTupList.index((1, 2))
                idx13 = layout.bladeTupList.index((1, 3))
                idx14 = layout.bladeTupList.index((1, 4))

                b2 = R.value[idx12]
                b3 = R.value[idx13]
                b4 = R.value[idx14]

                rotor_4d = geom_scale * np.array([b2, b3, b4, 0.0], dtype=np.float32)
                rotors_4d.append(rotor_4d)

                idx_map[(i, j, k)] = len(verts_4d) - 1

    edges = []
    for i in range(n_theta):
        for j in range(n_theta):
            for k in range(n_phi):
                vi = idx_map[(i, j, k)]
                if i + 1 < n_theta:
                    edges.append((vi, idx_map[(i + 1, j, k)]))
                if j + 1 < n_theta:
                    edges.append((vi, idx_map[(i, j + 1, k)]))
                k_next = (k + 1) % n_phi
                edges.append((vi, idx_map[(i, j, k_next)]))

    verts_4d = np.array(verts_4d, dtype=np.float32)
    rotors_4d = np.array(rotors_4d, dtype=np.float32)
    return verts_4d, rotors_4d, np.array(edges, dtype=np.int32)


# ----------------------------------------------------------------------
# Custom visual: rotating markers with GPU rotation
# ----------------------------------------------------------------------
class RotatingMarkers(Visual):
    VERTEX_SHADER = """
        attribute vec4 a_position;
        attribute vec4 a_fg_color;
        attribute float a_size;
        uniform mat4 u_rotation;
        varying vec4 v_fg_color;
        void main() {
            vec4 rotated = u_rotation * a_position;
            gl_Position = $transform(rotated.xyz);
            gl_PointSize = a_size;
            v_fg_color = a_fg_color;
        }
    """
    FRAGMENT_SHADER = """
        varying vec4 v_fg_color;
        void main() {
            gl_FragColor = v_fg_color;
        }
    """

    def __init__(self, pos_4d, face_color=(0,1,1,0.8), size=5):
        self._pos_4d = pos_4d
        self._size = size
        self._face_color = face_color
        self._rotation = np.eye(4, dtype=np.float32)

        self._program = Program(self.VERTEX_SHADER, self.FRAGMENT_SHADER)
        self._pos_vbo = VertexBuffer(pos_4d.astype(np.float32))
        self._program['a_position'] = self._pos_vbo
        colors = np.tile(face_color[:3] + (face_color[3],), (len(pos_4d), 1))
        self._color_vbo = VertexBuffer(colors.astype(np.float32))
        self._program['a_fg_color'] = self._color_vbo
        sizes = np.full(len(pos_4d), size, dtype=np.float32)
        self._size_vbo = VertexBuffer(sizes)
        self._program['a_size'] = self._size_vbo
        self._program['u_rotation'] = self._rotation

        Visual.__init__(self)

    def _prepare_transforms(self, view):
        self._transform_system = TransformSystem(self)

    def _prepare_draw(self, view):
        tr = view.transforms
        transform = tr.get_transform('visual', 'render')
        self._program['transform'] = transform.matrix.astype(np.float32)

    def set_rotation(self, matrix):
        self._rotation = matrix.astype(np.float32)
        self._program['u_rotation'] = self._rotation
        self.update()

    def set_data(self, pos_4d=None, face_color=None, size=None):
        if pos_4d is not None:
            self._pos_vbo.set_data(pos_4d.astype(np.float32))
            self._pos_4d = pos_4d
        if face_color is not None:
            colors = np.tile(face_color[:3] + (face_color[3],), (len(self._pos_4d), 1))
            self._color_vbo.set_data(colors.astype(np.float32))
            self._face_color = face_color
        if size is not None:
            sizes = np.full(len(self._pos_4d), size, dtype=np.float32)
            self._size_vbo.set_data(sizes)
            self._size = size
        self.update()

    def _compute_bounds(self, axis, view):
        max_val = np.max(np.abs(self._pos_4d[:, :3]))
        if max_val == 0:
            return (0, 1)
        return (-max_val, max_val)


# ----------------------------------------------------------------------
# Helper: update/project axes (in‑place)
# ----------------------------------------------------------------------
class AxesGroup:
    def __init__(self, parent_view, scale, rotation_matrix):
        self.view = parent_view
        self.scale = scale
        self.group = scene.Node(parent=self.view.scene)
        self.lines = {}
        self.texts = {}
        self.colors = {
            'x': (1, 0, 0, 0.8),
            'y': (0, 1, 0, 0.8),
            'z': (0, 0, 1, 0.8),
            'w': (1, 0, 1, 0.8)
        }
        self.basis = np.eye(4, dtype=np.float32)
        self.update(rotation_matrix)

    def update(self, rotation_matrix):
        axis_len = 1.2 * self.scale
        threshold = 0.05 * self.scale
        for label, col in self.colors.items():
            idx = list(self.colors.keys()).index(label)
            vec_4d = rotation_matrix @ self.basis[idx]
            vec_3d = vec_4d[:3] * axis_len
            length = np.linalg.norm(vec_3d)
            if length < threshold:
                if label in self.lines:
                    self.lines[label].parent = None
                    self.texts[label].parent = None
                    del self.lines[label]
                    del self.texts[label]
                continue

            if label not in self.lines:
                self.lines[label] = Line(pos=np.array([[0,0,0], vec_3d]), color=col, parent=self.group)
                self.texts[label] = Text(label, pos=vec_3d * 1.05, color=col,
                                         font_size=14, anchor_x='center', anchor_y='center',
                                         parent=self.group)
            else:
                self.lines[label].set_data(pos=np.array([[0,0,0], vec_3d]))
                self.texts[label].pos = vec_3d * 1.05

    def clear(self):
        self.group.parent = None


# ----------------------------------------------------------------------
# Main Window
# ----------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, scale_values, n_theta=8, n_phi=12):
        super().__init__()
        self.scale_values = scale_values
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.current_scale_index = 0

        self.angles = {
            'xy': 0.0, 'xz': 0.0, 'xw': 0.0,
            'yz': 0.0, 'yw': 0.0, 'zw': 0.0
        }

        # ---- Central widget ----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # ---- Vispy canvas ----
        self.canvas = SceneCanvas(title='4D Hypersphere & Rotor (with projection)', size=(1200, 600))
        self.canvas.native.setParent(central)
        layout.addWidget(self.canvas.native)

        # ---- Two ViewBoxes ----
        self.grid = self.canvas.central_widget.add_grid()
        self.view_hypersphere = self.grid.add_view(0, 0)
        self.view_rotor = self.grid.add_view(0, 1)
        self.view_hypersphere.camera = 'turntable'
        self.view_rotor.camera = 'turntable'

        # ---- Title labels ----
        self.label_hypersphere = Label('Hypersphere (4D projection)', color='white', font_size=14)
        self.label_hypersphere.bgcolor = (0, 0, 0, 0.5)
        self.view_hypersphere.add_widget(self.label_hypersphere)
        self.label_hypersphere.pos = (10, 10)

        self.label_rotor = Label('Rotor Manifold (4D projection)', color='white', font_size=14)
        self.label_rotor.bgcolor = (0, 0, 0, 0.5)
        self.view_rotor.add_widget(self.label_rotor)
        self.label_rotor.pos = (10, 10)

        # ---- GPU points ----
        self.hypersphere_points = None
        self.rotor_points = None

        # ---- Lines (CPU) ----
        self.hypersphere_edges = Line(color=(0.8, 0.9, 1, 0.3), connect='segments',
                                      parent=self.view_hypersphere.scene)
        self.rotor_edges = Line(color=(1, 0.65, 0, 0.3), connect='segments',
                                parent=self.view_rotor.scene)

        # ---- Axes ----
        self.axes_hypersphere = None
        self.axes_rotor = None

        # ---- Pre‑allocated segment buffers ----
        self.seg_buffer_hypersphere = None
        self.seg_buffer_rotor = None
        self.edge_indices = None
        self.n_edges = 0

        # ---- Qt Controls ----
        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls_widget)
        layout.addWidget(controls_widget)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(scale_values) - 1)
        self.slider.setValue(0)
        self.slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.slider.setTickInterval(1)
        self.slider.valueChanged.connect(self.on_slider_changed)
        controls_layout.addWidget(self.slider)

        self.play_button = QtWidgets.QPushButton('Play')
        self.play_button.clicked.connect(self.toggle_play)
        controls_layout.addWidget(self.play_button)

        reset_button = QtWidgets.QPushButton('Reset Rotation')
        reset_button.clicked.connect(self.reset_rotation)
        controls_layout.addWidget(reset_button)

        self.status_bar = self.statusBar()
        self.status_bar.showMessage('Precomputing geometry...')

        # ---- Precompute ----
        self.all_data = self.precompute_all(scale_values, n_theta, n_phi)

        # Store edge indices and allocate segment buffers
        self.edge_indices = self.all_data[0][2]
        self.n_edges = len(self.edge_indices)
        if self.n_edges > 0:
            self.seg_buffer_hypersphere = np.zeros((self.n_edges * 2, 3), dtype=np.float32)
            self.seg_buffer_rotor = np.zeros((self.n_edges * 2, 3), dtype=np.float32)

        # Create GPU point visuals with initial data (first scale)
        verts4d_0, rotors4d_0, _ = self.all_data[0]
        self.hypersphere_points = RotatingMarkers(verts4d_0, face_color=(0,1,1,0.8), size=5*scales[0]+2)
        self.hypersphere_points.parent = self.view_hypersphere.scene
        self.rotor_points = RotatingMarkers(rotors4d_0, face_color=(1,0.84,0,0.8), size=5*scales[0]+2)
        self.rotor_points.parent = self.view_rotor.scene

        # ---- Initial render ----
        self.update_visuals(0)

        # Timer for animation
        self.timer = QTimer()
        self.timer.timeout.connect(self.advance_scale)
        self.timer.setInterval(50)

        # Debounce timer
        self.pending_update = False
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._perform_update)

        self.setFocusPolicy(Qt.StrongFocus)
        self.status_bar.showMessage('Ready – Use Q/A, W/S, E/D, R/F, T/G, Y/H to rotate 4D planes')

    def precompute_all(self, scale_values, n_theta, n_phi):
        n = len(scale_values)
        progress = QtWidgets.QProgressDialog('Computing geometry...', 'Cancel', 0, n, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        all_data = []
        for idx, scale in enumerate(scale_values):
            if progress.wasCanceled():
                break
            verts4d, rotors4d, edges = build_data(scale, n_theta, n_phi)
            all_data.append((verts4d, rotors4d, edges))
            progress.setValue(idx + 1)
            QtWidgets.QApplication.processEvents()
        progress.setValue(n)
        return all_data

    def get_rotation_matrix(self):
        R = np.eye(4, dtype=np.float32)
        planes = ['xy', 'xz', 'xw', 'yz', 'yw', 'zw']
        for plane in planes:
            angle = self.angles[plane]
            if abs(angle) > 1e-12:
                i, j = {'xy':(0,1), 'xz':(0,2), 'xw':(0,3),
                        'yz':(1,2), 'yw':(1,3), 'zw':(2,3)}[plane]
                R_plane = rotation_matrix_4d(angle, (i, j))
                R = R_plane @ R
        return R

    def update_visuals(self, index, rotation_matrix=None):
        if rotation_matrix is None:
            rotation_matrix = self.get_rotation_matrix()

        scale = self.scale_values[index]
        verts4d, rotors4d, edges = self.all_data[index]

        # ---- Update GPU points: recreate if scale changed ----
        if self.current_scale_index != index:
            # Remove old visuals
            if self.hypersphere_points is not None:
                self.hypersphere_points.parent = None
            if self.rotor_points is not None:
                self.rotor_points.parent = None

            # Create new visuals with new data
            self.hypersphere_points = RotatingMarkers(verts4d, face_color=(0,1,1,0.8), size=5*scale+2)
            self.hypersphere_points.parent = self.view_hypersphere.scene
            self.rotor_points = RotatingMarkers(rotors4d, face_color=(1,0.84,0,0.8), size=5*scale+2)
            self.rotor_points.parent = self.view_rotor.scene
            self.current_scale_index = index

        # Always apply the current rotation matrix
        self.hypersphere_points.set_rotation(rotation_matrix)
        self.rotor_points.set_rotation(rotation_matrix)

        # ---- Update edges (CPU) using pre‑allocated buffers ----
        verts_3d = (rotation_matrix @ verts4d.T).T[:, :3]
        rotors_3d = (rotation_matrix @ rotors4d.T).T[:, :3]

        if self.n_edges > 0:
            segs = self.seg_buffer_hypersphere
            segs[0::2] = verts_3d[self.edge_indices[:, 0]]
            segs[1::2] = verts_3d[self.edge_indices[:, 1]]
            self.hypersphere_edges.set_data(pos=segs)

            segs_rotor = self.seg_buffer_rotor
            segs_rotor[0::2] = rotors_3d[self.edge_indices[:, 0]]
            segs_rotor[1::2] = rotors_3d[self.edge_indices[:, 1]]
            self.rotor_edges.set_data(pos=segs_rotor)
        else:
            self.hypersphere_edges.set_data(pos=np.empty((0, 3)))
            self.rotor_edges.set_data(pos=np.empty((0, 3)))

        # ---- Axes ----
        if self.axes_hypersphere is None:
            self.axes_hypersphere = AxesGroup(self.view_hypersphere, scale, rotation_matrix)
        else:
            self.axes_hypersphere.update(rotation_matrix)
            self.axes_hypersphere.scale = scale

        if self.axes_rotor is None:
            self.axes_rotor = AxesGroup(self.view_rotor, scale, rotation_matrix)
        else:
            self.axes_rotor.update(rotation_matrix)
            self.axes_rotor.scale = scale

        # Fixed camera extent
        extent = 1.5 * scale
        self.view_hypersphere.camera.set_range(
            x=(-extent, extent), y=(-extent, extent), z=(-extent, extent)
        )
        self.view_rotor.camera.set_range(
            x=(-extent, extent), y=(-extent, extent), z=(-extent, extent)
        )

        self.setWindowTitle(f'Scale={scale:.2f}  |  Angles: xy={self.angles["xy"]:.2f}, xz={self.angles["xz"]:.2f}, xw={self.angles["xw"]:.2f}, yz={self.angles["yz"]:.2f}, yw={self.angles["yw"]:.2f}, zw={self.angles["zw"]:.2f}')
        self.canvas.update()

    # ---- Debounced update ----
    def request_update(self, index):
        self._pending_index = index
        if not self.pending_update:
            self.pending_update = True
            self.debounce_timer.start(0)

    def _perform_update(self):
        self.pending_update = False
        self.update_visuals(self._pending_index)

    # ---- Event handlers ----
    def on_slider_changed(self, value):
        self.request_update(value)
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play')

    def toggle_play(self):
        if self.timer.isActive():
            self.timer.stop()
            self.play_button.setText('Play')
        else:
            if self.slider.value() == len(self.scale_values) - 1:
                self.slider.setValue(0)
            self.timer.start()
            self.play_button.setText('Stop')

    def advance_scale(self):
        current = self.slider.value()
        next_idx = (current + 1) % len(self.scale_values)
        self.slider.setValue(next_idx)

    def reset_rotation(self):
        for key in self.angles:
            self.angles[key] = 0.0
        self.request_update(self.slider.value())

    def keyPressEvent(self, event):
        key = event.key()
        delta = 0.1
        mapping = {
            (Qt.Key_Q, Qt.Key_A): ('xy', delta),
            (Qt.Key_W, Qt.Key_S): ('xz', delta),
            (Qt.Key_E, Qt.Key_D): ('xw', delta),
            (Qt.Key_R, Qt.Key_F): ('yz', delta),
            (Qt.Key_T, Qt.Key_G): ('yw', delta),
            (Qt.Key_Y, Qt.Key_H): ('zw', delta),
        }
        for (inc_key, dec_key), (plane, step) in mapping.items():
            if key == inc_key:
                self.angles[plane] += step
                self.request_update(self.slider.value())
                return
            elif key == dec_key:
                self.angles[plane] -= step
                self.request_update(self.slider.value())
                return
        super().keyPressEvent(event)


# ----------------------------------------------------------------------
# Run
# ----------------------------------------------------------------------
if __name__ == "__main__":
    scales = np.linspace(0.1, 1.5, 20).tolist()
    qt_app = QtWidgets.QApplication([])
    app.use_app('pyqt5')
    window = MainWindow(scales, n_theta=8, n_phi=12)
    window.show()
    qt_app.exec_()