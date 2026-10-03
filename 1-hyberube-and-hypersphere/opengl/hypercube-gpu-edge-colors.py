import numpy as np

from vispy import app, scene
from vispy.scene import SceneCanvas
from vispy.scene.visuals import Line, Mesh, Text
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
# Data builder for a 4D hypercube (tesseract)
# ----------------------------------------------------------------------
def build_data(geom_scale):
    half = geom_scale / 2.0
    verts_4d = np.array([
        [x, y, z, w]
        for x in (-half, half)
        for y in (-half, half)
        for z in (-half, half)
        for w in (-half, half)
    ], dtype=np.float32)

    edges = []
    n = len(verts_4d)
    for i in range(n):
        for j in range(i + 1, n):
            if np.sum(np.abs(verts_4d[i] - verts_4d[j]) > 1e-12) == 1:
                edges.append((i, j))
    edges = np.array(edges, dtype=np.int32)
    return verts_4d, edges


def _palette_between(start, end, count, alpha=0.92):
    start = np.array(start, dtype=np.float32)
    end = np.array(end, dtype=np.float32)
    if count == 1:
        weights = np.array([0.0], dtype=np.float32)
    else:
        weights = np.linspace(0.0, 1.0, count, dtype=np.float32)

    colors = []
    for weight in weights:
        rgb = (1.0 - weight) * start + weight * end
        colors.append((*rgb.tolist(), alpha))
    return colors


def build_edge_segment_colors(verts_4d, edges):
    shell_minus_w = _palette_between((0.15, 0.95, 1.00), (0.10, 0.46, 1.00), 12)
    shell_plus_w = _palette_between((1.00, 0.82, 0.18), (1.00, 0.34, 0.10), 12)
    w_connectors = _palette_between((0.95, 0.38, 1.00), (0.58, 0.36, 1.00), 8, alpha=0.86)
    palettes = {
        'minus_w': shell_minus_w,
        'plus_w': shell_plus_w,
        'connector': w_connectors,
    }
    counts = {name: 0 for name in palettes}

    colors = []
    for start_idx, end_idx in edges:
        start = verts_4d[start_idx]
        end = verts_4d[end_idx]
        edge_axis = int(np.argmax(np.abs(start - end)))
        if edge_axis == 3:
            family = 'connector'
        elif start[3] < 0 and end[3] < 0:
            family = 'minus_w'
        else:
            family = 'plus_w'

        palette = palettes[family]
        color = palette[counts[family] % len(palette)]
        counts[family] += 1
        colors.extend((color, color))

    return np.array(colors, dtype=np.float32)


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
        self.heads = {}
        self.texts = {}
        self.colors = {
            'x': (1, 0, 0, 0.8),
            'y': (0, 1, 0, 0.8),
            'z': (0, 0, 1, 0.8),
            'w': (1, 0, 1, 0.8)
        }
        self.axis_indices = {label: idx for idx, label in enumerate(self.colors)}
        self.basis = np.eye(4, dtype=np.float32)
        for label, col in self.colors.items():
            self.lines[label] = Line(pos=np.zeros((2, 3), dtype=np.float32), color=col,
                                     width=2, parent=self.group)
            self.heads[label] = Mesh(vertices=np.zeros((3, 3), dtype=np.float32),
                                     faces=np.array([[0, 1, 2]], dtype=np.uint32),
                                     color=col, parent=self.group)
            self.texts[label] = Text(label, pos=(0, 0, 0), color=col,
                                     font_size=14, anchor_x='center', anchor_y='center',
                                     parent=self.group)
        self.update(rotation_matrix)

    def _cone_mesh(self, tip, direction, length, radius, segments=18):
        direction = direction.astype(np.float32)
        norm = np.linalg.norm(direction)
        if norm == 0:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint32)

        axis = direction / norm
        reference = np.array([0, 0, 1], dtype=np.float32)
        if abs(float(np.dot(axis, reference))) > 0.92:
            reference = np.array([0, 1, 0], dtype=np.float32)

        side_a = np.cross(axis, reference)
        side_a /= np.linalg.norm(side_a)
        side_b = np.cross(axis, side_a)
        base_center = tip - axis * length

        angles = np.linspace(0, 2 * np.pi, segments, endpoint=False)
        ring = np.array([
            base_center + radius * (np.cos(angle) * side_a + np.sin(angle) * side_b)
            for angle in angles
        ], dtype=np.float32)
        vertices = np.vstack([tip.astype(np.float32), ring, base_center.astype(np.float32)])
        base_center_index = segments + 1

        faces = []
        for idx in range(segments):
            nxt = 1 + ((idx + 1) % segments)
            cur = 1 + idx
            faces.append((0, cur, nxt))
            faces.append((base_center_index, nxt, cur))
        return vertices, np.array(faces, dtype=np.uint32)

    def update(self, rotation_matrix):
        axis_len = 1.2 * self.scale
        threshold = 0.05 * self.scale
        for label, col in self.colors.items():
            idx = self.axis_indices[label]
            vec_4d = rotation_matrix @ self.basis[idx]
            vec_3d = vec_4d[:3] * axis_len
            length = np.linalg.norm(vec_3d)
            visible = length >= threshold
            self.lines[label].visible = visible
            self.heads[label].visible = visible
            self.texts[label].visible = visible
            if visible:
                direction = vec_3d / length
                head_len = min(0.2 * self.scale, 0.35 * length)
                head_radius = min(0.075 * self.scale, 0.16 * length)
                shaft_tip = vec_3d - direction * head_len * 0.75
                vertices, faces = self._cone_mesh(vec_3d, direction, head_len, head_radius)
                self.lines[label].set_data(pos=np.array([[0, 0, 0], shaft_tip], dtype=np.float32))
                self.heads[label].set_data(vertices=vertices, faces=faces, color=col)
                self.texts[label].pos = vec_3d * 1.12

    def clear(self):
        self.group.parent = None


# ----------------------------------------------------------------------
# Main Window
# ----------------------------------------------------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, scale_values):
        super().__init__()
        self.scale_values = scale_values
        self.default_scale_index = len(scale_values) // 2
        self.current_scale_index = self.default_scale_index
        self.camera_extent = 1.5 * max(abs(scale) for scale in scale_values)
        self.camera_range_initialized = False

        # ---- Set initial rotation to show the w axis ----
        self.angles = {
            'xy': 0.0, 'xz': 0.0, 'xw': 0.5,  # <-- default xw rotation
            'yz': 0.0, 'yw': 0.0, 'zw': 0.0
        }

        # ---- Central widget ----
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QVBoxLayout(central)

        # ---- Vispy canvas ----
        self.canvas = SceneCanvas(title='4D Hypercube (with projection)', size=(1200, 600))
        self.canvas.native.setParent(central)
        layout.addWidget(self.canvas.native)

        # ---- Two ViewBoxes ----
        self.grid = self.canvas.central_widget.add_grid()
        self.view_hypercube = self.grid.add_view(0, 0)
        self.view_rotor = self.grid.add_view(0, 1)
        self.view_hypercube.camera = 'turntable'
        self.view_rotor.camera = 'turntable'

        # ---- Title labels ----
        self.label_hypercube = self._create_canvas_title('Hypercube (4D projection)')
        self.label_rotor = self._create_canvas_title('Rotor Manifold (4D projection)')

        # ---- GPU points ----
        self.hypercube_points = None
        self.rotor_points = None

        # ---- Lines (CPU) ----
        self.hypercube_edges = Line(color=(0.8, 0.9, 1, 0.3), connect='segments',
                                    parent=self.view_hypercube.scene)
        self.rotor_edges = Line(color=(1, 0.65, 0, 0.3), connect='segments',
                                parent=self.view_rotor.scene)

        # ---- Axes ----
        self.axes_hypercube = None
        self.axes_rotor = None
        self.show_axes = True

        # ---- Pre‑allocated segment buffers ----
        self.seg_buffer_hypercube = None
        self.seg_buffer_rotor = None
        self.edge_segment_colors = None
        self.edge_indices = None
        self.n_edges = 0

        # ---- Qt Controls ----
        instructions = QtWidgets.QLabel(
            'Controls: use the slider to choose the 4D geometry size; Play animates through the size sequence. '
            'Mouse wheel zooms the graph under the cursor, and mouse drag orbits the 3D camera view. '
            'Keyboard rotates the object through 4D planes: Q/A = x-y, W/S = x-z, E/D = x-w, '
            'R/F = y-z, T/G = y-w, Y/H = z-w. Press O to show or hide the axis arrows. '
            'Axis colors are red x, green y, blue z, and magenta w.'
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet('color: #d8d8d8; background: #111; padding: 6px; font-size: 10pt;')
        layout.addWidget(instructions)

        controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(controls_widget)
        layout.addWidget(controls_widget)

        self.slider = QtWidgets.QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(len(scale_values) - 1)
        self.slider.setValue(self.default_scale_index)
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
        self.all_data = self.precompute_all(scale_values)

        # Store edge indices and allocate segment buffers
        self.edge_indices = self.all_data[0][1]
        self.n_edges = len(self.edge_indices)
        if self.n_edges > 0:
            self.seg_buffer_hypercube = np.zeros((self.n_edges * 2, 3), dtype=np.float32)
            self.seg_buffer_rotor = np.zeros((self.n_edges * 2, 3), dtype=np.float32)
            self.edge_segment_colors = build_edge_segment_colors(
                self.all_data[self.default_scale_index][0], self.edge_indices
            )

        # Create GPU point visuals with initial data (first scale)
        initial_scale = self.scale_values[self.default_scale_index]
        verts4d_0, edges_0 = self.all_data[self.default_scale_index]
        self.hypercube_points = RotatingMarkers(verts4d_0, face_color=(0,1,1,0.8), size=5*initial_scale+2)
        self.hypercube_points.parent = self.view_hypercube.scene
        self.rotor_points = RotatingMarkers(verts4d_0, face_color=(1,0.84,0,0.8), size=5*initial_scale+2)
        self.rotor_points.parent = self.view_rotor.scene

        # ---- Initial render with the non‑trivial rotation ----
        self.update_visuals(self.default_scale_index)

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
        self.status_bar.showMessage('Ready - slider/play change size; mouse wheel zooms; keyboard rotates; O toggles axis arrows.')

    def _create_canvas_title(self, text):
        label = QtWidgets.QLabel(text, self.canvas.native)
        label.setAlignment(Qt.AlignCenter)
        label.setAttribute(Qt.WA_TransparentForMouseEvents)
        label.setStyleSheet('color: white; background: transparent; font-size: 14pt;')
        label.adjustSize()
        label.show()
        return label

    def _position_title_labels(self, scale):
        canvas_width = self.canvas.native.width()
        canvas_height = self.canvas.native.height()
        if canvas_width <= 0 or canvas_height <= 0:
            return

        panel_width = canvas_width / 2
        scale_fraction = scale / max(abs(value) for value in self.scale_values)
        title_offset = max(46, min(145, canvas_height * 0.23 * scale_fraction))
        title_y = int(canvas_height * 0.5 - title_offset - self.label_hypercube.height() / 2)

        for col, label in enumerate((self.label_hypercube, self.label_rotor)):
            label.adjustSize()
            title_x = int(panel_width * col + (panel_width - label.width()) / 2)
            label.move(title_x, title_y)
            label.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if hasattr(self, 'slider'):
            self._position_title_labels(self.scale_values[self.slider.value()])

    def _set_axes_visible(self):
        for axes in (self.axes_hypercube, self.axes_rotor):
            if axes is not None:
                axes.group.visible = self.show_axes

    def precompute_all(self, scale_values):
        n = len(scale_values)
        progress = QtWidgets.QProgressDialog('Computing geometry...', 'Cancel', 0, n, self)
        progress.setWindowModality(Qt.WindowModal)
        progress.setMinimumDuration(0)

        all_data = []
        for idx, scale in enumerate(scale_values):
            if progress.wasCanceled():
                break
            verts4d, edges = build_data(scale)
            all_data.append((verts4d, edges))
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
        verts4d, edges = self.all_data[index]

        # ---- Update GPU points: recreate if scale changed ----
        if self.current_scale_index != index:
            if self.hypercube_points is not None:
                self.hypercube_points.parent = None
            if self.rotor_points is not None:
                self.rotor_points.parent = None

            self.hypercube_points = RotatingMarkers(verts4d, face_color=(0,1,1,0.8), size=5*scale+2)
            self.hypercube_points.parent = self.view_hypercube.scene
            self.rotor_points = RotatingMarkers(verts4d, face_color=(1,0.84,0,0.8), size=5*scale+2)
            self.rotor_points.parent = self.view_rotor.scene
            self.current_scale_index = index

        # Always apply the current rotation matrix
        self.hypercube_points.set_rotation(rotation_matrix)
        self.rotor_points.set_rotation(rotation_matrix)

        # ---- Update edges (CPU) ----
        verts_3d = (rotation_matrix @ verts4d.T).T[:, :3]

        if self.n_edges > 0:
            segs = self.seg_buffer_hypercube
            segs[0::2] = verts_3d[self.edge_indices[:, 0]]
            segs[1::2] = verts_3d[self.edge_indices[:, 1]]
            self.hypercube_edges.set_data(pos=segs, color=self.edge_segment_colors)

            segs_rotor = self.seg_buffer_rotor
            segs_rotor[0::2] = verts_3d[self.edge_indices[:, 0]]
            segs_rotor[1::2] = verts_3d[self.edge_indices[:, 1]]
            self.rotor_edges.set_data(pos=segs_rotor, color=self.edge_segment_colors)
        else:
            self.hypercube_edges.set_data(pos=np.empty((0, 3)))
            self.rotor_edges.set_data(pos=np.empty((0, 3)))

        # ---- Axes ----
        if self.axes_hypercube is None:
            self.axes_hypercube = AxesGroup(self.view_hypercube, scale, rotation_matrix)
        else:
            self.axes_hypercube.scale = scale
            self.axes_hypercube.update(rotation_matrix)

        if self.axes_rotor is None:
            self.axes_rotor = AxesGroup(self.view_rotor, scale, rotation_matrix)
        else:
            self.axes_rotor.scale = scale
            self.axes_rotor.update(rotation_matrix)
        self._set_axes_visible()

        # Initialize the view once; later updates must preserve mouse-wheel zoom.
        if not self.camera_range_initialized:
            extent = self.camera_extent
            self.view_hypercube.camera.set_range(
                x=(-extent, extent), y=(-extent, extent), z=(-extent, extent)
            )
            self.view_rotor.camera.set_range(
                x=(-extent, extent), y=(-extent, extent), z=(-extent, extent)
            )
            self.camera_range_initialized = True
        self._position_title_labels(scale)

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
        self.slider.blockSignals(True)
        self.slider.setValue(next_idx)
        self.slider.blockSignals(False)
        self.request_update(next_idx)

    def reset_rotation(self):
        for key in self.angles:
            self.angles[key] = 0.0
        # Re‑apply a non‑trivial default so the user sees the w axis again
        self.angles['xw'] = 0.5
        self.request_update(self.slider.value())

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key_O:
            self.show_axes = not self.show_axes
            self._set_axes_visible()
            self.canvas.update()
            return

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
    scales = np.linspace(0.2, 2.0, 20).tolist()
    qt_app = QtWidgets.QApplication([])
    app.use_app('pyqt5')
    window = MainWindow(scales)
    window.show()
    qt_app.exec_()
