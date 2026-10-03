import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from clifford.g4 import layout

# ---- helper to build data at a given scale ----
def build_data(geom_scale, n_theta=8, n_phi=12):
    """
    Returns:
        vertices_3d: list of (x,y,z) for each hypersphere point (drop e4)
        rotor_points_3d: list of (x,y,z) for each rotor manifold point
        edges: list of (i, j) indices into vertices_3d
        rotor_edges: list of (i, j) indices into rotor_points_3d
    """
    e1, e2, e3, e4 = layout.basis_vectors.values()

    theta1_vals = np.linspace(0.01, np.pi - 0.01, n_theta)
    theta2_vals = np.linspace(0.01, np.pi - 0.01, n_theta)
    phi_vals = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)

    vertices_3d = []          # (x,y,z) from first three coordinates
    rotors_3d = []            # (x,y,z) from bivector components
    idx_map = {}

    # Generate grid points and rotors (scaled)
    for i, t1 in enumerate(theta1_vals):
        for j, t2 in enumerate(theta2_vals):
            for k, phi in enumerate(phi_vals):
                x1 = np.cos(t1)
                x2 = np.sin(t1) * np.cos(t2)
                x3 = np.sin(t1) * np.sin(t2) * np.cos(phi)
                x4 = np.sin(t1) * np.sin(t2) * np.sin(phi)

                # 3D coordinates for plotting (drop e4)
                v_3d = geom_scale * np.array([x1, x2, x3])
                vertices_3d.append(v_3d)

                # Compute rotor (multivector) and extract bivector components
                v_mv = geom_scale * (x1 * e1 + x2 * e2 + x3 * e3 + x4 * e4)
                R = (1 + v_mv * e1) / abs(1 + v_mv * e1)

                idx12 = layout.bladeTupList.index((1, 2))
                idx13 = layout.bladeTupList.index((1, 3))
                idx14 = layout.bladeTupList.index((1, 4))

                b2 = R.value[idx12]
                b3 = R.value[idx13]
                b4 = R.value[idx14]
                rotor_3d = geom_scale * np.array([b2, b3, b4])
                rotors_3d.append(rotor_3d)

                idx_map[(i, j, k)] = len(vertices_3d) - 1

    # Build edges for hypersphere
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

    # Rotor edges follow the same connectivity
    rotor_edges = edges[:]   # same indices, but applied to rotors_3d

    return vertices_3d, rotors_3d, edges, rotor_edges


# ---- interactive GUI using matplotlib ----
class InteractiveViewer:
    def __init__(self, scale_values, n_theta=8, n_phi=12):
        self.scale_values = scale_values
        self.n_theta = n_theta
        self.n_phi = n_phi
        self.current_scale_index = 0

        # Set up the figure and axes
        self.fig = plt.figure(figsize=(12, 6))
        self.ax_hypersphere = self.fig.add_subplot(121, projection='3d')
        self.ax_rotor = self.fig.add_subplot(122, projection='3d')

        # Make room for sliders and buttons
        plt.subplots_adjust(bottom=0.2)

        # Slider for scale
        ax_slider = plt.axes([0.25, 0.05, 0.5, 0.03])
        self.scale_slider = Slider(
            ax_slider, 'Scale', min(scale_values), max(scale_values),
            valinit=scale_values[0], valstep=scale_values
        )
        self.scale_slider.on_changed(self.update)

        # Play button
        ax_button = plt.axes([0.8, 0.05, 0.1, 0.04])
        self.play_button = Button(ax_button, 'Play')
        self.play_button.on_clicked(self.play)

        # Initial plot
        self.update(scale_values[0])

        plt.show()

    def update(self, scale):
        """Redraw the plots for the given scale."""
        # Get new data
        verts, rotors, edges, rotor_edges = build_data(scale, self.n_theta, self.n_phi)

        # Clear axes
        self.ax_hypersphere.cla()
        self.ax_rotor.cla()

        # ---- Hypersphere ----
        verts_arr = np.array(verts)
        self.ax_hypersphere.scatter(verts_arr[:, 0], verts_arr[:, 1], verts_arr[:, 2],
                                    c='cyan', s=10 * scale + 1, alpha=0.8)
        # Draw edges
        for i, j in edges:
            p1 = verts_arr[i]
            p2 = verts_arr[j]
            self.ax_hypersphere.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                                     color='lightblue', alpha=0.3)
        self.ax_hypersphere.set_title(f'Hypersphere (scale={scale:.2f})')
        self.ax_hypersphere.set_xlabel('x')
        self.ax_hypersphere.set_ylabel('y')
        self.ax_hypersphere.set_zlabel('z')
        self._set_equal_aspect(self.ax_hypersphere)

        # ---- Rotor manifold ----
        rotors_arr = np.array(rotors)
        self.ax_rotor.scatter(rotors_arr[:, 0], rotors_arr[:, 1], rotors_arr[:, 2],
                              c='gold', s=10 * scale + 1, alpha=0.8)
        for i, j in rotor_edges:
            p1 = rotors_arr[i]
            p2 = rotors_arr[j]
            self.ax_rotor.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                               color='orange', alpha=0.3)
        self.ax_rotor.set_title(f'Rotor manifold (scale={scale:.2f})')
        self.ax_rotor.set_xlabel('x')
        self.ax_rotor.set_ylabel('y')
        self.ax_rotor.set_zlabel('z')
        self._set_equal_aspect(self.ax_rotor)

        # Redraw
        self.fig.canvas.draw_idle()

    def play(self, event):
        """Animate through the scale list."""
        for s in self.scale_values:
            self.scale_slider.set_val(s)   # triggers update
            time.sleep(0.5)

    @staticmethod
    def _set_equal_aspect(ax):
        """Force equal aspect ratio for 3D axes."""
        x_limits = ax.get_xlim3d()
        y_limits = ax.get_ylim3d()
        z_limits = ax.get_zlim3d()
        x_range = abs(x_limits[1] - x_limits[0])
        y_range = abs(y_limits[1] - y_limits[0])
        z_range = abs(z_limits[1] - z_limits[0])
        max_range = max(x_range, y_range, z_range)
        mid_x = (x_limits[0] + x_limits[1]) * 0.5
        mid_y = (y_limits[0] + y_limits[1]) * 0.5
        mid_z = (z_limits[0] + z_limits[1]) * 0.5
        ax.set_xlim3d(mid_x - max_range*0.5, mid_x + max_range*0.5)
        ax.set_ylim3d(mid_y - max_range*0.5, mid_y + max_range*0.5)
        ax.set_zlim3d(mid_z - max_range*0.5, mid_z + max_range*0.5)


# ---- main ----
if __name__ == "__main__":
    # Choose scale values to explore
    scales = [0.1, 0.3, 0.6, 1.0, 1.5]   # you can edit this list
    # Optional: adjust resolution for speed vs. detail
    viewer = InteractiveViewer(scales, n_theta=8, n_phi=12)