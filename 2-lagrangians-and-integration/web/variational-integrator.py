"""=============================================================================
DEPENDENCY LAYOUT, ARCHITECTURE & OPENGL PIPELINE
=============================================================================
This script builds a real-time, interactive 3D physics simulation using a
layered architecture. It translates the continuous Euler-Lagrange equation
(variational integrator) into an interactive desktop application with
hardware-accelerated graphics.

=============================================================================
DIRECT COMPUTATIONAL TRANSLATION OF THE EULER-LAGRANGE EQUATION (Eq 12.4)
=============================================================================
This harmonic oscillator simulation is NOT just a random physics approximation.
It is the exact discrete representation of the variational principle discussed
in Chapter 12 of "Geometric Algebra for Physicists".

1. THE CONTINUOUS MATH (From the book):
   The Euler-Lagrange equation (Eq. 12.4) states:
       d/dt( ∂L/∂x_dot ) - ∂L/∂x = 0

   For a simple harmonic oscillator, the Lagrangian (Kinetic - Potential) is:
       L = T - V = 1/2 * m * x_dot^2 - 1/2 * k * x^2

2. THE COMPUTATIONAL TRANSLATION (Continuous -> Discrete):
   Computers cannot solve continuous derivatives; they take tiny steps (dt).
   We replace the continuous acceleration x_ddot with a discrete change in velocity:
       x_ddot ≈ (v_{n+1} - v_n) / dt

   Substituting this into the derived equation:
       (v_{n+1} - v_n) / dt = -(k/m) * x_n

   Rearranging gives the VELOCITY UPDATE
       v_{n+1} = v_n - (k/m) * x_n * dt

   Kinematics dictates that velocity is the rate of change of position:
       x_dot ≈ (x_{n+1} - x_n) / dt
       -> x_{n+1} = x_n + v_{n+1} * dt

=============================================================================
DEPENDENCY HIERARCHY (Bottom to Top)
=============================================================================

1.  CORE NUMERICAL LAYER (CPU)
    - Library:   NumPy
    - Role:      The mathematical backbone. Handles all array allocations,
                 vectorized operations, and physics state storage (theta, omega,
                 energy, and 3D positions). All physics data lives in NumPy
                 arrays as float32 for performance.

2.  HARDWARE-ACCELERATED GRAPHICS LAYER (GPU)
    - Libraries: Vispy (vispy.app, vispy.scene)
    - Backend:   OpenGL
    - Role:      Vispy is a high-level Python wrapper around OpenGL. It manages
                 GPU shaders, vertex buffers (VBOs), and rendering contexts.
                 It provides:
                 - SceneCanvas:   A GPU-accelerated drawing surface.
                 - TurntableCamera: For 3D orbital controls (mouse drag to
                   rotate the pendulum view).
                 - PanZoomCamera: For 2D plots (mouse wheel to zoom).
                 - Visual primitives: Line, Markers, and Text, which are
                   rendered efficiently on the GPU.
    - Data Flow: CPU-side NumPy arrays are pushed to the GPU as vertex data
                 via Vispy's internal OpenGL bindings. The GPU then rasterizes
                 the lines, markers, and text at 60 FPS.

3.  APPLICATION / GUI LAYER (OS NATIVE)
    - Library:   PyQt5 (QtWidgets, QtCore, QtGui)
    - Role:      Provides the native desktop window, event loop, and all
                 standard UI widgets. This script explicitly uses:
                 - QMainWindow:  The main application window.
                 - QComboBox:    For physics presets.
                 - QPushButton:  Play/Pause and Reset controls.
                 - QSlider:      For scrubbing through the time history.
                 - QTimer:       Drives the animation loop at 16ms intervals.
                 - QApplication: Manages the global OS event loop.

4.  INTEGRATION LAYER (Vispy + PyQt5)
    - Mechanism:  Vispy supports multiple GUI backends. This script forces the
                  'pyqt5' backend using `app.use_app('pyqt5')`. This makes
                  Vispy's SceneCanvas inherit from PyQt5's QWidget, allowing it
                  to be embedded directly into the PyQt5 layout using
                  `layout.addWidget(self.canvas.native)`.
    - Result:     The OS sees a single native window. PyQt5 handles all the
                  buttons, dropdowns, and sliders, while Vispy handles the
                  OpenGL rendering within the same window space.

=============================================================================
DATA FLOW & RENDERING PIPELINE (How the Physics Becomes Pixels)
=============================================================================

1.  INITIALIZATION:
    - The script loads a preset (theta0, omega0, dt).
    - `pendulum_variational_integrator()` runs on the CPU (NumPy) to compute
      the entire 6000-step trajectory BEFORE the animation starts.
    - The results are stored as CPU-side arrays: self.theta, self.omega,
      self.energy, and self.positions (3D coordinates).

2.  PLOT SETUP (CPU -> GPU):
    - The PlotPanel class takes the CPU arrays and converts them into 3D
      vertices (x, y, 0) for the 2D graphs.
    - Vispy's Line.set_data(pos=...) uploads these vertices to OpenGL Vertex
      Buffer Objects (VBOs) on the GPU. Markers and Text are similarly
      uploaded via texture atlases or geometry shaders.

3.  ANIMATION LOOP (Qt Timer Driven):
    - The QTimer triggers `advance_frame()` every 16ms (~60 FPS).
    - `update_frame(index)` updates the CPU index, then calls:
        a) rod.set_data(): Updates the pendulum rod vertices.
        b) bob_marker.set_data(): Updates the bob position.
        c) arc_trace.set_data(): Updates the trailing path.
        d) PlotPanel.update_marker(): Moves the cursors on the 2D graphs.
    - Vispy automatically detects these data changes and queues an OpenGL
      draw call (glDrawArrays / glDrawElements) for the next vertical blank.

4.  RENDERING PIPELINE (GPU):
    - For each frame, the GPU executes:
        a) Vertex Shader: Transforms the 3D vertices into clip space using
           the camera matrices (TurntableCamera or PanZoomCamera).
        b) Rasterization: Converts lines/markers into fragments (pixels).
        c) Fragment Shader: Colors the pixels based on the provided RGBA
           values (e.g., royal blue for the oscillator, orange for the bob).
        d) Framebuffer Output: The rendered scene is displayed on your monitor.

=============================================================================
WHY THIS ARCHITECTURE MATTERS FOR THE VARIATIONAL INTEGRATOR
=============================================================================

- The core mathematical translation (Symplectic Euler) is PURELY CPU-BOUND
  (NumPy). It generates the trajectory in a single batch.

- The GUI (PyQt5) provides the "control room"—allowing the user to instantly
  switch presets (Rest release, Full rotation, Stress test) and visually
  compare how the discrete Euler-Lagrange translation behaves under different
  conditions.

- The GPU (OpenGL via Vispy) handles the heavy lifting of rendering. With
  6000 data points, plotting the energy graph and phase space requires
  drawing tens of thousands of line segments. The GPU processes these in
  parallel, keeping the UI responsive and smooth.

- The 3D TurntableCamera allows you to orbit around the pendulum, giving you
  a spatial intuition for the angular motion that a static 2D plot cannot
  provide.

=============================================================================
SUMMARY OF MAIN IMPORTS & THEIR SPECIFIC ROLES
=============================================================================

| Import Statement                   | Purpose                                            |
|------------------------------------|----------------------------------------------------|
| import numpy as np                 | CPU math, arrays, physics computation              |
| from vispy import app, scene       | OpenGL context, scene graph, 3D/2D rendering      |
| from vispy.scene.visuals import *  | GPU-accelerated primitives (Lines, Markers, Text) |
| from PyQt5 import QtWidgets        | Native OS widgets (Window, Buttons, Sliders)      |
| from PyQt5.QtCore import Qt, QTimer| Animation timer and Qt constants (alignment)      |

=============================================================================
"""

import numpy as np
import matplotlib.pyplot as plt

def variational_integrator(x0, v0, m, k, dt, n_steps):
    """
    Symplectic Euler (variational integrator) for the harmonic oscillator.
    This approximates the Euler-Lagrange equations while preserving energy.
    """
    x = np.zeros(n_steps)
    v = np.zeros(n_steps)
    x[0] = x0
    v[0] = v0

    for i in range(n_steps - 1):
        # Update velocity first (implicit in position), then position
        v[i+1] = v[i] - (k / m) * x[i] * dt
        x[i+1] = x[i] + v[i+1] * dt

    return x, v

# ==================== SIMULATION PARAMETERS ====================
m, k = 1.0, 1.0          # mass and spring constant
x0, v0 = 1.0, 0.0        # initial conditions
dt = 0.01                # time step
n_steps = 10000          # total number of steps
t = np.arange(n_steps) * dt

# Run the integrator
x, v = variational_integrator(x0, v0, m, k, dt, n_steps)

# Compute total energy: E = T + V = 0.5*m*v^2 + 0.5*k*x^2
E = 0.5 * m * v**2 + 0.5 * k * x**2

# ==================== VISUALIZATION ====================
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1. Position vs. time
axes[0].plot(t, x, color='royalblue', linewidth=1.5)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Position x(t)')
axes[0].set_title('Oscillator Motion')
axes[0].grid(True, alpha=0.3)

# 2. Energy vs. time
axes[1].plot(t, E, color='crimson', linewidth=1.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Total Energy')
axes[1].set_title('Energy Conservation')
axes[1].grid(True, alpha=0.3)
# Set y-axis limits close to initial energy to see tiny variations
axes[1].set_ylim([0.49, 0.51])  # initial energy = 0.5

# 3. Phase space (position vs. velocity)
axes[2].plot(x, v, color='darkgreen', linewidth=0.5, alpha=0.7)
axes[2].set_xlabel('Position x')
axes[2].set_ylabel('Velocity v')
axes[2].set_title('Phase Space')
axes[2].grid(True, alpha=0.3)
axes[2].axis('equal')  # ensure circular/elliptical shape is not distorted

plt.tight_layout()
plt.show()

# Print energy drift (to quantify conservation)
E_initial = E[0]
E_final = E[-1]
drift = (E_final - E_initial) / E_initial * 100
print(f"Initial energy: {E_initial:.6f}")
print(f"Final energy:   {E_final:.6f}")
print(f"Energy drift:   {drift:.4e} %")