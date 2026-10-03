"""Plot Example 5 using the revised delta-first coupling theory.

The raw gradient components are always evaluated in their original order.
The algorithm then constructs the ordinary update increments and exchanges
only those completed increments on a step whose swap indicator is one.
Neither the coordinates nor the gradient components themselves are swapped.
"""

import matplotlib.pyplot as plt
import numpy as np


SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#1a1a19"
TEXT_SECONDARY = "#5f5e56"
GRID = "#e8e7e2"
C_GX = "#2a78d6"  # Blue: raw G_x.
C_GY = "#008300"  # Green: raw G_y.

N, ETA = 40, 0.1
X0, Y0 = 1.0, 2.0


def run(period):
    """Return raw gradient trajectories and empirical coupling coefficient.

    ``period=0`` never exchanges increments.  ``period=p`` exchanges the
    already-calculated increments on steps whose one-based index is divisible
    by ``p``.
    """
    x, y = X0, Y0
    swap_indicators = np.zeros(N, dtype=np.int8)
    gradient_x = np.empty(N, dtype=np.float64)
    gradient_y = np.empty(N, dtype=np.float64)

    for iteration in range(1, N + 1):
        index = iteration - 1

        # 1. Evaluate the unchanged raw gradient at the current (x, y).
        gradient_x[index] = 2.0 * x * y
        gradient_y[index] = x**2

        # 2. Construct the ordinary gradient-descent increments first.
        delta_x = -ETA * gradient_x[index]
        delta_y = -ETA * gradient_y[index]

        # 3. Decide whether only those completed increments are exchanged.
        swap_indicators[index] = int(
            period > 0 and iteration % period == 0
        )
        if swap_indicators[index] == 1:
            applied_delta_x = delta_y
            applied_delta_y = delta_x
        else:
            applied_delta_x = delta_x
            applied_delta_y = delta_y

        # 4. Add the selected deltas without reordering x and y.
        x = x + applied_delta_x
        y = y + applied_delta_y

    coefficient = float(swap_indicators.mean())
    return gradient_x, gradient_y, coefficient


panels = [
    (0, "no increment swaps"),
    (4, "swap increments every 4th step"),
    (2, "swap increments every 2nd step"),
    (1, "swap increments every step"),
]

fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True)
fig.patch.set_facecolor(SURFACE)

for axis, (period, schedule_label) in zip(axes, panels):
    gradient_x, gradient_y, coefficient = run(period)
    iterations = np.arange(1, N + 1)

    axis.set_facecolor(SURFACE)
    axis.plot(iterations, gradient_x, color=C_GX, lw=2)
    axis.plot(iterations, gradient_y, color=C_GY, lw=2)
    axis.set_title(
        f"$C_{{40}} = {coefficient:g}$",
        fontsize=12,
        color=TEXT_PRIMARY,
        pad=10,
    )
    axis.text(
        0.5,
        1.005,
        schedule_label,
        transform=axis.transAxes,
        ha="center",
        va="bottom",
        fontsize=8.5,
        color=TEXT_SECONDARY,
    )
    axis.grid(True, color=GRID, lw=0.8)
    axis.set_axisbelow(True)

    for spine in ("top", "right"):
        axis.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        axis.spines[spine].set_color(GRID)

    axis.tick_params(colors=TEXT_SECONDARY, labelsize=9)
    axis.set_xlabel("iteration $i$", fontsize=10, color=TEXT_SECONDARY)
    axis.set_xlim(1, N)

axes[0].set_ylabel("raw gradient value", fontsize=10, color=TEXT_SECONDARY)
axes[0].set_ylim(bottom=0)

# Direct labels on the first panel and a shared legend for the full figure.
uncoupled_gradient_x, uncoupled_gradient_y, _ = run(0)
axes[0].annotate(
    "$G_x = 2xy$",
    xy=(4, uncoupled_gradient_x[3]),
    xytext=(8, 3.1),
    fontsize=10,
    color=TEXT_PRIMARY,
)
axes[0].annotate(
    "$G_y = x^2$",
    xy=(4, uncoupled_gradient_y[3]),
    xytext=(8, 0.55),
    fontsize=10,
    color=TEXT_PRIMARY,
)
fig.legend(
    handles=[
        plt.Line2D(
            [],
            [],
            color=C_GX,
            lw=2,
            label="$G_x^{(i)} = 2x_i y_i$",
        ),
        plt.Line2D(
            [],
            [],
            color=C_GY,
            lw=2,
            label="$G_y^{(i)} = x_i^2$",
        ),
    ],
    loc="upper right",
    ncol=2,
    frameon=False,
    fontsize=10,
    bbox_to_anchor=(0.99, 1.02),
    labelcolor=TEXT_PRIMARY,
)
fig.suptitle(
    "Raw gradient trajectories after frequency-coupled increments",
    x=0.01,
    ha="left",
    fontsize=13,
    color=TEXT_PRIMARY,
)

fig.tight_layout(rect=[0, 0, 1, 0.90])
fig.savefig(
    "coupling-frequency-gradients.png",
    dpi=160,
    facecolor=SURFACE,
    bbox_inches="tight",
)
