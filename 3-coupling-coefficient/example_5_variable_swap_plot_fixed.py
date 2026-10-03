"""Plot Example 5 with intentional whole-variable swaps.

On an indicated step, this experiment first performs the standard gradient
update and then deliberately exchanges the complete state with ``x, y = y, x``.
That operation is a feature of this variant.  Some schedules grow by many
orders of magnitude and cross zero, so the plots use independent symmetric-log
y-axes instead of a shared nonnegative linear axis.
"""

from pathlib import Path

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
    """Return raw gradients and C for an intentional variable-swap schedule.

    ``period=0`` never swaps variables.  ``period=p`` exchanges the complete
    post-update state on steps whose one-based index is divisible by ``p``.
    The raw gradient is evaluated before both the update and state exchange.
    """
    if isinstance(period, (bool, np.bool_)) or not isinstance(
        period, (int, np.integer)
    ):
        raise TypeError("period must be a non-negative integer")
    period = int(period)
    if period < 0:
        raise ValueError("period must be a non-negative integer")

    x, y = X0, Y0
    swap_indicators = np.zeros(N, dtype=np.int8)
    gradient_x = np.empty(N, dtype=np.float64)
    gradient_y = np.empty(N, dtype=np.float64)

    for iteration in range(1, N + 1):
        index = iteration - 1

        # 1. Evaluate the raw gradient before the state exchange.
        gradient_x[index] = 2.0 * x * y
        gradient_y[index] = x**2

        # 2. Record whether this step includes the intentional state swap.
        swap_indicators[index] = int(
            period > 0 and iteration % period == 0
        )

        # 3. Perform the standard gradient update.
        x = x - ETA * gradient_x[index]
        y = y - ETA * gradient_y[index]

        # 4. Feature: exchange the whole post-update state when indicated.
        if swap_indicators[index] == 1:
            x, y = y, x

        if not np.all(
            np.isfinite((gradient_x[index], gradient_y[index], x, y))
        ):
            raise FloatingPointError(
                f"non-finite dynamics encountered at iteration {iteration}"
            )

    coefficient = float(swap_indicators.mean())
    return gradient_x, gradient_y, coefficient


def build_figure():
    """Build the readable four-panel variable-swap comparison."""
    panels = [
        (0, "no variable swaps"),
        (4, "swap variables every 4th step"),
        (2, "swap variables every 2nd step"),
        (1, "swap variables every step"),
    ]

    # The schedules occupy radically different ranges, so sharing one linear
    # y-axis would flatten three panels.  Independent symlog axes preserve
    # negative values, behavior around zero, and growth up to about 1e41.
    figure, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=False)
    figure.patch.set_facecolor(SURFACE)
    iterations = np.arange(1, N + 1)

    for axis, (period, schedule_label) in zip(axes, panels):
        gradient_x, gradient_y, coefficient = run(period)

        axis.set_facecolor(SURFACE)
        axis.plot(iterations, gradient_x, color=C_GX, lw=2)
        axis.plot(iterations, gradient_y, color=C_GY, lw=2)
        axis.set_yscale("symlog", linthresh=1e-2, linscale=1.0)
        axis.axhline(0.0, color=GRID, lw=0.9)
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

        axis.tick_params(colors=TEXT_SECONDARY, labelsize=8)
        axis.set_xlabel("iteration $i$", fontsize=10, color=TEXT_SECONDARY)
        axis.set_xlim(1, N)

    axes[0].set_ylabel(
        "raw gradient value (symmetric log scale)",
        fontsize=10,
        color=TEXT_SECONDARY,
    )

    # Direct labels on the uncoupled panel and one legend for the figure.
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
    figure.legend(
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
    figure.suptitle(
        "Raw gradient trajectories under frequency-coupled variable swaps",
        x=0.01,
        ha="left",
        fontsize=13,
        color=TEXT_PRIMARY,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.90])
    return figure


def main():
    """Generate the variable-swap plot next to this script."""
    figure = build_figure()
    output_path = Path(__file__).with_name(
        "coupling-variable-swap-gradients.png"
    )
    figure.savefig(
        output_path,
        dpi=160,
        facecolor=SURFACE,
        bbox_inches="tight",
    )
    plt.close(figure)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
