"""coupled_gradients — gradient descent with swap-coupled partial gradients.

Implements the coupling-frequency construction from coupling-frequency-theory.md:
a two-variable descent on f(x, y) where, on steps flagged by a swap indicator
s_i in {0, 1}, the partial-gradient updates are exchanged (x absorbs G_y and
y absorbs G_x).  The empirical swap frequency

    nu_N = (1/N) * sum_i s_i

is the coupling coefficient C(df/dx)_y of the run.

Example
-------
>>> import coupled_gradients as cg
>>> f = lambda x, y: x**2 * y
>>> s = cg.periodic_schedule(period=2, n=5)          # [0, 1, 0, 1, 0]
>>> traj = cg.run(f, x0=1.0, y0=2.0, schedule=s, eta=0.1)
>>> round(traj.coupling, 1)
0.4
"""

from dataclasses import dataclass

import numpy as np

__all__ = [
    "coupling_coefficient",
    "periodic_schedule",
    "bernoulli_schedule",
    "numerical_gradients",
    "Trajectory",
    "run",
]


def coupling_coefficient(s):
    """Empirical swap frequency nu_N of a swap-indicator array."""
    s = np.asarray(s)
    if s.size == 0:
        raise ValueError("schedule is empty")
    return float(s.mean())


def periodic_schedule(period, n):
    """Swap every `period`-th step over `n` steps; period=0 never swaps.

    periodic_schedule(2, 6) -> [0, 1, 0, 1, 0, 1], coupling 1/2.
    """
    i = np.arange(1, n + 1)
    if period == 0:
        return np.zeros(n, dtype=int)
    return (i % period == 0).astype(int)


def bernoulli_schedule(nu, n, rng=None):
    """Random schedule whose expected coupling coefficient is `nu`."""
    if not 0.0 <= nu <= 1.0:
        raise ValueError(f"nu must be in [0, 1], got {nu}")
    rng = np.random.default_rng(rng)
    return (rng.random(n) < nu).astype(int)


def numerical_gradients(f, eps=1e-6):
    """Central-difference (df/dx, df/dy) callables for a scalar f(x, y)."""

    def gx(x, y):
        return (f(x + eps, y) - f(x - eps, y)) / (2 * eps)

    def gy(x, y):
        return (f(x, y + eps) - f(x, y - eps)) / (2 * eps)

    return gx, gy


@dataclass
class Trajectory:
    """Arrays recorded over a run of n steps.

    x, y have length n + 1 (initial point included); gx, gy, s have length n.
    """

    x: np.ndarray
    y: np.ndarray
    gx: np.ndarray
    gy: np.ndarray
    s: np.ndarray

    @property
    def coupling(self):
        """Coupling coefficient C(df/dx)_y = nu_N of this run."""
        return coupling_coefficient(self.s)

    @property
    def final(self):
        """Final point (x_{N+1}, y_{N+1})."""
        return float(self.x[-1]), float(self.y[-1])


def run(f, x0, y0, schedule, eta=0.1, gradients=None):
    """Descend f(x, y) from (x0, y0) under a swap schedule.

    On step i the partial gradients G_x = df/dx and G_y = df/dy are evaluated
    at (x_i, y_i); if schedule[i] == 0 each variable absorbs its own gradient,
    if schedule[i] == 1 the updates are exchanged.

    Parameters
    ----------
    f : callable(x, y) -> float, or None if `gradients` is given.
    schedule : array of {0, 1} swap indicators; its length sets the step count.
    gradients : optional (gx, gy) pair of callables; defaults to central
        differences of f.

    Returns
    -------
    Trajectory
    """
    s = np.asarray(schedule, dtype=int)
    if s.ndim != 1 or not np.isin(s, (0, 1)).all():
        raise ValueError("schedule must be a 1-D array of 0s and 1s")
    if gradients is None:
        if f is None:
            raise ValueError("provide f or an explicit (gx, gy) pair")
        gradients = numerical_gradients(f)
    gx_f, gy_f = gradients

    n = len(s)
    x = np.empty(n + 1)
    y = np.empty(n + 1)
    gx = np.empty(n)
    gy = np.empty(n)
    x[0], y[0] = x0, y0

    for i, swap in enumerate(s):
        gx[i], gy[i] = gx_f(x[i], y[i]), gy_f(x[i], y[i])
        if swap:
            x[i + 1] = x[i] - eta * gy[i]
            y[i + 1] = y[i] - eta * gx[i]
        else:
            x[i + 1] = x[i] - eta * gx[i]
            y[i + 1] = y[i] - eta * gy[i]

    return Trajectory(x=x, y=y, gx=gx, gy=gy, s=s)


if __name__ == "__main__":
    f = lambda x, y: x**2 * y

    # Example 3 of the theory doc: s = [0,1,0,1,0] from (1.0, 2.0), eta = 0.1
    traj = run(f, 1.0, 2.0, schedule=[0, 1, 0, 1, 0])
    print(f"Example 3: C = {traj.coupling}, final (x, y) = "
          f"({traj.final[0]:.4f}, {traj.final[1]:.4f})")

    # Example 5: periodic schedules sweeping the coupling coefficient
    for period in (0, 4, 2, 1):
        t = run(f, 1.0, 2.0, schedule=periodic_schedule(period, 40))
        print(f"Example 5: C = {t.coupling:<5} final (x, y) = "
              f"({t.final[0]:.4f}, {t.final[1]:.4f})")
