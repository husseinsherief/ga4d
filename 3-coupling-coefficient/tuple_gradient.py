"""Numerically differentiate a real scalar-valued function.

This module implements the experiment's proposed representation of a gradient.
The partial derivatives are returned together as an ordered tuple and are never
added together.  For a two-variable function the raw result is therefore

    (df/dx, df/dy)

rather than ``df/dx + df/dy``.

It also implements the Theory's frequency-driven swapping procedure.  At
iteration ``i``, the partial derivatives themselves remain unchanged, while the
components used for the update are either

    (G_x, G_y)  when s_i = 0
    (G_y, G_x)  when s_i = 1

For a finite experiment of ``N`` iterations, an exact coefficient ``C`` is
implemented with ``N*C`` swaps.  The deterministic indicator
``s_i = floor(i*C) - floor((i-1)*C)`` distributes those swaps through the
experiment.  The coefficient is then measured from the actual indicators as
``C = nu_N = sum(s_i) / N``.

The derivative in each coordinate is approximated with a central difference:

    df/dx_i ~= (f(point + h_i * e_i) - f(point - h_i * e_i)) / (2 * h_i)

Here ``e_i`` changes only coordinate ``i``.  This is the same derivative
calculation used by the original spherical-pendulum ``numerical_gradient``,
but the tuple return type makes the no-summing rule explicit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from typing import Any

import numpy as np


def _coupling_fraction(coupling_coefficient: float) -> Fraction:
    """Validate C and retain its user-visible decimal value exactly."""
    try:
        coefficient = float(coupling_coefficient)
    except (TypeError, ValueError) as exc:
        raise TypeError("coupling_coefficient must be a real number") from exc

    if not np.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
        raise ValueError("coupling_coefficient must be between 0 and 1")

    # Fraction(str(...)) avoids boundary errors such as an intended 0.1 swap
    # failing to occur at iteration 10 because of binary floating-point noise.
    return Fraction(str(coefficient))


def _positive_integer(value: int, name: str) -> int:
    """Validate a positive integer without accepting booleans as integers."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{name} must be a positive integer")
    result = int(value)
    if result <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return result


def coupling_coefficient(
    swap_indicators: Sequence[int] | np.ndarray,
) -> float:
    """Calculate the Theory's empirical coefficient ``nu_N``.

    This implements the definition exactly:

        nu_N = (1/N) * sum(s_i),  where every s_i is either 0 or 1.
    """
    indicators = np.asarray(swap_indicators)
    if indicators.ndim != 1 or indicators.size == 0:
        raise ValueError("swap_indicators must be a non-empty sequence")
    if np.iscomplexobj(indicators):
        raise TypeError("swap indicators must be real binary values")

    try:
        numeric_indicators = indicators.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("swap indicators must be binary values") from exc

    if not np.all(
        np.logical_or(numeric_indicators == 0.0, numeric_indicators == 1.0)
    ):
        raise ValueError("every swap indicator must be either 0 or 1")

    return float(np.sum(numeric_indicators) / numeric_indicators.size)


def empirical_swap_frequency(
    swap_indicators: Sequence[int] | np.ndarray,
) -> float:
    """Descriptive alias for :func:`coupling_coefficient`."""
    return coupling_coefficient(swap_indicators)


def build_swap_schedule(
    total_iterations: int,
    coupling_coefficient_value: float,
) -> tuple[int, ...]:
    """Build an evenly distributed finite schedule with exact ``nu_N = C``.

    Because each indicator is binary, a coefficient is exactly realizable over
    ``N`` iterations only when ``N*C`` is an integer.  For example, C=0.25 is
    realizable for N=8 (two swaps), but not for N=10 (2.5 swaps).
    """
    count = _positive_integer(total_iterations, "total_iterations")
    frequency = _coupling_fraction(coupling_coefficient_value)
    requested_swaps = count * frequency

    if requested_swaps.denominator != 1:
        raise ValueError(
            "coupling_coefficient is not exactly realizable for "
            f"total_iterations={count}; choose a coefficient k/{count}"
        )

    number_of_swaps = requested_swaps.numerator
    schedule = tuple(
        int(
            (iteration * number_of_swaps) // count
            > ((iteration - 1) * number_of_swaps) // count
        )
        for iteration in range(1, count + 1)
    )

    # This assertion directly checks the boxed definition in the Theory.
    assert coupling_coefficient(schedule) == float(frequency)
    return schedule


def swap_indicator(
    iteration: int,
    coupling_coefficient: float,
    *,
    total_iterations: int | None = None,
) -> int:
    """Return the Theory's binary swap indicator ``s_i``.

    ``coupling_coefficient`` is the frequency in swaps per iteration and must
    lie in ``[0, 1]``.  The deterministic schedule is

        s_i = floor(i*C) - floor((i-1)*C).

    When ``total_iterations=N`` is supplied, ``N*C`` must be an integer and
    the resulting finite schedule satisfies ``nu_N=C`` exactly.  Without N,
    this function provides a streaming schedule whose frequency converges to C;
    after any finite N its measured value is ``floor(N*C)/N``.
    """
    step = _positive_integer(iteration, "iteration")

    if total_iterations is not None:
        schedule = build_swap_schedule(total_iterations, coupling_coefficient)
        if step > len(schedule):
            raise ValueError("iteration cannot exceed total_iterations")
        return schedule[step - 1]

    frequency = _coupling_fraction(coupling_coefficient)
    swaps_through_step = (step * frequency.numerator) // frequency.denominator
    swaps_before_step = (
        (step - 1) * frequency.numerator
    ) // frequency.denominator
    return int(swaps_through_step > swaps_before_step)


def swapping_procedure(
    partial_derivatives: Sequence[float] | np.ndarray,
    *,
    iteration: int,
    coupling_coefficient: float,
    total_iterations: int | None = None,
) -> tuple[float, float]:
    """Return the ordered update components selected by ``s_i``.

    The input must be the unchanged raw bivariate gradient ``(G_x, G_y)``.
    When ``s_i`` is zero it is returned in standard order.  When ``s_i`` is
    one, the updates are exchanged and the result is ``(G_y, G_x)``.
    """
    raw_components = np.asarray(partial_derivatives)
    if np.iscomplexobj(raw_components):
        raise TypeError("partial derivatives must be real numbers")

    components = np.asarray(partial_derivatives, dtype=np.float64)
    if components.shape != (2,):
        raise ValueError("swapping requires the two components (G_x, G_y)")
    if not np.all(np.isfinite(components)):
        raise ValueError("partial derivatives must be finite")

    gradient = (float(components[0]), float(components[1]))
    if swap_indicator(
        iteration,
        coupling_coefficient,
        total_iterations=total_iterations,
    ):
        return gradient[1], gradient[0]
    return gradient


def numerical_gradient(
    function: Callable[..., float],
    parameters: Sequence[float] | np.ndarray,
    eps: float | Sequence[float] | np.ndarray = 1e-6,
    *,
    unpack: bool = False,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    iteration: int = 1,
    coupling_coefficient: float = 0.0,
    total_iterations: int | None = None,
) -> tuple[float, ...]:
    """Return numerical partial derivatives in the scheduled update order.

    Parameters
    ----------
    function:
        A real scalar-valued function.  By default it receives one NumPy array,
        matching the calling convention in the supplied pendulum program.  Set
        ``unpack=True`` for a function declared like ``f(x, y)`` instead.
    parameters:
        The point at which to calculate the gradient, such as ``(x, y)``.
        Any positive number of variables is supported.
    eps:
        A positive central-difference step.  Supply one number for every
        coordinate or a sequence with a separate step for each coordinate.
    unpack:
        If false, call ``function(point, *args, **kwargs)``.  If true, call
        ``function(*point, *args, **kwargs)``.
    args, kwargs:
        Optional fixed arguments forwarded to ``function``.  They are not
        differentiated.
    iteration:
        The positive discrete index ``i`` used by the swap schedule.
    coupling_coefficient:
        The Theory's frequency ``C = nu_N`` in swaps per iteration, from zero
        to one.  Coupling is defined for exactly two differentiated variables.
    total_iterations:
        Optional finite experiment length ``N``.  When supplied, the schedule
        is required to realize ``nu_N=C`` exactly.  Omitting it uses the
        streaming schedule whose finite empirical frequency approaches C.

    Returns
    -------
    tuple[float, ...]
        For two variables, ``(G_x, G_y)`` if ``s_i=0`` or ``(G_y, G_x)`` if
        ``s_i=1``.  The components are never summed.  Functions with more than
        two variables remain supported when ``coupling_coefficient=0``.

    Notes
    -----
    A gradient is defined for a scalar-valued function.  A vector-valued
    function would instead require a Jacobian matrix, so non-scalar results are
    rejected rather than silently combined.
    """
    raw_parameters = np.asarray(parameters)
    if np.iscomplexobj(raw_parameters):
        raise TypeError("parameters must be real numbers")

    point = np.asarray(parameters, dtype=np.float64)
    if point.ndim != 1 or point.size == 0:
        raise ValueError("parameters must be a non-empty one-dimensional sequence")
    if not np.all(np.isfinite(point)):
        raise ValueError("parameters must contain only finite values")

    # A scalar eps is applied to every coordinate.  A sequence lets callers
    # choose a different finite-difference scale for each coordinate.
    step_input = np.asarray(eps, dtype=np.float64)
    if step_input.ndim == 0:
        steps = np.full(point.shape, float(step_input), dtype=np.float64)
    elif step_input.shape == point.shape:
        steps = step_input.copy()
    else:
        raise ValueError("eps must be a scalar or have one entry per parameter")

    if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("every eps value must be positive and finite")

    fixed_args = tuple(args)
    fixed_kwargs = {} if kwargs is None else dict(kwargs)

    def evaluate(at_point: np.ndarray) -> float:
        """Evaluate the target while enforcing its scalar-output contract."""
        if unpack:
            result = function(*at_point.tolist(), *fixed_args, **fixed_kwargs)
        else:
            result = function(at_point, *fixed_args, **fixed_kwargs)

        result_array = np.asarray(result)
        if result_array.ndim != 0:
            raise TypeError(
                "function must return one scalar; use a Jacobian for vector output"
            )
        if np.iscomplexobj(result_array):
            raise TypeError("function must return a real scalar")

        scalar = float(result_array)
        if not np.isfinite(scalar):
            raise ValueError("function returned a non-finite value near parameters")
        return scalar

    partial_derivatives: list[float] = []

    # Perturb exactly one coordinate at a time.  Each derivative is appended
    # independently; deliberately, there is no sum of these components.
    for index, step in enumerate(steps):
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step

        derivative = (evaluate(plus) - evaluate(minus)) / (2.0 * step)
        partial_derivatives.append(float(derivative))

    raw_gradient = tuple(partial_derivatives)
    frequency = _coupling_fraction(coupling_coefficient)

    if len(raw_gradient) == 2:
        return swapping_procedure(
            raw_gradient,
            iteration=iteration,
            coupling_coefficient=coupling_coefficient,
            total_iterations=total_iterations,
        )

    # The requested swap is specifically between x and y.  Preserve support
    # for arbitrary-dimensional scalar functions only when coupling is off.
    if frequency != 0:
        raise ValueError(
            "frequency-driven coupling requires exactly two parameters"
        )

    # Validate the iteration even when no bivariate swapping is requested, so
    # the public API behaves consistently for every number of parameters.
    swap_indicator(
        iteration,
        coupling_coefficient,
        total_iterations=total_iterations,
    )
    return raw_gradient


def _demonstration() -> None:
    """Run a small analytic check when this file is executed directly."""

    # Vector-style input, matching the original numerical_gradient API.
    def example(point: np.ndarray) -> float:
        x, y = point
        return x**2 + 3.0 * x * y + np.sin(y)

    point = (2.0, 0.5)
    measured = numerical_gradient(example, point)
    expected = (
        2.0 * point[0] + 3.0 * point[1],
        3.0 * point[0] + np.cos(point[1]),
    )

    assert isinstance(measured, tuple)
    assert len(measured) == 2
    assert np.allclose(measured, expected, rtol=1e-7, atol=1e-7)

    print("point:", point)
    print("numerical (df/dx, df/dy):", measured)
    print("analytic  (df/dx, df/dy):", expected)

    # At C=0.25, every fourth update is exchanged.  Raw G_x and G_y retain
    # their values; only the variables that receive them change at step 4.
    total_iterations = 8
    indicators = build_swap_schedule(total_iterations, 0.25)
    assert indicators == (0, 0, 0, 1, 0, 0, 0, 1)
    assert coupling_coefficient(indicators) == 0.25

    swapped_update = numerical_gradient(
        example,
        point,
        iteration=4,
        coupling_coefficient=0.25,
        total_iterations=total_iterations,
    )
    assert np.allclose(swapped_update, expected[::-1], rtol=1e-7, atol=1e-7)
    print("swap indicators for C=0.25:", indicators)
    print("step 4 update (G_y, G_x):", swapped_update)

    # The same differentiator also accepts a conventional f(x, y) signature.
    unpacked = numerical_gradient(
        lambda x, y: x**2 + 3.0 * x * y + np.sin(y),
        point,
        unpack=True,
    )
    assert np.allclose(unpacked, expected, rtol=1e-7, atol=1e-7)


if __name__ == "__main__":
    _demonstration()
