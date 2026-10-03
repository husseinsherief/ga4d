"""Reusable numerical gradients and post-processed component coupling.

This is the canonical library for the current experiment.  It follows the
same rule as ``coupling-gradient-gpu-four-spheres.py``:

1. Compute or receive one uncoupled result array.
2. Copy that array.
3. At scheduled sample indices, exchange two complete component values.
4. Never feed the exchanged display values back into the source calculation.

The experiment's coupling control uses these schedules:

    C=0.00 -> no swaps
    C=0.25 -> every 8th sample
    C=0.50 -> every 4th sample
    C=1.00 -> every 2nd sample

Equivalently, the target swap frequency is ``C / 2``.  A
:class:`CoupledResult` therefore reports both ``requested_coupling`` and the
finite array's measured ``actual_swap_fraction``.

NumPy is the only required dependency.  SymPy and Matplotlib are imported
only inside their adapter functions, so the core library works without them.

NumPy example
-------------
>>> base = np.column_stack((theta, phi))
>>> result = couple_results(base, coupling=0.5, has_initial_sample=True)
>>> coupled_theta = result.coupled_results[:, 0]
>>> coupled_phi = result.coupled_results[:, 1]

SymPy example
-------------
>>> x, y = sp.symbols("x y")
>>> symbolic_gradient(x**2 * y, (x, y))
(2*x*y, x**2)

Matplotlib example
------------------
>>> family = coupled_family(base, (0, 0.25, 0.5, 1))
>>> figure, axes = plot_coupled_family(family, component_labels=("theta", "phi"))
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import numpy as np


__version__ = "2.0.0"

STANDARD_COUPLINGS = (0.0, 0.25, 0.5, 1.0)

__all__ = [
    "STANDARD_COUPLINGS",
    "CoupledResult",
    "couple_gradient_history",
    "couple_results",
    "coupled_family",
    "coupling_swap_mask",
    "empirical_swap_frequency",
    "numerical_gradient",
    "numerical_gradient_array",
    "numerical_gradient_history",
    "period_for_coupling",
    "periodic_swap_mask",
    "plot_coupled_components",
    "plot_coupled_family",
    "swap_result_components",
    "symbolic_gradient",
    "sympy_gradient_history",
]


def _nonnegative_integer(value: int, name: str) -> int:
    """Validate a non-negative integer without accepting booleans."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return result


def _normalize_axis(axis: int, number_of_dimensions: int, name: str) -> int:
    """Return a non-negative array axis with a useful error message."""
    if isinstance(axis, (bool, np.bool_)) or not isinstance(
        axis,
        (int, np.integer),
    ):
        raise TypeError(f"{name} must be an integer")
    normalized = int(axis)
    if normalized < 0:
        normalized += number_of_dimensions
    if not 0 <= normalized < number_of_dimensions:
        raise ValueError(
            f"{name}={axis} is invalid for an array with "
            f"{number_of_dimensions} dimensions"
        )
    return normalized


def _coupling_fraction(coupling: float) -> Fraction:
    """Validate ``C`` and preserve its written decimal value."""
    if isinstance(coupling, (bool, np.bool_)):
        raise TypeError("coupling must be a real number between 0 and 1")
    try:
        coefficient = float(coupling)
    except (TypeError, ValueError) as exc:
        raise TypeError("coupling must be a real number between 0 and 1") from exc
    if not np.isfinite(coefficient) or not 0.0 <= coefficient <= 1.0:
        raise ValueError("coupling must be between 0 and 1")
    return Fraction(str(coefficient))


def _validate_component_indices(
    component_indices: Sequence[int],
    number_of_components: int,
) -> tuple[int, int]:
    """Validate the two component columns that will exchange values."""
    if len(component_indices) != 2:
        raise ValueError("component_indices must contain exactly two indices")

    normalized_indices = []
    for value in component_indices:
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError("component indices must be integers")
        index = int(value)
        if index < 0:
            index += number_of_components
        if not 0 <= index < number_of_components:
            raise IndexError("component index is outside the component axis")
        normalized_indices.append(index)

    first, second = normalized_indices
    if first == second:
        raise ValueError("the two component indices must be different")
    return first, second


def _validated_swap_mask(
    swap_mask: Sequence[int] | np.ndarray,
    number_of_samples: int,
    *,
    has_initial_sample: bool,
) -> np.ndarray:
    """Return a validated one-dimensional Boolean mask."""
    raw_mask = np.asarray(swap_mask)
    if raw_mask.ndim != 1 or len(raw_mask) != number_of_samples:
        raise ValueError(
            "swap_mask must be one-dimensional with one entry per sample"
        )
    if np.iscomplexobj(raw_mask):
        raise TypeError("swap_mask must contain only binary values")
    try:
        numeric_mask = raw_mask.astype(np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError("swap_mask must contain only binary values") from exc
    if not np.all(np.logical_or(numeric_mask == 0.0, numeric_mask == 1.0)):
        raise ValueError("every swap-mask entry must be 0 or 1")

    result = numeric_mask.astype(bool)
    if has_initial_sample and len(result) and result[0]:
        raise ValueError("the initial sample cannot be marked for swapping")
    return result


def periodic_swap_mask(
    number_of_samples: int,
    period: int,
    *,
    has_initial_sample: bool = False,
) -> np.ndarray:
    """Return a mask that swaps every ``period``-th one-based sample.

    ``period=0`` disables swapping.  With ``has_initial_sample=True``, array
    index zero is excluded and index one represents integration step one.
    """
    count = _nonnegative_integer(number_of_samples, "number_of_samples")
    swap_period = _nonnegative_integer(period, "period")
    mask = np.zeros(count, dtype=bool)
    if count == 0 or swap_period == 0:
        return mask

    offset = 1 if has_initial_sample else 0
    number_of_steps = count - offset
    if number_of_steps <= 0:
        return mask

    one_based_steps = np.arange(1, number_of_steps + 1, dtype=np.int64)
    mask[offset:] = (one_based_steps % swap_period) == 0
    return mask


def coupling_swap_mask(
    number_of_samples: int,
    coupling: float,
    *,
    has_initial_sample: bool = False,
) -> np.ndarray:
    """Build the current experiment's deterministic schedule for ``C``.

    The target swap frequency is ``C/2``.  A floor-difference accumulator
    supports every real ``C`` in ``[0, 1]`` while recovering the exact periodic
    schedules used by the GUI for C=0, 0.25, 0.5, and 1.
    """
    count = _nonnegative_integer(number_of_samples, "number_of_samples")
    frequency = _coupling_fraction(coupling) / 2
    mask = np.zeros(count, dtype=bool)

    offset = 1 if has_initial_sample else 0
    number_of_steps = count - offset
    if number_of_steps <= 0 or frequency == 0:
        return mask

    numerator = frequency.numerator
    denominator = frequency.denominator
    schedule = [
        (step * numerator) // denominator
        > ((step - 1) * numerator) // denominator
        for step in range(1, number_of_steps + 1)
    ]
    mask[offset:] = schedule
    return mask


def period_for_coupling(coupling: float) -> int | None:
    """Return the exact periodic interval for ``C``, or ``None`` if aperiodic."""
    coefficient = _coupling_fraction(coupling)
    if coefficient == 0:
        return 0
    period = Fraction(2, 1) / coefficient
    if period.denominator == 1:
        return int(period)
    return None


def empirical_swap_frequency(
    swap_mask: Sequence[int] | np.ndarray,
    *,
    has_initial_sample: bool = False,
) -> float:
    """Return the measured fraction of eligible samples that were swapped."""
    raw = np.asarray(swap_mask)
    mask = _validated_swap_mask(
        raw,
        len(raw) if raw.ndim == 1 else 0,
        has_initial_sample=has_initial_sample,
    )
    eligible = mask[1:] if has_initial_sample else mask
    if len(eligible) == 0:
        return 0.0
    return float(np.mean(eligible))


@dataclass(frozen=True)
class CoupledResult:
    """One uncoupled result array and its independently coupled copy."""

    base_results: np.ndarray
    coupled_results: np.ndarray
    swap_mask: np.ndarray
    sample_axis: int
    component_axis: int
    component_indices: tuple[int, int]
    has_initial_sample: bool
    requested_coupling: float | None
    period: int | None
    schedule_kind: str

    @property
    def actual_swap_fraction(self) -> float:
        """Measured swap rate, excluding the initial sample when applicable."""
        return empirical_swap_frequency(
            self.swap_mask,
            has_initial_sample=self.has_initial_sample,
        )

    @property
    def target_swap_fraction(self) -> float | None:
        """The theoretical ``C/2`` target for a coupling-controlled schedule."""
        if self.requested_coupling is None:
            return None
        return self.requested_coupling / 2.0

    @property
    def shape(self) -> tuple[int, ...]:
        return self.coupled_results.shape

    def to_numpy(self, *, coupled: bool = True, copy: bool = True) -> np.ndarray:
        """Return either result as a NumPy array."""
        selected = self.coupled_results if coupled else self.base_results
        return np.array(selected, copy=True) if copy else selected

    def component(self, index: int, *, coupled: bool = True) -> np.ndarray:
        """Return one component using the original array's component axis."""
        selected = self.coupled_results if coupled else self.base_results
        return np.take(selected, index, axis=self.component_axis)

    @property
    def first_component(self) -> np.ndarray:
        return self.component(self.component_indices[0])

    @property
    def second_component(self) -> np.ndarray:
        return self.component(self.component_indices[1])

    def as_sample_component_matrix(
        self,
        *,
        coupled: bool = True,
    ) -> np.ndarray:
        """Return an ``(samples, 2)`` view for plotting or tabular use.

        Arrays containing additional non-sample dimensions are intentionally
        rejected because flattening them would silently change their meaning.
        """
        selected = self.coupled_results if coupled else self.base_results
        moved = np.moveaxis(
            selected,
            (self.sample_axis, self.component_axis),
            (0, 1),
        )
        if moved.ndim != 2:
            raise ValueError(
                "plot/table conversion requires only sample and component axes"
            )
        return moved[:, list(self.component_indices)]

    def to_sympy_matrix(self, *, coupled: bool = True):
        """Convert a two-dimensional result to ``sympy.Matrix`` lazily."""
        try:
            import sympy as sp
        except ImportError as exc:
            raise ImportError(
                "to_sympy_matrix requires SymPy; install the 'sympy' package"
            ) from exc
        matrix = self.as_sample_component_matrix(coupled=coupled)
        return sp.Matrix(matrix.tolist())


def swap_result_components(
    base_results: Any,
    swap_mask: Sequence[int] | np.ndarray,
    *,
    sample_axis: int = 0,
    component_axis: int = -1,
    component_indices: Sequence[int] = (0, 1),
    has_initial_sample: bool = False,
    requested_coupling: float | None = None,
    period: int | None = None,
    schedule_kind: str = "explicit-mask",
) -> CoupledResult:
    """Copy an array and exchange two complete components at masked samples.

    The input is never modified.  Numeric, object, and SymPy-backed arrays are
    supported because this operation preserves the original NumPy dtype.
    """
    source = np.asarray(base_results)
    if source.ndim < 2:
        raise ValueError("base_results must have sample and component axes")

    normalized_sample_axis = _normalize_axis(
        sample_axis,
        source.ndim,
        "sample_axis",
    )
    normalized_component_axis = _normalize_axis(
        component_axis,
        source.ndim,
        "component_axis",
    )
    if normalized_sample_axis == normalized_component_axis:
        raise ValueError("sample_axis and component_axis must be different")

    number_of_samples = source.shape[normalized_sample_axis]
    if number_of_samples == 0:
        raise ValueError("base_results must contain at least one sample")
    indices = _validate_component_indices(
        component_indices,
        source.shape[normalized_component_axis],
    )
    mask = _validated_swap_mask(
        swap_mask,
        number_of_samples,
        has_initial_sample=has_initial_sample,
    )

    base_copy = np.array(source, copy=True)
    moved = np.moveaxis(
        base_copy,
        (normalized_sample_axis, normalized_component_axis),
        (0, -1),
    )
    coupled_moved = moved.copy()
    first, second = indices
    if np.any(mask):
        first_values = coupled_moved[mask, ..., first].copy()
        coupled_moved[mask, ..., first] = coupled_moved[mask, ..., second]
        coupled_moved[mask, ..., second] = first_values

    coupled = np.moveaxis(
        coupled_moved,
        (0, coupled_moved.ndim - 1),
        (normalized_sample_axis, normalized_component_axis),
    )
    return CoupledResult(
        base_results=base_copy,
        coupled_results=coupled,
        swap_mask=mask.copy(),
        sample_axis=normalized_sample_axis,
        component_axis=normalized_component_axis,
        component_indices=indices,
        has_initial_sample=bool(has_initial_sample),
        requested_coupling=requested_coupling,
        period=period,
        schedule_kind=str(schedule_kind),
    )


def couple_results(
    base_results: Any,
    *,
    coupling: float | None = None,
    period: int | None = None,
    swap_mask: Sequence[int] | np.ndarray | None = None,
    sample_axis: int = 0,
    component_axis: int = -1,
    component_indices: Sequence[int] = (0, 1),
    has_initial_sample: bool = False,
) -> CoupledResult:
    """Create a coupled copy using ``C``, a period, or an explicit mask.

    Supply at most one of ``coupling``, ``period``, or ``swap_mask``.  Omitting
    all three is equivalent to ``coupling=0``.
    """
    source = np.asarray(base_results)
    if source.ndim < 2:
        raise ValueError("base_results must have sample and component axes")
    normalized_sample_axis = _normalize_axis(sample_axis, source.ndim, "sample_axis")
    number_of_samples = source.shape[normalized_sample_axis]

    supplied = sum(
        value is not None for value in (coupling, period, swap_mask)
    )
    if supplied > 1:
        raise ValueError("supply only one of coupling, period, or swap_mask")

    if swap_mask is not None:
        mask = _validated_swap_mask(
            swap_mask,
            number_of_samples,
            has_initial_sample=has_initial_sample,
        )
        requested = None
        resolved_period = None
        schedule_kind = "explicit-mask"
    elif period is not None:
        resolved_period = _nonnegative_integer(period, "period")
        mask = periodic_swap_mask(
            number_of_samples,
            resolved_period,
            has_initial_sample=has_initial_sample,
        )
        requested = None
        schedule_kind = "periodic"
    else:
        requested_fraction = _coupling_fraction(
            0.0 if coupling is None else coupling
        )
        requested = float(requested_fraction)
        resolved_period = period_for_coupling(requested)
        mask = coupling_swap_mask(
            number_of_samples,
            requested,
            has_initial_sample=has_initial_sample,
        )
        schedule_kind = "coupling-control"

    return swap_result_components(
        source,
        mask,
        sample_axis=normalized_sample_axis,
        component_axis=component_axis,
        component_indices=component_indices,
        has_initial_sample=has_initial_sample,
        requested_coupling=requested,
        period=resolved_period,
        schedule_kind=schedule_kind,
    )


def coupled_family(
    base_results: Any,
    couplings: Iterable[float] = STANDARD_COUPLINGS,
    *,
    sample_axis: int = 0,
    component_axis: int = -1,
    component_indices: Sequence[int] = (0, 1),
    has_initial_sample: bool = False,
) -> dict[float, CoupledResult]:
    """Build several coupled copies from the same untouched base array."""
    family: dict[float, CoupledResult] = {}
    for value in couplings:
        coefficient = float(_coupling_fraction(value))
        if coefficient in family:
            raise ValueError(f"duplicate coupling value: {coefficient}")
        family[coefficient] = couple_results(
            base_results,
            coupling=coefficient,
            sample_axis=sample_axis,
            component_axis=component_axis,
            component_indices=component_indices,
            has_initial_sample=has_initial_sample,
        )
    if not family:
        raise ValueError("couplings must contain at least one value")
    return family


def numerical_gradient(
    function: Callable[..., float],
    parameters: Sequence[float] | np.ndarray,
    eps: float | Sequence[float] | np.ndarray = 1e-6,
    *,
    unpack: bool = False,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> tuple[float, ...]:
    """Return independent central-difference partials as an ordered tuple."""
    raw_parameters = np.asarray(parameters)
    if np.iscomplexobj(raw_parameters):
        raise TypeError("parameters must contain real values")
    point = np.asarray(parameters, dtype=np.float64)
    if point.ndim != 1 or point.size == 0:
        raise ValueError("parameters must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(point)):
        raise ValueError("parameters must contain only finite values")

    raw_steps = np.asarray(eps, dtype=np.float64)
    if raw_steps.ndim == 0:
        steps = np.full(point.shape, float(raw_steps), dtype=np.float64)
    elif raw_steps.shape == point.shape:
        steps = raw_steps.copy()
    else:
        raise ValueError("eps must be scalar or have one value per parameter")
    if not np.all(np.isfinite(steps)) or np.any(steps <= 0.0):
        raise ValueError("every eps value must be positive and finite")

    fixed_args = tuple(args)
    fixed_kwargs = {} if kwargs is None else dict(kwargs)

    def evaluate(location: np.ndarray) -> float:
        if unpack:
            value = function(
                *location.tolist(),
                *fixed_args,
                **fixed_kwargs,
            )
        else:
            value = function(location, *fixed_args, **fixed_kwargs)
        array = np.asarray(value)
        if array.ndim != 0:
            raise TypeError("function must return one scalar")
        if np.iscomplexobj(array):
            raise TypeError("function must return a real scalar")
        scalar = float(array)
        if not np.isfinite(scalar):
            raise ValueError("function returned a non-finite value")
        return scalar

    partials = []
    for index, step in enumerate(steps):
        plus = point.copy()
        minus = point.copy()
        plus[index] += step
        minus[index] -= step
        partials.append((evaluate(plus) - evaluate(minus)) / (2.0 * step))
    return tuple(float(value) for value in partials)


def numerical_gradient_array(*args, **kwargs) -> np.ndarray:
    """NumPy-array form of :func:`numerical_gradient`."""
    return np.asarray(numerical_gradient(*args, **kwargs), dtype=np.float64)


def numerical_gradient_history(
    function: Callable[..., float],
    parameter_points: Sequence[Sequence[float]] | np.ndarray,
    eps: float | Sequence[float] | np.ndarray = 1e-6,
    *,
    unpack: bool = False,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Evaluate one uncoupled numerical gradient tuple at every point."""
    points = np.asarray(parameter_points, dtype=np.float64)
    if points.ndim != 2 or len(points) == 0 or points.shape[1] == 0:
        raise ValueError("parameter_points must have shape (samples, variables)")
    rows = [
        numerical_gradient(
            function,
            point,
            eps,
            unpack=unpack,
            args=args,
            kwargs=kwargs,
        )
        for point in points
    ]
    return np.asarray(rows, dtype=np.float64)


def couple_gradient_history(
    function: Callable[..., float],
    parameter_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    coupling: float | None = None,
    period: int | None = None,
    swap_mask: Sequence[int] | np.ndarray | None = None,
    eps: float | Sequence[float] | np.ndarray = 1e-6,
    unpack: bool = False,
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    component_indices: Sequence[int] = (0, 1),
    has_initial_sample: bool = False,
) -> CoupledResult:
    """Compute one base gradient history, then couple only its stored values."""
    base_gradients = numerical_gradient_history(
        function,
        parameter_points,
        eps,
        unpack=unpack,
        args=args,
        kwargs=kwargs,
    )
    return couple_results(
        base_gradients,
        coupling=coupling,
        period=period,
        swap_mask=swap_mask,
        component_indices=component_indices,
        has_initial_sample=has_initial_sample,
    )


def _require_sympy():
    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError(
            "this adapter requires SymPy; install the 'sympy' package"
        ) from exc
    return sp


def symbolic_gradient(expression: Any, variables: Sequence[Any]) -> tuple[Any, ...]:
    """Return an exact SymPy gradient tuple without adding its components."""
    sp = _require_sympy()
    symbols = tuple(variables)
    if not symbols:
        raise ValueError("variables must contain at least one SymPy symbol")
    return tuple(sp.diff(expression, symbol) for symbol in symbols)


def sympy_gradient_history(
    expression: Any,
    variables: Sequence[Any],
    parameter_points: Sequence[Sequence[float]] | np.ndarray,
    *,
    coupling: float | None = None,
    period: int | None = None,
    swap_mask: Sequence[int] | np.ndarray | None = None,
    modules: Any = "numpy",
    dtype: Any = np.float64,
    component_indices: Sequence[int] = (0, 1),
    has_initial_sample: bool = False,
) -> CoupledResult:
    """Evaluate a SymPy gradient once, then couple the resulting NumPy array."""
    sp = _require_sympy()
    symbols = tuple(variables)
    gradient_expressions = symbolic_gradient(expression, symbols)
    evaluator = sp.lambdify(symbols, gradient_expressions, modules=modules)

    points = np.asarray(parameter_points)
    if points.ndim != 2 or len(points) == 0:
        raise ValueError("parameter_points must have shape (samples, variables)")
    if points.shape[1] != len(symbols):
        raise ValueError("each parameter point must match the number of variables")

    rows = []
    for point in points:
        evaluated = evaluator(*point.tolist())
        row = np.asarray(evaluated, dtype=dtype).reshape(-1)
        if len(row) != len(symbols):
            raise ValueError("SymPy gradient evaluation returned an invalid shape")
        rows.append(row)
    base_gradients = np.stack(rows, axis=0)

    return couple_results(
        base_gradients,
        coupling=coupling,
        period=period,
        swap_mask=swap_mask,
        component_indices=component_indices,
        has_initial_sample=has_initial_sample,
    )


def _require_matplotlib_pyplot():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "plotting requires Matplotlib; install the 'matplotlib' package"
        ) from exc
    return plt


def plot_coupled_components(
    result: CoupledResult,
    *,
    x: Sequence[Any] | np.ndarray | None = None,
    ax: Any = None,
    component_labels: Sequence[str] = ("component 0", "component 1"),
    show_base: bool = False,
    mark_swaps: bool = True,
    title: str | None = None,
    line_kwargs: Mapping[str, Any] | None = None,
):
    """Plot the two selected components on an existing or new Matplotlib axis."""
    if not isinstance(result, CoupledResult):
        raise TypeError("result must be a CoupledResult")
    if len(component_labels) != 2:
        raise ValueError("component_labels must contain exactly two labels")

    plt = _require_matplotlib_pyplot()
    if ax is None:
        _, ax = plt.subplots()

    coupled_matrix = result.as_sample_component_matrix(coupled=True)
    base_matrix = result.as_sample_component_matrix(coupled=False)
    horizontal = (
        np.arange(len(coupled_matrix))
        if x is None
        else np.asarray(x)
    )
    if horizontal.ndim != 1 or len(horizontal) != len(coupled_matrix):
        raise ValueError("x must be one-dimensional with one value per sample")

    options = {} if line_kwargs is None else dict(line_kwargs)
    plotted_lines = []
    for component in range(2):
        label = str(component_labels[component])
        line = ax.plot(
            horizontal,
            coupled_matrix[:, component],
            label=label,
            **options,
        )[0]
        plotted_lines.append(line)
        if show_base:
            ax.plot(
                horizontal,
                base_matrix[:, component],
                color=line.get_color(),
                linestyle="--",
                alpha=0.38,
                linewidth=max(0.8, line.get_linewidth() * 0.8),
                label=f"base {label}",
                zorder=max(0, line.get_zorder() - 1),
            )

    if mark_swaps and np.any(result.swap_mask):
        for component, line in enumerate(plotted_lines):
            ax.scatter(
                horizontal[result.swap_mask],
                coupled_matrix[result.swap_mask, component],
                s=28,
                facecolors="none",
                edgecolors=line.get_color(),
                linewidths=1.0,
                zorder=line.get_zorder() + 1,
            )

    if title is None:
        if result.requested_coupling is None:
            title = (
                f"{result.schedule_kind}; "
                f"swap fraction={result.actual_swap_fraction:.3g}"
            )
        else:
            title = (
                f"C={result.requested_coupling:g}; "
                f"swap fraction={result.actual_swap_fraction:.3g}"
            )
    ax.set_title(title)
    ax.set_xlabel("sample")
    ax.set_ylabel("component value")
    ax.grid(True, alpha=0.25)
    ax.legend()
    return ax


def plot_coupled_family(
    family: Mapping[float, CoupledResult] | Sequence[CoupledResult],
    *,
    x: Sequence[Any] | np.ndarray | None = None,
    axes: Sequence[Any] | np.ndarray | None = None,
    component_labels: Sequence[str] = ("component 0", "component 1"),
    show_base: bool = False,
    mark_swaps: bool = True,
    figsize: tuple[float, float] | None = None,
):
    """Plot a family such as C=0, 0.25, 0.5, and 1 in separate panels."""
    plt = _require_matplotlib_pyplot()
    if isinstance(family, Mapping):
        items = list(family.items())
    else:
        items = [
            (
                result.requested_coupling
                if result.requested_coupling is not None
                else index,
                result,
            )
            for index, result in enumerate(family)
        ]
    if not items:
        raise ValueError("family must contain at least one CoupledResult")
    if not all(isinstance(result, CoupledResult) for _, result in items):
        raise TypeError("every family value must be a CoupledResult")

    if axes is None:
        if figsize is None:
            figsize = (3.4 * len(items), 3.5)
        figure, axes_array = plt.subplots(
            1,
            len(items),
            figsize=figsize,
            squeeze=False,
        )
        flat_axes = axes_array.ravel()
    else:
        flat_axes = np.asarray(axes, dtype=object).ravel()
        if len(flat_axes) != len(items):
            raise ValueError("axes must contain one Matplotlib axis per result")
        figure = flat_axes[0].figure

    for axis, (label, result) in zip(flat_axes, items):
        plot_coupled_components(
            result,
            x=x,
            ax=axis,
            component_labels=component_labels,
            show_base=show_base,
            mark_swaps=mark_swaps,
            title=(
                f"C={label:g}; swap={result.actual_swap_fraction:.3g}"
                if isinstance(label, (int, float, np.number))
                else str(label)
            ),
        )
    figure.tight_layout()
    return figure, flat_axes


def _demonstration() -> None:
    """Small dependency-free verification when the library is run directly."""
    base = np.column_stack(
        (
            np.linspace(0.0, 1.0, 17),
            np.linspace(10.0, 11.0, 17),
        )
    )
    family = coupled_family(
        base,
        STANDARD_COUPLINGS,
        has_initial_sample=True,
    )
    fractions = {
        coupling: result.actual_swap_fraction
        for coupling, result in family.items()
    }
    assert fractions == {0.0: 0.0, 0.25: 0.125, 0.5: 0.25, 1.0: 0.5}
    print("coupled_results_library", __version__)
    print("actual swap fractions:", fractions)


if __name__ == "__main__":
    _demonstration()
