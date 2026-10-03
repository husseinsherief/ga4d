# Coupled Results Library

`coupled_results_library.py` is the reusable implementation of the current
two-component coupling experiment. NumPy is its only required dependency.
SymPy and Matplotlib support is available through optional, lazily imported
adapter functions.

## Core contract

The library separates **source calculation** from **result coupling**:

1. Calculate one uncoupled result array.
2. Preserve that array as `base_results`.
3. Copy it to `coupled_results`.
4. At scheduled sample indices, exchange two complete component values in the
   copy.
5. Do not feed exchanged values back into the source calculation.

For a two-column array, a swapped sample is therefore

\[
(a_i,b_i)\longmapsto(b_i,a_i).
\]

This is a post-processing operation. It does not rerun an integrator, change a
function, or change later rows of the base array.

## Meaning of `C`

The current experiment intentionally uses:

| Requested `C` | Schedule | Target swap fraction |
|---:|---|---:|
| `0` | never | `0` |
| `0.25` | every 8th step | `1/8 = 0.125` |
| `0.5` | every 4th step | `1/4 = 0.25` |
| `1` | every 2nd step | `1/2 = 0.5` |

Consequently, the schedule target is

\[
\text{target swap fraction}=\frac{C}{2}.
\]

This distinction is explicit in every `CoupledResult`:

- `requested_coupling` is the requested `C`;
- `target_swap_fraction` is `C/2`;
- `actual_swap_fraction` is the fraction measured from the finite mask.

The actual fraction can differ slightly from the target for short arrays.

## Installation and imports

Only NumPy is needed for the core:

```python
import numpy as np

from coupled_results_library import (
    couple_results,
    coupled_family,
    coupling_swap_mask,
    numerical_gradient,
)
```

The optional adapters require their corresponding packages:

```bash
python -m pip install sympy matplotlib
```

Importing `coupled_results_library` does not import either optional package.

## NumPy: couple an existing result array

The common input shape is `(samples, components)`:

```python
import numpy as np

from coupled_results_library import couple_results

base = np.array(
    [
        [0.8, 0.0],
        [0.7, 0.2],
        [0.6, 0.4],
        [0.5, 0.6],
        [0.4, 0.8],
    ],
    dtype=float,
)

result = couple_results(base, coupling=1.0)

print(result.base_results)
print(result.coupled_results)
print(result.swap_mask)
print(result.actual_swap_fraction)
```

`base` is never modified. `result.base_results` and
`result.coupled_results` are independent NumPy arrays.

Useful accessors include:

```python
coupled_array = result.to_numpy()
base_array = result.to_numpy(coupled=False)

first = result.first_component
second = result.second_component

matrix = result.as_sample_component_matrix()
```

## Histories containing an initial condition

Integrator output commonly has `N+1` rows:

- row `0`: initial condition;
- rows `1...N`: one-based integration steps.

Set `has_initial_sample=True` so row zero is never swapped:

```python
result = couple_results(
    theta_phi_history,
    coupling=0.5,
    has_initial_sample=True,
)
```

For `C=0.5`, this marks integration steps `4, 8, 12, ...`, which correspond
to array indices `4, 8, 12, ...`.

## Build all four standard coupling views

```python
from coupled_results_library import coupled_family

family = coupled_family(
    theta_phi_history,
    couplings=(0, 0.25, 0.5, 1),
    has_initial_sample=True,
)

c_zero = family[0.0].coupled_results
c_quarter = family[0.25].coupled_results
c_half = family[0.5].coupled_results
c_one = family[1.0].coupled_results
```

Every member is generated from the same untouched input array. Coupled members
are not generated from one another.

## Periodic and explicit schedules

Use an explicit period instead of `C`:

```python
every_eighth = couple_results(base, period=8)
```

Or supply a binary mask:

```python
mask = np.array([0, 1, 0, 0, 1], dtype=np.int8)
custom = couple_results(base, swap_mask=mask)
```

Only one of `coupling`, `period`, or `swap_mask` may be supplied.

Masks can also be generated independently:

```python
from coupled_results_library import coupling_swap_mask, periodic_swap_mask

from_c = coupling_swap_mask(40, coupling=0.5)
periodic = periodic_swap_mask(40, period=4)

assert np.array_equal(from_c, periodic)
```

## Arbitrary NumPy axes and additional components

The sample and component axes are configurable. Suppose an array is organized
as `(batch, components, samples)`:

```python
result = couple_results(
    data,
    coupling=0.25,
    sample_axis=2,
    component_axis=1,
    component_indices=(0, 2),
)
```

At each marked sample, components `0` and `2` are exchanged independently for
every batch. Other components are unchanged.

The input dtype is preserved, including `float32`, integers, and `object`
arrays.

## Numerical gradients

`numerical_gradient` differentiates an arbitrary real scalar-valued Python or
NumPy callable with central differences. It always returns independent
components as an ordered tuple; it never sums them.

Array-style callable:

```python
from coupled_results_library import numerical_gradient

def f(point):
    x, y = point
    return x**2 * y

gradient = numerical_gradient(f, (1.0, 2.0))
assert np.allclose(gradient, (4.0, 1.0))
```

Unpacked callable:

```python
def f(x, y):
    return x**2 * y

gradient = numerical_gradient(f, (1.0, 2.0), unpack=True)
```

Different finite-difference steps can be used per coordinate:

```python
gradient = numerical_gradient(f, (1.0, 2.0), eps=(1e-5, 1e-6), unpack=True)
```

For an array result instead of a tuple:

```python
from coupled_results_library import numerical_gradient_array

gradient_array = numerical_gradient_array(f, (1.0, 2.0), unpack=True)
```

## Calculate one gradient history, then couple it

```python
from coupled_results_library import couple_gradient_history

points = np.array(
    [
        [1.0, 2.0],
        [0.8, 1.9],
        [0.6, 1.7],
        [0.4, 1.5],
    ]
)

result = couple_gradient_history(
    lambda x, y: x**2 * y,
    points,
    coupling=1.0,
    unpack=True,
)
```

All raw gradients are calculated first from `points`. Coupling is applied only
to the completed `(G_x, G_y)` result array.

## SymPy

Exact symbolic gradient:

```python
import sympy as sp

from coupled_results_library import symbolic_gradient

x, y = sp.symbols("x y")
gradient = symbolic_gradient(x**2 * y, (x, y))

assert gradient == (2*x*y, x**2)
```

Evaluate a symbolic gradient over points and couple the numeric result:

```python
from coupled_results_library import sympy_gradient_history

points = np.array([[1.0, 2.0], [0.8, 1.9], [0.6, 1.7]])

result = sympy_gradient_history(
    x**2 * y,
    (x, y),
    points,
    coupling=0.5,
)
```

A two-dimensional `CoupledResult` can be converted back to SymPy:

```python
sympy_matrix = result.to_sympy_matrix()
```

Object arrays containing SymPy expressions can also be passed directly to
`couple_results`; the object dtype is preserved.

## Matplotlib

Plot one result:

```python
import matplotlib.pyplot as plt

from coupled_results_library import plot_coupled_components

ax = plot_coupled_components(
    result,
    component_labels=("theta", "phi"),
    show_base=True,
    mark_swaps=True,
)
plt.show()
```

The function returns the Matplotlib axis and never calls `plt.show()` itself.

Plot the standard family:

```python
from coupled_results_library import coupled_family, plot_coupled_family

family = coupled_family(
    theta_phi_history,
    has_initial_sample=True,
)

figure, axes = plot_coupled_family(
    family,
    x=time,
    component_labels=("theta", "phi"),
    show_base=False,
    mark_swaps=True,
)

figure.savefig("coupled-theta-phi.png", dpi=160)
```

You may supply existing Matplotlib axes to either plotting helper.

## API summary

| Name | Purpose |
|---|---|
| `couple_results` | Main entry point for one coupled copy |
| `coupled_family` | Build several `C` views from one base array |
| `swap_result_components` | Low-level explicit-mask operation |
| `coupling_swap_mask` | Generate the current `C/2` schedule |
| `periodic_swap_mask` | Generate an every-`p` schedule |
| `period_for_coupling` | Recover an exact period when one exists |
| `empirical_swap_frequency` | Measure the finite mask |
| `numerical_gradient` | Central-difference tuple gradient |
| `numerical_gradient_history` | Evaluate base gradients at many points |
| `couple_gradient_history` | Calculate once, then couple stored gradients |
| `symbolic_gradient` | Exact SymPy derivative tuple |
| `sympy_gradient_history` | Evaluate and couple a SymPy gradient |
| `plot_coupled_components` | Plot one `CoupledResult` |
| `plot_coupled_family` | Plot several coupling panels |

## Important non-behaviors

The current result-coupling API does **not**:

- sum gradient components;
- modify its input array;
- swap only update deltas;
- rerun a function after a swap;
- feed a swapped row into calculation of the next row;
- equate requested `C` with the measured swap fraction.

Those distinctions are intentional and tested.

## Legacy module

`tuple_gradient.py` is retained so the older experimental notebook can still
import its increment-coupling functions. New work should import
`coupled_results_library.py`. The two APIs describe different experiments and
should not be mixed accidentally.
