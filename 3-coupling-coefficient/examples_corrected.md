## Examples

### Example 1: Computing $\nu_N$ from a swap-indicator array

Suppose over $N=10$ steps we record the swap indicators as an array:

$$
\mathbf{s}=[\,0,\ 1,\ 0,\ 0,\ 1,\ 0,\ 1,\ 0,\ 0,\ 1\,].
$$

The sum of the entries is $\sum_{i=1}^{10}s_i=4$, so the empirical swap frequency is

$$
\nu_{10}=\frac{4}{10}=0.4,
\qquad
C_{10}(G_x,G_y)=0.4.
$$

The update increments were exchanged on 4 of the 10 steps, corresponding to moderate coupling.

### Example 2: Comparing strong and weak coupling

Consider two runs of the same length, $N=10$:

$$
\mathbf{s}^{(A)}=[\,1,\ 1,\ 0,\ 1,\ 1,\ 1,\ 0,\ 1,\ 1,\ 1\,]
\quad\Rightarrow\quad
\nu_{10}^{(A)}=\frac{8}{10}=0.8.
$$

$$
\mathbf{s}^{(B)}=[\,0,\ 0,\ 0,\ 1,\ 0,\ 0,\ 0,\ 0,\ 0,\ 0\,]
\quad\Rightarrow\quad
\nu_{10}^{(B)}=\frac{1}{10}=0.1.
$$

Run $A$ has $C_{10}(G_x,G_y)=0.8$, representing strong or fast coupling because the $x$- and $y$-update increments are exchanged on most steps. Run $B$ has $C_{10}(G_x,G_y)=0.1$, representing weak or slow coupling because the standard increment assignment dominates.

A **periodic** swap schedule is a special case. Swapping the update increments every third step gives

$$
\mathbf{s}=[\,0,\ 0,\ 1,\ 0,\ 0,\ 1,\ 0,\ 0,\ 1,\ \dots\,].
$$

For a finite run of $N$ steps,

$$
\nu_N=\frac{\lfloor N/3\rfloor}{N},
$$

and therefore

$$
\lim_{N\to\infty}\nu_N=\frac{1}{3}.
$$

Thus, $C=1/3$ corresponds asymptotically to one increment exchange every three iterations.

### Example 3: A full trajectory with $f(x,y)=x^2y$

Take $f(x,y)=x^2y$, whose raw partial-gradient components are

$$
G_x^{(i)}
=\frac{\partial f}{\partial x}(x_i,y_i)
=2x_i y_i,
\qquad
G_y^{(i)}
=\frac{\partial f}{\partial y}(x_i,y_i)
=x_i^2.
$$

Using a gradient-descent step size $\eta=0.1$, first calculate the standard increments

$$
\Delta x_i=-\eta G_x^{(i)},
\qquad
\Delta y_i=-\eta G_y^{(i)}.
$$

- **Standard step** ($s_i=0$): $\quad x_{i+1}=x_i+\Delta x_i, \quad y_{i+1}=y_i+\Delta y_i$.
- **Swapped-increment step** ($s_i=1$): $\quad x_{i+1}=x_i+\Delta y_i, \quad y_{i+1}=y_i+\Delta x_i$.

Because the same step size $\eta$ is used for both components, the swapped-increment equations can also be written as

$$
x_{i+1}=x_i-\eta G_y^{(i)},
\qquad
y_{i+1}=y_i-\eta G_x^{(i)}.
$$

This equivalent form does not swap $x_i$ and $y_i$, nor does it redefine or reorder the raw gradient. It only exchanges the already-calculated increments applied to the two coordinates.

Starting from $(x_1,y_1)=(1.0,2.0)$ with swap-indicator array $\mathbf{s}=[\,0,\ 1,\ 0,\ 1,\ 0\,]$ over $N=5$ steps, the trajectory is:

| $i$ | $x_i$ | $y_i$ | $G_x^{(i)}=2x_i y_i$ | $G_y^{(i)}=x_i^2$ | $s_i$ | increment applied to $x$ | increment applied to $y$ |
|---:|---:|---:|---:|---:|:---:|:---:|:---:|
| 1 | 1.0000 | 2.0000 | 4.0000 | 1.0000 | 0 | $\Delta x_1$ | $\Delta y_1$ |
| 2 | 0.6000 | 1.9000 | 2.2800 | 0.3600 | 1 | $\Delta y_2$ | $\Delta x_2$ |
| 3 | 0.5640 | 1.6720 | 1.8860 | 0.3181 | 0 | $\Delta x_3$ | $\Delta y_3$ |
| 4 | 0.3754 | 1.6402 | 1.2314 | 0.1409 | 1 | $\Delta y_4$ | $\Delta x_4$ |
| 5 | 0.3613 | 1.5170 | 1.0962 | 0.1305 | 0 | $\Delta x_5$ | $\Delta y_5$ |
| 6 | 0.2517 | 1.5040 | — | — | — | — | — |

Collected as arrays:

$$
\mathbf{x}=[\,1.0000,\ 0.6000,\ 0.5640,\ 0.3754,\ 0.3613,\ 0.2517\,],
$$

$$
\mathbf{y}=[\,2.0000,\ 1.9000,\ 1.6720,\ 1.6402,\ 1.5170,\ 1.5040\,],
$$

$$
\mathbf{s}=[\,0,\ 1,\ 0,\ 1,\ 0\,].
$$

The coupling coefficient for this run is

$$
C_5(G_x,G_y)
=\nu_5
=\frac{0+1+0+1+0}{5}
=\frac{2}{5}
=0.4.
$$

At the swapped-increment steps $i=2$ and $i=4$, the smaller increment $\Delta y_i$, generated from $G_y^{(i)}$, is applied to $x$, while the larger increment $\Delta x_i$, generated from $G_x^{(i)}$, is applied to $y$. This slows the decay of $x$ and accelerates the decay of $y$ relative to the standard steps. The state coordinates and raw gradient components themselves are never swapped.

### Example 4: Computing $\nu_N$ with NumPy

The empirical coupling coefficient can be calculated directly from a validated binary NumPy array:

```python
import numpy as np

s = np.array([0, 1, 0, 0, 1, 0, 1, 0, 0, 1], dtype=np.int8)

if s.ndim != 1 or s.size == 0:
    raise ValueError("s must be a non-empty one-dimensional array")
if not np.all((s == 0) | (s == 1)):
    raise ValueError("every swap indicator must be either 0 or 1")

N = s.size
nu_N = float(s.sum() / N)  # Equivalently: float(s.mean())
C_N = nu_N

print(nu_N)  # 0.4
print(C_N)   # 0.4
```

Here, `s.sum()` counts the steps on which the update increments were exchanged. Dividing by `N` implements

$$
\nu_N=\frac{1}{N}\sum_{i=1}^{N}s_i,
$$

so this example gives $C_{10}(G_x,G_y)=\nu_{10}=0.4$.
