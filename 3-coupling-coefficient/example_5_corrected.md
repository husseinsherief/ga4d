### Example 5: How the revised increment coupling changes the gradient trajectories

To observe the effect of the coupling coefficient on the gradient trajectories, consider the same dynamics as in Example 3: $f(x,y)=x^2y$, step size $\eta=0.1$, initial state $(x_1,y_1)=(1.0,2.0)$, and $N=40$ update steps. We compare four **periodic increment-swap schedules**, each realizing a different coupling coefficient over the 40-step run:

| Schedule | Swap-indicator pattern | $C_{40}(G_x,G_y)=\nu_{40}$ |
|---|---|:---:|
| no swaps | $\mathbf{s}=[\,0,\ 0,\ 0,\ 0,\ \dots\,]$ | $0$ |
| swap every 4th step | $\mathbf{s}=[\,0,\ 0,\ 0,\ 1,\ 0,\ 0,\ 0,\ 1,\ \dots\,]$ | $0.25$ |
| swap every 2nd step | $\mathbf{s}=[\,0,\ 1,\ 0,\ 1,\ 0,\ 1,\ \dots\,]$ | $0.5$ |
| swap every step | $\mathbf{s}=[\,1,\ 1,\ 1,\ 1,\ \dots\,]$ | $1$ |

At every iteration, the raw gradient components retain their original definitions:

$$
G_x^{(i)}=2x_i y_i,
\qquad
G_y^{(i)}=x_i^2.
$$

The standard gradient-descent increments are calculated first:

$$
\Delta x_i=-\eta G_x^{(i)},
\qquad
\Delta y_i=-\eta G_y^{(i)}.
$$

The swap indicator changes only which increment is applied to each coordinate:

$$
s_i=0:
\quad
x_{i+1}=x_i+\Delta x_i,
\qquad
y_{i+1}=y_i+\Delta y_i,
$$

$$
s_i=1:
\quad
x_{i+1}=x_i+\Delta y_i,
\qquad
y_{i+1}=y_i+\Delta x_i.
$$

The revised theory is implemented by the following delta-first update. Notice that neither the state coordinates nor the raw gradient components are reversed:

```python
for i, swap in enumerate(s):
    # 1. Evaluate the unchanged raw gradient at the current state.
    Gx = 2.0 * x[i] * y[i]
    Gy = x[i] ** 2

    # 2. Calculate the ordinary update increments first.
    delta_x = -eta * Gx
    delta_y = -eta * Gy

    # 3. Exchange only those completed increments when s_i = 1.
    if swap:
        applied_x = delta_y
        applied_y = delta_x
    else:
        applied_x = delta_x
        applied_y = delta_y

    # 4. Add the selected increments to the unchanged coordinate ordering.
    x[i + 1] = x[i] + applied_x
    y[i + 1] = y[i] + applied_y
```

For the common scalar step size $\eta=0.1$ used in this example,

$$
\operatorname{swap}(-\eta G_x^{(i)},-\eta G_y^{(i)})
=(-\eta G_y^{(i)},-\eta G_x^{(i)}).
$$

Consequently, the revised delta-first procedure produces the same numerical trajectory as the shorthand equations $x_{i+1}=x_i-\eta G_y^{(i)}$ and $y_{i+1}=y_i-\eta G_x^{(i)}$ on an indicated step. This numerical equivalence holds here because the same scalar $\eta$ multiplies both gradient components. The interpretation is nevertheless different: the completed increments are exchanged, not the gradients or state coordinates.

For example, the $C_{40}=0.5$ schedule begins with $s_1=0$ and $s_2=1$. After the first standard step, the state is

$$
(x_2,y_2)=(0.600,1.900).
$$

At the second step, the unchanged raw gradient and ordinary increments are

$$
(G_x^{(2)},G_y^{(2)})=(2.280,0.360),
$$

$$
(\Delta x_2,\Delta y_2)=(-0.228,-0.036).
$$

Because $s_2=1$, only the increments are exchanged:

$$
(\widetilde{\Delta x_2},\widetilde{\Delta y_2})
=(-0.036,-0.228).
$$

The next state is therefore

$$
(x_3,y_3)
=(0.600,1.900)+(-0.036,-0.228)
=(0.564,1.672),
$$

not $(1.900,0.600)$. The swapped increments alter the subsequent state, causing later evaluations of the unchanged gradient formulas to follow different trajectories.

Each panel below plots the raw gradient trajectories $G_x^{(i)}=2x_i y_i$ and $G_y^{(i)}=x_i^2$ against the iteration index for one value of $C_{40}(G_x,G_y)$:

![Raw gradient trajectories of f(x,y) = x²y after frequency-coupled increments](coupling-frequency-gradients.png)

Recomputing all four runs with the revised delta-first algorithm gives:

| $C_{40}$ | Final $x_{41}$ | Final $y_{41}$ |
|---:|---:|---:|
| $0$ | $0.000000$ | $1.841126$ |
| $0.25$ | $0.000002$ | $1.724836$ |
| $0.5$ | $0.001498$ | $1.243038$ |
| $1$ | $0.193474$ | $0.068278$ |

Reading from left to right:

- **$C_{40}=0$ (uncoupled):** each coordinate receives its own standard increment. The relatively large increment $\Delta x_i$, generated from $G_x^{(i)}$, drives $x$ toward zero within approximately 10 steps. Because both $G_x=2xy$ and $G_y=x^2$ depend on $x$, both gradient components also approach zero. The endpoint is $(x,y)\approx(0.000,1.841)$, so $y$ changes comparatively little.
- **$C_{40}=0.25$:** every fourth step applies $\Delta y_i$ to $x$ and $\Delta x_i$ to $y$. These increment exchanges produce the visible kinks in the $G_x$ trajectory. On a swapped step, $x$ receives the smaller-magnitude increment generated from $G_y$, delaying the decay of $x$ and allowing the gradient components to remain nonzero slightly longer. The endpoint is $(x,y)\approx(0.000,1.725)$.
- **$C_{40}=0.5$:** the increments are exchanged every second step. On alternating iterations, $x$ receives the smaller-magnitude increment $\Delta y_i$, while $y$ receives the larger-magnitude increment $\Delta x_i$. This produces a staircase-like decay in $G_x$, prolongs both gradient trajectories, and causes $y$ to decay substantially. The endpoint is $(x,y)\approx(0.002,1.243)$.
- **$C_{40}=1$ (fully coupled):** the increments are exchanged on every step. Therefore, $x$ always receives $\Delta y_i$, generated from $G_y^{(i)}$, while $y$ always receives $\Delta x_i$, generated from $G_x^{(i)}$. Both gradient trajectories decay more slowly and smoothly, and $y$ is driven close to zero. The endpoint is $(x,y)\approx(0.194,0.068)$.

For this experiment, increasing $C_{40}$ applies the larger-magnitude increment $\Delta x_i$, generated from $G_x^{(i)}$, to $y$ more frequently while applying the smaller-magnitude increment $\Delta y_i$, generated from $G_y^{(i)}$, to $x$. This prolongs the gradient trajectories and shifts more of the decay from $x$ to $y$. The coupling redirects update increments only; it does not exchange the state coordinates or the raw gradient components.

If different coordinate-specific step sizes or different update transformations were used, swapping completed increments would no longer be equivalent to swapping gradient labels before constructing the increments. Under the revised theory, the correct order is always: calculate the raw gradient, construct $\Delta x_i$ and $\Delta y_i$, exchange those completed increments when $s_i=1$, and then add them to $(x_i,y_i)$ without reordering the state.
