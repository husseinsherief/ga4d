# Specification
# The gradient of a given function \grad f (aka. \nabla f) with parameters (x, y) is currently defined as
# 
# \[
# \grad f = \frac{\partial f}{\partial x} + \frac{\partial f}{\partial y}
#        = \frac{f(x) - f(e_x)}{2e_x} + \frac{f(y) - f(e_y)}{2e_y}
#        = a_x + a_y .
# \]
#
#
# Our idea
#
# We propose to modify the gradient derivative by introducing coupling of the individual components
#
# \[
# \grad f = \left( \frac{\partial f}{\partial x}, \frac{\partial f}{\partial y} \right)
#         = \left( \frac{f(x) - f(e_x)}{2e_x}, \frac{f(y) - f(e_y)}{2e_y} \right).
#   \]
# Where the component sums of the gradient are not summed but rather left independently.
#
# 
#
# We define a coupling frequency for the derivative, \nu(\frac{\partial f}{\partial x})_{y}, for a given function f(x, y),
# defined over a discrete iteration index i∈N (from 1 to ), whereby the partial gradients for x + e_i, y and x, y + e_i remain the same but are
# swapped at a particular sequence of increments proportional to the value of \nu(\frac{\partial f}{\partial x})_{y}.
#
# A more frequent swapping indicates that \nu(\frac{\partial f}{\partial x})_{y} is larger, indicating higher empirical swap frequency,
# whereas a less frequent swapping indicates that it is smaller, and the frequency is lower.
#
#
# Definition
#
# Let \(f: \mathbb{R}^2 \to \mathbb{R}\) be differentiable. At each discrete step \(i\), we evaluate the partial derivatives at the perturbed points:
#
# \[
# G_x^{(i)} = \frac{\partial f}{\partial x}(x + \Delta x_i,\; y), \qquad
# G_y^{(i)} = \frac{\partial f}{\partial y}(x,\; y + \Delta y_i),
# \]
#
# where \(\Delta x_i, \Delta y_i\) are small perturbations or step sizes.
#
# Define a **swap indicator** \(s_i \in \{0,1\}\):
# 
# - \(s_i = 1\) if the **updates are exchanged** at step \(i\) (i.e., we update \(x\) using \(G_y^{(i)}\) and \(y\) using \(G_x^{(i)}\)).
# - \(s_i = 0\) otherwise (standard update).
# 
# The empirical swap frequency over \(N\) steps is:
# \[
# \nu_N = \frac{1}{N} \sum_{i=1}^{N} s_i.
# \]
#
# We define the **coupling coefficient** as this frequency:
# \[
# \boxed{C\!\left(\frac{\partial f}{\partial x}\right)_{\!y} := \nu_N}
# \]
# Thus, a larger \(C\) means swaps occur more often (stronger/faster coupling), and a smaller \(C\) means swaps occur less often (weaker/slower coupling).







