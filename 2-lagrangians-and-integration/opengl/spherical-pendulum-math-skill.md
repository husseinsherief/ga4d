# Math Skill: Euler-Lagrange Motion, Spherical Pendulum Geometry, and Coupled Angles

## Action, Lagrangian, and Euler-Lagrange Equations

For particle mechanics, the dynamical variable is a generalized coordinate

\[
q=q(t).
\]

The action is

\[
S[q]=\int_{t_0}^{t_1} L(q,\dot q,t)\,dt,
\]

where the Lagrangian is usually

\[
L=T-V.
\]

The Euler-Lagrange equation is

\[
\frac{\partial L}{\partial q}
-
\frac{d}{dt}
\left(
\frac{\partial L}{\partial \dot q}
\right)
=0.
\]

Equivalently,

\[
\frac{d}{dt}
\left(
\frac{\partial L}{\partial \dot q}
\right)
=
\frac{\partial L}{\partial q}.
\]

For a one-dimensional Cartesian system with

\[
L=\frac{1}{2}m\dot q^2 - V(q),
\]

we have

\[
\frac{\partial L}{\partial \dot q}=m\dot q,
\qquad
\frac{d}{dt}\frac{\partial L}{\partial \dot q}=m\ddot q,
\]

and

\[
\frac{\partial L}{\partial q}
=
-\frac{dV}{dq}.
\]

Thus the Euler-Lagrange equation becomes

\[
m\ddot q=-\frac{dV}{dq},
\]

which is Newton's law in potential form:

\[
F=ma,
\qquad
F=-\nabla V.
\]

The Lagrangian \(L=T-V\) is the integrand of the action.  
The Euler-Lagrange equation is the stationarity condition of the action.

## Field-Theory Euler-Lagrange Equation

For a field,

\[
\psi=\psi(x),
\]

where

\[
x=(x^0,x^1,x^2,x^3),
\]

the action is

\[
S[\psi]=\int \mathcal L(\psi,\partial_\mu\psi,x)\,d^4x.
\]

Here

\[
d^4x = dx^0\,dx^1\,dx^2\,dx^3
\]

is the four-dimensional volume element, and \(\mathcal L\) is the Lagrangian density.

Under a variation

\[
\psi \mapsto \psi+\epsilon\eta,
\]

with \(\eta\) vanishing on the boundary, the stationary-action condition gives

\[
\partial_\mu
\left(
\frac{\partial \mathcal L}
{\partial(\partial_\mu \psi)}
\right)
-
\frac{\partial \mathcal L}{\partial \psi}
=0.
\]

For a scalar field with

\[
\mathcal L
=
\frac{1}{2}
(\partial_\mu\psi)(\partial^\mu\psi)
-
\frac{1}{2}m^2\psi^2,
\]

the field Euler-Lagrange equation becomes

\[
\partial_\mu\partial^\mu \psi + m^2\psi = 0.
\]

This is the Klein-Gordon equation.

## Variational and Vertical Integrator Idea

For a separable mechanical system,

\[
H(q,p)=T(p)+V(q),
\]

the Hamiltonian flow can be split into two geometric pieces.

The vertical, or kick, part holds \(q\) fixed and updates momentum:

\[
q_{n+1}=q_n,
\qquad
p_{n+1}=p_n-h\nabla V(q_n).
\]

The horizontal, or drift, part holds \(p\) fixed and updates position:

\[
q_{n+1}=q_n+hM^{-1}p_n,
\qquad
p_{n+1}=p_n.
\]

The Störmer-Verlet composition is

\[
\Phi_h
=
\Phi^V_{h/2}
\circ
\Phi^H_h
\circ
\Phi^V_{h/2}.
\]

The key geometric idea is that the vertical step changes velocity or momentum while the base coordinate is fixed.

## Spherical Pendulum Coordinates

For a spherical pendulum of radius \(R\), use angular coordinates

\[
q=(\theta,\phi),
\]

where:

\[
\theta = \text{polar angle from the vertical axis},
\]

\[
\phi = \text{azimuthal angle around the vertical axis}.
\]

The position on the sphere is

\[
\mathbf x(\theta,\phi)
=
R
\begin{pmatrix}
\sin\theta\cos\phi \\
\sin\theta\sin\phi \\
\cos\theta
\end{pmatrix}.
\]

The velocity magnitude is

\[
|\dot{\mathbf x}|^2
=
R^2
\left(
\dot\theta^2
+
\sin^2\theta\,\dot\phi^2
\right).
\]

Therefore the kinetic energy is

\[
T
=
\frac{1}{2}mR^2
\left(
\dot\theta^2
+
\sin^2\theta\,\dot\phi^2
\right).
\]

With vertical coordinate

\[
z=R\cos\theta,
\]

the potential energy is

\[
V=mgR\cos\theta.
\]

Thus the spherical-pendulum Lagrangian is

\[
L
=
\frac{1}{2}mR^2
\left(
\dot\theta^2
+
\sin^2\theta\,\dot\phi^2
\right)
-
mgR\cos\theta.
\]

## Euler-Lagrange Equations for the Spherical Pendulum

For \(\theta\),

\[
\frac{\partial L}{\partial \dot\theta}
=
mR^2\dot\theta,
\]

\[
\frac{d}{dt}
\left(
\frac{\partial L}{\partial \dot\theta}
\right)
=
mR^2\ddot\theta.
\]

Also,

\[
\frac{\partial L}{\partial \theta}
=
mR^2\sin\theta\cos\theta\,\dot\phi^2
+
mgR\sin\theta.
\]

Therefore,

\[
mR^2\ddot\theta
=
mR^2\sin\theta\cos\theta\,\dot\phi^2
+
mgR\sin\theta,
\]

or

\[
\ddot\theta
=
\sin\theta\cos\theta\,\dot\phi^2
+
\frac{g}{R}\sin\theta.
\]

For \(\phi\),

\[
\frac{\partial L}{\partial \phi}=0,
\]

so \(\phi\) is cyclic. Hence

\[
\frac{d}{dt}
\left(
\frac{\partial L}{\partial \dot\phi}
\right)
=0.
\]

Since

\[
\frac{\partial L}{\partial \dot\phi}
=
mR^2\sin^2\theta\,\dot\phi,
\]

we get the conserved angular momentum relation

\[
mR^2\sin^2\theta\,\dot\phi = \ell,
\]

or

\[
\sin^2\theta\,\dot\phi = C.
\]

Therefore,

\[
\dot\phi
=
\frac{C}{\sin^2\theta}.
\]

Differentiating gives

\[
\ddot\phi
=
-2\cot\theta\,\dot\theta\,\dot\phi.
\]

Thus the spherical pendulum may be written as

\[
\boxed{
\ddot\theta
=
\sin\theta\cos\theta\,\dot\phi^2
+
\frac{g}{R}\sin\theta
}
\]

\[
\boxed{
\ddot\phi
=
-2\cot\theta\,\dot\theta\,\dot\phi
}
\]

with state vector

\[
\mathbf s
=
\begin{pmatrix}
\theta \\
\phi \\
\dot\theta \\
\dot\phi
\end{pmatrix}.
\]

## First-Order Form

The second-order equations can be written as a first-order system:

\[
\frac{d}{dt}
\begin{pmatrix}
\theta \\
\phi \\
\dot\theta \\
\dot\phi
\end{pmatrix}
=
\begin{pmatrix}
\dot\theta \\
\dot\phi \\
\ddot\theta \\
\ddot\phi
\end{pmatrix}.
\]

That is,

\[
\dot{\mathbf s}=f(\mathbf s).
\]

## RK4 Slopes

For a step size \(h\), classical fourth-order Runge-Kutta uses four slopes:

\[
k_1=f(\mathbf s_n),
\]

\[
k_2=f\left(\mathbf s_n+\frac{h}{2}k_1\right),
\]

\[
k_3=f\left(\mathbf s_n+\frac{h}{2}k_2\right),
\]

\[
k_4=f\left(\mathbf s_n+h k_3\right).
\]

The next state is

\[
\mathbf s_{n+1}
=
\mathbf s_n
+
\frac{h}{6}
\left(
k_1+2k_2+2k_3+k_4
\right).
\]

The local truncation error is

\[
O(h^5),
\]

and the global accumulated error over a fixed interval is

\[
O(h^4).
\]

## Geometric Algebra Rotor Parameterization

In three-dimensional Euclidean geometric algebra, let

\[
e_1,e_2,e_3
\]

be an orthonormal basis.

The north pole of the sphere is

\[
e_3.
\]

A rotation in a plane represented by a unit bivector \(B\) is generated by the rotor

\[
R(\alpha)
=
\exp\left(-\frac{\alpha}{2}B\right)
=
\cos\frac{\alpha}{2}
-
B\sin\frac{\alpha}{2}.
\]

The rotated vector is

\[
\mathbf x'
=
R\mathbf x \widetilde R.
\]

For the spherical pendulum, one may view the two basic angular rotations as:

\[
B_\theta = e_{23},
\]

\[
B_\phi = e_{12}.
\]

The \(\theta\)-motion is associated with rotation in the \(e_2e_3\) plane.

The \(\phi\)-motion is associated with rotation in the \(e_1e_2\) plane.

## Coupled Omega Plane

The coupled omega plane is defined as the normalized halfway bivector between the theta and phi planes:

\[
B_\omega
=
\frac{B_\theta+B_\phi}{\sqrt{2}}
=
\frac{e_{23}+e_{12}}{\sqrt{2}}.
\]

It satisfies

\[
B_\omega\cdot B_\omega=-1,
\]

and has equal projection onto the theta and phi planes:

\[
B_\omega\cdot B_\theta
=
B_\omega\cdot B_\phi
=
-\frac{1}{\sqrt{2}}.
\]

Thus \(B_\omega\) is a \(45^\circ\) halfway rotation plane between \(B_\theta\) and \(B_\phi\).

## Coupled Omega Angle

The omega angle is not an independent dynamical coordinate.  
It is derived from the composed \(\theta\)- and \(\phi\)-rotor.

Let

\[
a=\frac{\theta}{2},
\qquad
b=\frac{\phi}{2}.
\]

The scalar component of the composed rotor is

\[
S=\cos a\cos b.
\]

The component projected onto the omega plane is

\[
W
=
\frac{
\cos b\sin a
+
\sin b\cos a
}{\sqrt{2}}.
\]

Therefore the coupled omega angle is

\[
\boxed{
\omega
=
2\arctan2(W,S)
}
\]

or explicitly,

\[
\boxed{
\omega(t)
=
2\arctan2
\left(
\frac{
\cos(\phi(t)/2)\sin(\theta(t)/2)
+
\sin(\phi(t)/2)\cos(\theta(t)/2)
}{\sqrt{2}},
\,
\cos(\theta(t)/2)\cos(\phi(t)/2)
\right).
}
\]

Thus

\[
\omega(t)=F(\theta(t),\phi(t)).
\]

It is a coupled angle induced by the two independent angular coordinates.

## Component Motions

The theta-only motion may be represented by holding \(\phi=0\):

\[
\mathbf x_\theta(t)
=
\begin{pmatrix}
\sin\theta(t) \\
0 \\
\cos\theta(t)
\end{pmatrix}.
\]

This lies in the theta rotation plane.

The phi-only motion may be represented at a fixed reference polar angle \(\theta_0\):

\[
\mathbf x_\phi(t)
=
\begin{pmatrix}
\sin\theta_0\cos\phi(t) \\
\sin\theta_0\sin\phi(t) \\
\cos\theta_0
\end{pmatrix}.
\]

This lies in a horizontal circular plane.

The omega-only motion is represented by rotating the north pole in the coupled plane:

\[
\mathbf x_\omega(t)
=
R_\omega(t)e_3\widetilde R_\omega(t),
\]

where

\[
R_\omega(t)
=
\exp\left(
-\frac{\omega(t)}{2}B_\omega
\right).
\]

The full spherical pendulum motion is

\[
\mathbf x(t)
=
\begin{pmatrix}
\sin\theta(t)\cos\phi(t) \\
\sin\theta(t)\sin\phi(t) \\
\cos\theta(t)
\end{pmatrix}.
\]

## North Pole Reference

For every sphere, the north pole is the fixed point

\[
\mathbf n=e_3
=
\begin{pmatrix}
0\\
0\\
1
\end{pmatrix}.
\]

The north pole serves as the reference point before angular rotation.

## Approximate Angular Behavior

For the current spherical-pendulum parameters,

\[
R=1,
\qquad
g=9.81,
\]

with initial data

\[
\theta(0)=0.8,
\qquad
\phi(0)=0,
\qquad
\dot\theta(0)=0,
\qquad
\dot\phi(0)=1.5,
\]

the conserved azimuthal quantity is

\[
C=\sin^2(0.8)(1.5)\approx 0.772.
\]

The theta angle nutates between approximately

\[
\theta_{\min}\approx 0.8,
\qquad
\theta_{\max}\approx 3.01.
\]

The approximate nutation period is

\[
T\approx 1.47,
\]

with angular frequency

\[
\Omega_\theta
\approx
\frac{2\pi}{T}
\approx
4.28.
\]

A simple leading-order visual approximation is

\[
\theta(t)
\approx
1.90-1.10\cos(4.28t).
\]

Because

\[
\dot\phi=\frac{C}{\sin^2\theta},
\]

the azimuthal angle \(\phi\) advances nonuniformly: it precesses faster when \(\sin\theta\) is smaller and slower when \(\sin\theta\) is larger.

A rough time-series approximation is

\[
\phi(t)
\approx
2.86t
-0.79\sin(4.28t)
+0.43\sin(8.56t)
-0.25\sin(12.84t).
\]

The omega angle is best understood through its coupled formula:

\[
\omega(t)=F(\theta(t),\phi(t)),
\]

rather than as a separate sinusoid.
