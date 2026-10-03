
import numpy as np
import matplotlib.pyplot as plt

A = np.diag([-1.0, -2.0])
P = np.array([[0.0, 1.0], [1.0, 0.0]])
A0 = A                 # unswapped field
A1 = P @ A             # swapped field
I = np.eye(2)

def M2(h):
    """Period-two monodromy for the alternating schedule."""
    return (I + h*A1) @ (I + h*A0)

def rho(h):
    return max(abs(np.linalg.eigvals(M2(h))))

# --- Reproduce Table 1 ---------------------------------------------
print(f"{'h':>6} {'trace':>10} {'det':>10} {'rho':>10}")
for h in [0.05, 0.20, 0.50, 0.90, 1.00, 1.20]:
    M = M2(h)
    print(f"{h:6.2f} {np.trace(M):10.4f} {np.linalg.det(M):10.4f} {rho(h):10.4f}")

# --- Figure 1: spectral radius vs h --------------------------------
hs = np.linspace(1e-3, 1.4, 600)
rhos = np.array([rho(h) for h in hs])
plt.figure()
plt.plot(hs, rhos, label=r'$\rho(\mathbf{M}_2)$')
plt.axhline(1.0, ls='--', c='k', label=r'$\rho=1$')
plt.axvline(1.0, ls=':', c='r', label=r'$h=1$')
plt.xlabel('step size h'); plt.ylabel('spectral radius')
plt.legend(); plt.tight_layout(); plt.savefig('Fig1.tif', dpi=300)

# --- Figure 2: trajectory norms at h = 0.5 -------------------------
h = 0.5
N = 40
def run(schedule, x0=np.array([1.0, 1.0])):
    x = x0.copy(); out = [x.copy()]
    for s in schedule:
        F = A @ x
        x = x + h * ((1 - s) * F + s * (P @ F))
        out.append(x.copy())
    return np.array(out)

sched_uncoupled  = [0]*N
sched_full       = [1]*N
sched_alternating= [k % 2 for k in range(N)]

plt.figure()
for name, sched in [('uncoupled', sched_uncoupled),
                    ('fully coupled', sched_full),
                    ('alternating', sched_alternating)]:
    traj = run(sched)
    plt.semilogy(np.linalg.norm(traj, axis=1), label=name)
plt.xlabel('step k'); plt.ylabel(r'$\|\mathbf{x}_k\|$')
plt.legend(); plt.tight_layout(); plt.savefig('Fig2.tif', dpi=300)

# --- Figure 3: verification scatter --------------------------------
plt.figure()
plt.scatter(hs, rhos, s=4, label='direct eigenvalue computation')
plt.axhline(1.0, ls='--', c='k')
plt.xlabel('step size h'); plt.ylabel('spectral radius')
plt.legend(); plt.tight_layout(); plt.savefig('Fig3.tif', dpi=300)
