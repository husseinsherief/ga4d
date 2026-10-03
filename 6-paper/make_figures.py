import numpy as np
import matplotlib.pyplot as plt

a, b = 1.0, 2.0
A = np.diag([-a, -b])
P = np.array([[0.0, 1.0], [1.0, 0.0]])
I = np.eye(2)

def L1(beta):
    """Damped permutation operator L_1(beta) = (1-beta)I + beta P."""
    return (1.0 - beta) * I + beta * P

def A1(beta):
    """Active field under s = 1 with damped permutation."""
    return L1(beta) @ A

def eig_A1(beta):
    return np.sort(np.real(np.linalg.eigvals(A1(beta))))

def h_stable(beta):
    """Closed-form stability window for damped full coupling."""
    lam = eig_A1(beta)
    if lam.max() >= 0.0:
        return 0.0
    return 2.0 / abs(lam.min())

def rho_1step(beta, h):
    """Spectral radius of the one-step forward Euler map at s=1."""
    M = I + h * A1(beta)
    return max(abs(np.linalg.eigvals(M)))

# --- Reproduce Table 1 ---------------------------------------------
print(f"{'beta':>6} {'lam+':>10} {'lam-':>10} {'det':>10} {'h_stable':>10}")
for beta in [0.00, 0.10, 0.25, 0.40, 0.49]:
    lam = eig_A1(beta)
    det = np.linalg.det(A1(beta))
    print(f"{beta:6.2f} {lam[1]:10.4f} {lam[0]:10.4f} {det:10.4f} {h_stable(beta):10.4f}")

# --- Figure 1: h_stable(beta) --------------------------------------
betas = np.linspace(0.0, 0.4999, 500)
hs = np.array([h_stable(b) for b in betas])
plt.figure()
plt.plot(betas, hs, label=r'$h_{\mathrm{stable}}(\beta)$')
plt.axhline(1.0, ls='--', c='k', label=r'uncoupled bound $h=1$')
plt.axhline(4.0/3.0, ls=':', c='r', label=r'$\beta \to 1/2$ limit $h=4/3$')
plt.xlabel(r'exchange strength $\beta$')
plt.ylabel(r'stability window $h$')
plt.legend(); plt.tight_layout(); plt.savefig('Fig1.tif', dpi=300)

# --- Figure 2: trajectory norms at h = 0.5 -------------------------
h = 0.5
N = 200
x0 = np.array([1.0, 1.0])

def run_full(beta, h, N, x0):
    x = x0.copy(); out = [x.copy()]
    M = I + h * A1(beta)
    for _ in range(N):
        x = M @ x
        out.append(x.copy())
    return np.array(out)

def run_uncoupled(h, N, x0):
    x = x0.copy(); out = [x.copy()]
    M = I + h * A
    for _ in range(N):
        x = M @ x
        out.append(x.copy())
    return np.array(out)

plt.figure()
plt.semilogy(np.linalg.norm(run_uncoupled(h, N, x0), axis=1),
             label='uncoupled')
plt.semilogy(np.linalg.norm(run_full(0.25, h, N, x0), axis=1),
             label=r'damped full coupling, $\beta=1/4$')
plt.semilogy(np.linalg.norm(run_full(1.0, h, N, x0), axis=1),
             label=r'undamped full coupling, $\beta=1$')
plt.xlabel('step k'); plt.ylabel(r'$\|\mathbf{x}_k\|$')
plt.legend(); plt.tight_layout(); plt.savefig('Fig2.tif', dpi=300)

# --- Figure 3: verification scatter --------------------------------
plt.figure()
test_betas = np.linspace(0.0, 0.499, 60)
h_ana = np.array([h_stable(b) for b in test_betas])
h_num = []
for b in test_betas:
    lo, hi = 1e-6, 4.0
    if rho_1step(b, lo) >= 1.0:
        h_num.append(0.0); continue
    if rho_1step(b, hi) < 1.0:
        h_num.append(hi); continue
    while hi - lo > 1e-9:
        mid = 0.5 * (lo + hi)
        if rho_1step(b, mid) < 1.0:
            lo = mid
        else:
            hi = mid
    h_num.append(0.5 * (lo + hi))
h_num = np.array(h_num)
plt.scatter(test_betas, h_num, s=6, label='numerical bisection')
plt.plot(test_betas, h_ana, c='r', lw=1.2, label='analytical')
plt.axhline(1.0, ls='--', c='k')
plt.xlabel(r'exchange strength $\beta$')
plt.ylabel(r'stability window $h$')
plt.legend(); plt.tight_layout(); plt.savefig('Fig3.tif', dpi=300)
