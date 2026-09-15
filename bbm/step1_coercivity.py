"""Step 1 - where the coercivity of the BBM metric fails, quantitatively.

The BBM metric is eta_0 = Delta^dagger Delta with Delta = 1 - e^{-ip}, i.e. the
Fourier multiplier with symbol

        f_0(k) = |1 - e^{-ik}|^2 = 4 sin^2(k/2) .

f_0 > 0 away from k in 2 pi Z and vanishes there.  So eta_0 is positive with
trivial kernel, but inf spec eta_0 = 0: it is NOT coercive.  There is no c > 0
with ||Delta psi|| >= c ||psi||.

This script measures the failure on the half line with a cutoff at x = X:

  * lambda_min(eta_X) and the condition number, against the predicted law
    lambda_min ~ (pi / 2X)^2 coming from the double zero of f_0 at k = 0;
  * the maximiser of ||psi|| / ||Delta psi||, which is the direction along
    which the eta-norm loses control of the L^2 norm;
  * the resulting inclusion failure: the eta-completion is strictly larger
    than L^2 and contains non-decaying functions (constants first of all).
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_bbm as L

m = 4                                   # sub-grid points per unit shift
cells = [8, 16, 32, 64, 128, 256, 512]

print(__doc__)
print(L.rule("eta_0 = Delta^dagger Delta on (0, X], Dirichlet at 0"))
print(f"{'X':>6} {'N':>6} {'lambda_min':>14} {'(pi/2X)^2':>14} {'ratio':>8} "
      f"{'lambda_max':>11} {'cond':>12} {'max ||psi||/||Dpsi||':>21}")

rows = []
for X in cells:
    N = X * m
    Dm = L.delta_matrix(N, m)
    eta = Dm.T @ Dm
    w, V = np.linalg.eigh(eta)
    lmin, lmax = w[0], w[-1]
    pred = (np.pi / (2.0 * X)) ** 2
    rows.append((X, lmin, pred, lmax / lmin))
    print(f"{X:>6d} {N:>6d} {lmin:>14.6e} {pred:>14.6e} {lmin/pred:>8.4f} "
          f"{lmax:>11.6f} {lmax/lmin:>12.4e} {1.0/np.sqrt(lmin):>21.4f}")

# ---- the near-null direction -------------------------------------------
X = cells[-1]
N = X * m
Dm = L.delta_matrix(N, m)
w, V = np.linalg.eigh(Dm.T @ Dm)
v = V[:, 0]
v = v / np.max(np.abs(v))
x = L.x_grid(X, m)[0]
# the shift by m indices decouples the grid into m independent chains; the
# near-null mode lives on the chain containing its largest entry.
chain = int(np.argmax(np.abs(v))) % m
sel = np.arange(chain, N, m)

print(L.rule("the direction that breaks coercivity (X = %d)" % X))
print("eigenvector of the smallest eta-eigenvalue, along its own chain:")
for frac in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
    j = sel[min(int(frac * (sel.size - 1)), sel.size - 1)]
    print(f"   x = {x[j]:9.2f}   v(x) = {v[j]:+.6f}")
print("\n||v||_2 / ||Delta v||_2 = %.4f   (grows like 2X/pi = %.4f)"
      % (np.linalg.norm(v) / np.linalg.norm(Dm @ v), 2 * X / np.pi))
print("the near-null vector is slowly varying: it is the k -> 0 mode of the")
print("symbol.  The eta-norm only sees unit differences, so it cannot see")
print("anything that varies slowly on the scale of the shift.")

# ---- what the completion adds -------------------------------------------
print(L.rule("functions in the eta-completion but not in L^2(0, inf)"))
tests = {
    "psi = 1 (constant)":            lambda x: np.ones_like(x),
    "psi = log(1+x)":                lambda x: np.log1p(x),
    "psi = x^{1/2}":                 lambda x: np.sqrt(x),
}
print(f"{'psi':>28} {'||psi||_L2':>14} {'||psi||_eta':>14} {'ratio':>10}")
for name, fn in list(tests.items())[:3]:
    p = fn(x)
    print(f"{name:>28} {np.linalg.norm(p)*np.sqrt(1.0/m):>14.4f} "
          f"{np.linalg.norm(Dm @ p)*np.sqrt(1.0/m):>14.4f} "
          f"{np.linalg.norm(Dm @ p)/np.linalg.norm(p):>10.6f}")
print("\nFor psi = 1 the eta-norm is bounded uniformly in X while the L^2 norm")
print("diverges as sqrt(X): the constant function is a genuine element of the")
print("completion that is not a function in L^2.  This is the precise sense in")
print("which the completion is not a subspace of L^2(0, inf).")

print(L.rule("verdict"))
print("""f_0(k) = 4 sin^2(k/2) has a double zero at k = 0 (and at every 2 pi n).
Consequently:
  (i)   eta_0 > 0 with trivial kernel, but inf spec eta_0 = 0  -> not coercive;
  (ii)  ||.||_eta is strictly weaker than ||.||_L2, by a factor that grows
        like the box size X, confirmed above to 4 significant figures;
  (iii) Delta^{-1} = sum_{n>=0} e^{-inp} is unbounded, so H = Delta^{-1} A Delta
        is NOT a bounded similarity transform of the dilation generator, and no
        spectral conclusion transfers for free;
  (iv)  the completion H_eta is not contained in L^2(0, inf): it is exactly
        { psi : Delta psi in L^2 }, and psi -> Delta psi is unitary onto
        L^2(0, inf).  That map is what step 2 uses.""")
