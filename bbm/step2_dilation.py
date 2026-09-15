"""Step 2 - the BBM operator in the dilation representation, checked numerically.

Write u = log x and let (U psi)(u) = e^{u/2} psi(e^u), a unitary
L^2(R_+, dx) -> L^2(R, du).  Then

        A = 2D = x p + p x   -->   -2i d/du ,

so A is the dilation generator: on the full line its spectrum is R, simple and
purely absolutely continuous, with generalised eigenfunctions

        A x^{-z} = i(2z-1) x^{-z} ,    E = i(2z-1) real  <=>  z = 1/2 - iE/2 .

The BBM Hamiltonian is H = Delta^{-1} A Delta.  On the eta-completion the map
V: psi -> Delta psi is unitary onto L^2(R_+) and V H V^{-1} = A.  So in the
dilation representation H is nothing but multiplication by the dilation
variable: clean spectrum, and clean bad news.

BBM's eigenfunctions are psi_z(x) = -zeta(z, x+1), which satisfy the Hurwitz
difference identity Delta psi_z = x^{-z} exactly, and psi_z(0) = -zeta(z), so
the boundary condition psi_z(0) = 0 selects the nontrivial zeros.

This script verifies, to 25 significant digits where mpmath is used:
  (1)  A x^{-z} = E x^{-z} with E = i(2z-1);
  (2)  the box spectrum of A is the arithmetic ladder 4 pi n / L_u;
  (3)  Delta psi_z = x^{-z} for psi_z = -zeta(z, x+1);
  (4)  psi_z(0) = 0 exactly at the nontrivial zeros;
  (5)  ||psi_z||^2_eta = ||Delta psi_z||^2 = int x^{-2 Re z} dx = int dx/x,
       which diverges logarithmically at BOTH ends.  The BBM eigenstates are
       therefore not in the completion of their own metric - and they miss it
       only logarithmically, exactly on the critical line.
"""

import sys, os
import numpy as np
import mpmath as mp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_bbm as L

mp.mp.dps = 30
print(__doc__)

# ---- (1) generalised eigenfunctions of the dilation generator -------------
print(L.rule("(1) A x^{-z} = E x^{-z} on a logarithmic grid"))
print(f"{'E':>10} {'z = 1/2 - iE/2':>26} {'rel. residual ||Ax^-z - Ex^-z||':>34}")
n = 4096
u, du = L.u_grid(-8.0, 8.0, n)
Au = L.dilation_matrix_u(n, 16.0)
for E in (2.0, 14.134725141734693, 28.0, -21.022039638771555):
    z = 0.5 - 0.5j * E
    phi = np.exp((0.5 - z) * u)              # = e^{u/2} x^{-z}, the U-image
    res = np.linalg.norm(Au @ phi - E * phi) / np.linalg.norm(E * phi)
    print(f"{E:>10.4f} {str(np.round(z,6)):>26} {res:>34.3e}")

# ---- (2) the box spectrum is an arithmetic ladder -------------------------
print(L.rule("(2) spectrum of A on a periodic u-box: the ladder 4 pi n / L_u"))
print(f"{'L_u':>8} {'4 pi / L_u':>13} {'measured spacing':>18} {'max |dev|':>12}")
for Lu in (8.0, 16.0, 32.0, 64.0):
    n = 512
    ev = np.sort(np.linalg.eigvalsh(L.dilation_matrix_u(n, Lu)))
    mid = ev[n // 2 - 40: n // 2 + 40]
    sp = np.diff(mid)
    pred = L.free_ladder(Lu, 80)
    print(f"{Lu:>8.1f} {4*np.pi/Lu:>13.8f} {sp.mean():>18.8f} "
          f"{np.max(np.abs(np.sort(mid) - np.sort(pred))):>12.2e}")
print("\nThe spacing is exactly 4 pi / L_u and goes to zero as the box grows:")
print("the spectrum is continuous, and any finite computation looks discrete.")

# ---- (3)(4) the Hurwitz identity and the boundary condition ---------------
print(L.rule("(3) Delta psi_z = x^{-z} for psi_z(x) = -zeta(z, x+1)"))
rho = mp.mpf('0.5') + 1j * mp.im(mp.zetazero(1))
print(f"z = rho_1 = {mp.nstr(rho, 20)}")
print(f"{'x':>8} {'(Delta psi_z)(x)':>46} {'x^{-z}':>46} {'|diff|':>12}")
for xv in ('0.5', '1.0', '2.5', '7.0', '40.0'):
    x = mp.mpf(xv)
    lhs = -mp.zeta(rho, x + 1) + mp.zeta(rho, x)          # psi(x) - psi(x-1)
    rhs = x ** (-rho)
    print(f"{xv:>8} {mp.nstr(lhs,12):>46} {mp.nstr(rhs,12):>46} "
          f"{mp.nstr(abs(lhs-rhs),3):>12}")

print(L.rule("(4) psi_z(0) = -zeta(z): the boundary condition IS zeta(z) = 0"))
for j in (1, 2, 3):
    r = mp.mpf('0.5') + 1j * mp.im(mp.zetazero(j))
    print(f"  rho_{j} = 1/2 + i*{mp.nstr(mp.im(mp.zetazero(j)),16):>20} "
          f"  psi(0) = -zeta(rho) = {mp.nstr(-mp.zeta(r), 6)}"
          f"   E = i(2z-1) = {mp.nstr(mp.im(1j*(2*r-1)) + mp.re(1j*(2*r-1)), 12)}")
print("  so the BBM eigenvalue attached to the zero 1/2 + i*gamma is E = -2*gamma.")

# ---- (5) the eta-norm of the eigenstates ---------------------------------
print(L.rule("(5) ||psi_z||_eta^2 = int_a^b |x^{-z}|^2 dx = log(b/a)"))
print(f"{'[a,b]':>22} {'numeric int |Delta psi_z|^2':>30} {'log(b/a)':>14} {'|diff|':>12}")
for a, b in ((mp.mpf('1e-3'), mp.mpf(1)), (mp.mpf(1), mp.mpf(10)),
             (mp.mpf(1), mp.mpf('1e3')), (mp.mpf('1e-6'), mp.mpf('1e6'))):
    val = mp.quad(lambda t: abs(-mp.zeta(rho, t + 1) + mp.zeta(rho, t)) ** 2, [a, b])
    print(f"{('[%s, %s]' % (mp.nstr(a,3), mp.nstr(b,3))):>22} {mp.nstr(val,12):>30} "
          f"{mp.nstr(mp.log(b/a),12):>14} {mp.nstr(abs(val-mp.log(b/a)),3):>12}")

print(L.rule("verdict"))
print("""In the dilation representation H is multiplication by the dilation variable:
spectrum R, simple, purely absolutely continuous.  The BBM eigenfunctions
psi_z = -zeta(z, .+1) are genuine solutions of the eigenvalue equation and the
boundary condition psi_z(0) = 0 really is zeta(z) = 0 - but

        ||psi_z||_eta^2 = int_0^inf x^{-2 Re z} dx = int_0^inf dx/x = infinity

for every z on the critical line, confirmed above against log(b/a) to 12
digits.  They are the delta-normalised generalised eigenfunctions of an
absolutely continuous spectrum, not eigenvectors.  The divergence is only
logarithmic, and Re z = 1/2 is exactly the borderline exponent: that is why
the construction is so seductive and why it fails.""")
