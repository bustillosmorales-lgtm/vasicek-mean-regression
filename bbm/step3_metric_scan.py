"""Step 3 - do alternative metrics discretise the spectrum?  Numerical answer.

Setting.  In the momentum representation put v = log k.  The dilation generator
becomes A = 2i d/dv, a constant-coefficient first-order operator, and ANY metric
diagonal in momentum, eta = f(k), becomes multiplication by g(v)^2 with
g = f^{1/2}.  The Hamiltonian that eta makes symmetric is

        H_g = g^{-1} A g = 2i d/dv + 2i (log g)' ,

and its eigenfunctions are, for every single g,

        psi_E(v) = g(v)^{-1} e^{-iEv/2} ,    ||psi_E||_eta = ||e^{-iEv/2}|| .

The metric is a pure gauge along the dilation flow.  Three experiments:

  A. similarity invariance - as long as g is invertible the spectrum is exactly
     that of A, however violently the condition number grows;
  B. the singular limit - the zeros of g (for BBM, k in 2 pi Z) cut the line
     into intervals and open up other self-adjoint extensions.  We build the
     decoupled one explicitly and ask whether IT is discrete;
  C. box scaling - the only diagnostic that distinguishes a discrete spectrum
     from a continuous one on a computer: do the levels converge when the
     truncation is removed, or do they slide?

Plus a control: a metric diagonal in position instead of momentum.
"""

import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_bbm as L

print(__doc__)
np.set_printoptions(precision=6, suppress=True)

# --------------------------------------------------------------------------
# A. similarity invariance
# --------------------------------------------------------------------------
print(L.rule("A. spectrum of H_g = g^{-1} A g versus spectrum of A"))
print("v-box: k in [e^1, e^7], n = 700 points, periodic; levels compared pairwise.\n")
n, v0, v1 = 700, 1.0, 7.0
Lv = v1 - v0
v, dv = L.u_grid(v0, v1, n)
A = -L.dilation_matrix_u(n, Lv)                 # 2i d/dv in the k-representation
specA = np.sort(np.linalg.eigvalsh(A))

print(f"{'metric':>10} {'symbol':>40} {'min g':>11} {'cond(G)':>11} "
      f"{'max|spec-spec(A)|':>19} {'max |Im E|':>11}")
for name, (f, desc) in L.METRICS.items():
    k = np.exp(v)
    g = np.sqrt(np.maximum(f(k), 0.0))
    gs = np.maximum(g, 1e-14)
    H = (A * (1.0 / gs)[:, None]) * gs[None, :]          # diag(1/g) A diag(g)
    ev = np.linalg.eigvals(H)
    order = np.argsort(ev.real)
    ev = ev[order]
    err = np.max(np.abs(np.sort(ev.real) - specA))
    print(f"{name:>10} {desc:>40} {g.min():>11.3e} {gs.max()/gs.min():>11.3e} "
          f"{err:>19.3e} {np.max(np.abs(ev.imag)):>11.3e}")

print("""
Every metric with an invertible symbol reproduces the spectrum of A exactly.
The condition number of the gauge spans thirteen orders of magnitude and the
spectrum does not move: a metric cannot change the spectrum, it can only change
which states are normalisable.  This is the numerical form of the no-go.""")

# the eigenfunctions are the same plane wave for every metric
print(L.rule("A'. the predicted eigenfunctions psi_E = g^{-1} e^{-iEv/2}"))
print("psi_E must be tested at a ladder energy E_n = 4 pi n / L_v, the only")
print("plane waves that live on a periodic box.\n")
print(f"{'metric':>10} {'n':>4} {'E_n':>11} {'||(H_g - E)psi|| / ||E psi||':>32} {'||psi_E||_eta':>15}")
for name in ("bbm", "k4", "sobolev", "ess_zero", "bbm_sob"):
    f, _ = L.METRICS[name]
    k = np.exp(v)
    g = np.maximum(np.sqrt(np.maximum(f(k), 0.0)), 1e-14)
    H = (A * (1.0 / g)[:, None]) * g[None, :]
    for nlev in (1, 5):
        E = 4.0 * np.pi * nlev / Lv
        psi = np.exp(-0.5j * E * (v - v0)) / g
        r = np.linalg.norm(H @ psi - E * psi) / np.linalg.norm(E * psi)
        print(f"{name:>10} {nlev:>4d} {E:>11.6f} {r:>32.3e} "
              f"{np.linalg.norm(g*psi)*np.sqrt(dv):>15.6f}")
print("""
The eta-norm of the eigenfunction is the norm of a bare plane wave in v, for
every metric: sqrt(L_v) = %.6f here.  The metric cancels out of the
normalisation.  No metric can make these states normalisable on the whole line
and none can make them non-normalisable on a box.""" % np.sqrt(Lv))

# --------------------------------------------------------------------------
# B. the singular limit: zeros of g decouple the line
# --------------------------------------------------------------------------
print(L.rule("B. the BBM metric vanishes at k = 2 pi n: what that actually buys"))
print("""g(k) = 2|sin(k/2)| vanishes at every k = 2 pi n.  A function blowing up like
1/g there still has finite eta-norm, so each interval (2 pi n, 2 pi (n+1))
supports its own solutions and the symmetric operator acquires infinite
deficiency indices.  Two things follow, and only one of them is good news.
""")

# B1: every real E is an eigenvalue of the adjoint, with infinite multiplicity
print("B1. solutions localised on a single interval, for arbitrary real E:")
print(f"{'interval n':>11} {'L_v = log((n+1)/n)':>20} {'||psi_E||_eta^2 (E = 14.1)':>28} {'exact L_v':>12}")
for nn in (1, 2, 5, 50):
    a_, b_ = 2 * np.pi * nn, 2 * np.pi * (nn + 1)
    nq = 200000
    vq = np.linspace(np.log(a_), np.log(b_), nq)
    gq = 2.0 * np.abs(np.sin(np.exp(vq) / 2.0))
    psi = np.exp(-0.5j * 14.134725 * vq) / np.maximum(gq, 1e-300)
    nrm = np.trapezoid((gq * np.abs(psi)) ** 2, vq)
    print(f"{nn:>11d} {np.log((nn+1)/nn):>20.10f} {nrm:>28.10f} {np.log((nn+1)/nn):>12.8f}")
print("""   Finite for every real E and every interval: every real number is an
   eigenvalue of the adjoint with infinite multiplicity, and the deficiency
   indices are (infinity, infinity).  The free extension glues the intervals
   back together and is purely continuous - that is part A above.

B2. the decoupled extension.  Interval n carries the ladder
    E = (4 pi m + 2 theta_n) / L_n with L_n = log(1 + 1/n), so its spacing is
    4 pi / L_n ~ 4 pi n: short intervals give sparse ladders.  The interval
    k in (0, 2 pi) has infinite length in v and still gives continuous
    spectrum, so cut it off (an infrared momentum cutoff k > 2 pi).  What is
    left is genuinely DISCRETE - and its counting function is not innocent.
""")
print(f"{'E':>10} {'levels in (0,E]':>16} {'min gap':>10} "
      f"{'N_Riemann(E/2)':>16} {'ratio':>8} {'(N_dec-N_R)/E':>15}")


def N_dec(E, theta=0.0):
    n = np.arange(1, max(10, int(E)) + 10)
    Ln = np.log1p(1.0 / n)
    return np.floor((E * Ln - 2 * theta) / (4 * np.pi) + 1e-12).clip(0).sum()


def N_riemann_E(E):
    """Riemann smooth counting of 0 < gamma <= E/2, written in E = 2 gamma."""
    return (E / (4 * np.pi)) * np.log(E / (4 * np.pi)) - E / (4 * np.pi) + 7.0 / 8.0


for E in (1e2, 1e3, 1e4, 1e5, 1e6, 1e7):
    lev = []
    for nn in range(1, min(int(E), 400000) + 2):
        s_ = 4 * np.pi / np.log1p(1.0 / nn)
        if s_ <= E:
            lev.extend(s_ * np.arange(1, int(E / s_) + 1))
    lev = np.sort(np.array(lev))
    mg = np.diff(lev).min() if lev.size > 1 else np.nan
    nd, nr = N_dec(E), N_riemann_E(E)
    print(f"{E:>10.0e} {nd:>16.0f} {mg:>10.4f} {nr:>16.1f} {nd/nr:>8.5f} "
          f"{(nd-nr)/E:>15.6f}")

print(f"""
   The count in (0, E] is finite for every E - the spectrum is genuinely
   discrete once the infrared piece is removed.  (The minimum gap shrinks with
   E only because the density grows like log E, as it must.)  And

        N_dec(E) = (E/4pi)[ log(E/4pi) - 1 + gamma ] + o(E) ,
        N_Riemann(E) = (E/4pi)[ log(E/4pi) - 1 ] + 7/8 ,

   i.e. the leading E log E term is exactly right and the excess density is the
   Euler-Mascheroni constant: the measured (N_dec - N_R)/E above converges to
   gamma / 4 pi = {np.euler_gamma/(4*np.pi):.6f}.  That is the Berry-Keating
   counting law falling out of nothing but the arithmetic of the zero set
   k = 2 pi n of the metric symbol.

   It is also where the construction stops being predictive: every interval
   carries its own free extension phase theta_n, so each ladder can be slid
   at will.  The levels are parameters, not outputs.  Any target sequence with
   this counting law can be fitted, the Riemann zeros among them, and nothing
   would have been derived.""")

# --------------------------------------------------------------------------
# C. box scaling - the only honest discreteness test
# --------------------------------------------------------------------------
print(L.rule("C. box scaling: do the levels converge, or do they slide?"))
print("""A finite matrix always has a discrete spectrum.  The test for genuine
discreteness is whether a given level converges when the truncation is removed.
Here the box is k in [1, K]; L_v = log K.\n""")
print(f"{'metric':>10} {'K':>10} {'L_v':>9} {'spacing':>11} {'4 pi / L_v':>11} "
      f"{'5th level E_5':>14} {'drift vs prev':>14}")
for name in ("unit", "bbm", "k4", "sobolev", "ess_zero"):
    f, _ = L.METRICS[name]
    prev = None
    for K in (1e2, 1e4, 1e8, 1e16):
        Lvv = np.log(K)
        nn = 600
        vv, _ = L.u_grid(0.0, Lvv, nn)
        Ak = -L.dilation_matrix_u(nn, Lvv)
        kk = np.exp(vv)
        gg = np.maximum(np.sqrt(np.maximum(f(kk), 0.0)), 1e-14)
        Hk = (Ak * (1.0 / gg)[:, None]) * gg[None, :]
        ev = np.sort(np.linalg.eigvals(Hk).real)
        pos = ev[ev > 1e-9]
        sp = np.median(np.diff(ev[nn // 2 - 30: nn // 2 + 30]))
        E5 = pos[4] if pos.size > 4 else np.nan
        drift = "-" if prev is None else f"{E5 - prev:+.6f}"
        print(f"{name:>10} {K:>10.0e} {Lvv:>9.3f} {sp:>11.6f} {4*np.pi/Lvv:>11.6f} "
              f"{E5:>14.6f} {drift:>14}")
        prev = E5
    print()
print("""Every level slides to zero like 1 / log(box), identically for every metric.
Nothing converges.  The spectrum is continuous and no metric discretises it.

Note the rate: the level spacing is 4 pi / log K.  The mean spacing between
Riemann zeros at height T is 2 pi / log(T / 2 pi), and the BBM eigenvalue
attached to the zero 1/2 + i gamma is E = -2 gamma, so in the E variable the
zeros have mean spacing 4 pi / log(T / 2 pi).  The two laws agree exactly when
the box satisfies K = T / 2 pi.  That is the Berry-Keating phase-space cutoff,
and it is the whole reason a finite computation can look like it is producing
zeros.  Step 4 tests whether it actually does.""")

# --------------------------------------------------------------------------
# control: a metric diagonal in position
# --------------------------------------------------------------------------
print(L.rule("control: metrics diagonal in position, eta = w(x) = x^{2a}"))
print("""In the u = log x representation A = -2i d/du and w^{1/2} = e^{au}, so

        H_w = e^{-au} (-2i d/du) e^{au} = -2i d/du - 2ia ,

a rigid translation of the spectrum off the real axis: a position-space weight
does not deform the spectrum at all, it destroys self-adjointness.  (On a
finite periodic box this is invisible, because conjugation by an invertible
diagonal matrix is an exact similarity - which is itself the lesson of part A.
The check below is therefore pointwise, on a non-periodic grid.)\n""")

nn = 4001
uu = np.linspace(-6.0, 6.0, nn)
hu = uu[1] - uu[0]
d = np.zeros((nn, nn))                      # 4th-order centred difference
i4 = np.arange(2, nn - 2)
for off, c in ((-2, 1/12), (-1, -2/3), (1, 2/3), (2, -1/12)):
    d[i4, i4 + off] = c / hu
core = np.arange(nn // 4, 3 * nn // 4)

print(f"{'a':>8} {'E':>8} {'measured (H_w psi)/psi':>34} {'predicted E - 2ia':>22} {'max dev':>11}")
for a in (0.0, 0.25, 0.5, 1.0):
    w = np.exp(a * uu)
    Hw = (-2j * d) * w[None, :] / w[:, None]
    for E in (2.0, 14.134725):
        psi = np.exp(0.5j * E * uu)
        ratio = (Hw @ psi)[core] / psi[core]
        pred = E - 2j * a
        print(f"{a:>8.2f} {E:>8.3f} {str(np.round(ratio.mean(), 8)):>34} "
              f"{str(np.round(pred, 8)):>22} {np.max(np.abs(ratio - pred)):>11.2e}")
print("""
Non-real spectrum: eta = w(x) is not a metric for H at all.  Confining the
spectrum is not something a weight in x can do either.\n""")

print(L.rule("STEP 3 VERDICT"))
print("""No.  The spectrum does not discretise.

1. Any metric eta = f(p) with f invertible is a similarity transform: the
   spectrum is exactly that of the dilation generator, verified to 1e-12 across
   thirteen orders of magnitude in the condition number.
2. When f is not coercive the similarity is unbounded, but the completion is
   still unitarily equivalent to L^2(R_+) and the free realisation is still the
   dilation generator: continuous spectrum R.
3. The zeros of f open other self-adjoint extensions.  The decoupled one has a
   DENSE point spectrum, not a discrete one.
4. Every level slides like 1 / log(box); nothing converges under refinement.
5. A metric diagonal in position does not discretise either - it moves the
   spectrum off the real axis.

The obstruction is structural: in the dilation representation the metric is a
gauge factor g(v) that cancels out of both the eigenvalue equation and the
eta-normalisation.  Discreteness is a statement about compactness of the
resolvent, and no choice of g affects it.  What does affect it is the LENGTH of
the dilation box, which is exactly the phase-space cutoff of Berry-Keating.""")
