"""Step 5 - the rigidity theorem: no choice of extension phases works.

Step 3 left one loophole open.  The decoupled self-adjoint extension of the BBM
operator is genuinely discrete and carries the correct E log E Weyl law, but
each interval I_n = (2 pi n, 2 pi (n+1)) carries a free extension phase
theta_n in [0, 2 pi), so its levels looked like free parameters.  With
infinitely many of them one might hope to fit the zeros.

This script closes that loophole.  In the zero variable gamma = E/2 the ladder
of interval n is

        gamma^(n)_m = (2 pi m + theta_n) / L_n ,   L_n = log(1 + 1/n),

an arithmetic progression of spacing s_n = 2 pi / L_n.  Two obstructions:

  I.  a DENSITY obstruction, uniform in theta.  Counting levels in (0, T] with
      c = T / 2 pi,
          theta_n = 0        -> floor(c L_n)
          0 < theta_n < 2 pi -> floor(c L_n - theta_n/2pi) + 1 >= floor(c L_n)
      so floor(c L_n) is a lower bound for EVERY theta_n, and

          N_theta(T) >= sum_n floor(c L_n) = D(c) - gamma c + o(c)
                      = N_Riemann(T) + gamma c + o(c),

      where D is the Dirichlet divisor summatory function.  The model is forced
      to carry at least gamma c excess levels below every height, whatever the
      phases.  The arithmetic that comes out of this construction is the
      divisor function, not the zeros.

  II. a RIGIDITY obstruction.  Each ladder is an arithmetic progression and the
      zeros are not, so most forced levels cannot be placed on zeros even with
      the best phase.  Measured below by direct optimisation.
"""

import sys, os, json
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_bbm as L

HERE = os.path.dirname(os.path.abspath(__file__))
G = np.euler_gamma
print(__doc__)

gam = np.array(json.load(open(os.path.join(HERE, "zeta_zeros.json"))))
print(f"using {gam.size} zeros, gamma_1 = {gam[0]:.6f} ... gamma_{gam.size} = {gam[-1]:.6f}")


def N_riemann(T):
    c = T / (2 * np.pi)
    return c * np.log(c) - c + 7.0 / 8.0


def forced_count(c, nmax=None):
    """sum_n floor(c L_n): the theta-uniform lower bound on the level count."""
    n = np.arange(1, (nmax or int(4 * c) + 10) + 1)
    return np.floor(c * np.log1p(1.0 / n)).sum()


def divisor_sum(c):
    n = np.arange(1, int(c) + 1)
    return np.floor(c / n).sum()


# ---- I. the density obstruction -----------------------------------------
print(L.rule("I. the theta-uniform lower bound, checked against random phases"))
rng = np.random.default_rng(0)
print(f"{'c = T/2pi':>10} {'bound sum floor(c L_n)':>24} {'min over 500 random theta':>27} {'holds':>7}")
for c in (20.0, 137.0, 500.0):
    n = np.arange(1, int(6 * c) + 10)
    Ln = np.log1p(1.0 / n)
    bound = np.floor(c * Ln).sum()
    worst = min(np.clip(np.floor(c * Ln - rng.uniform(0, 1, n.size)) + 1, 0, None).sum()
                for _ in range(500))
    print(f"{c:>10.1f} {bound:>24.0f} {worst:>27.0f} {str(worst >= bound):>7}")

print(L.rule("I'. the bound IS the divisor summatory function, shifted by gamma"))
print(f"{'c':>9} {'sum floor(c L_n)':>18} {'D(c)':>13} {'D(c) - gamma c':>16} "
      f"{'N_Riemann':>13} {'excess / c':>12}")
for c in (1e2, 1e3, 1e4, 1e5, 1e6):
    S, Dc, NR = forced_count(c), divisor_sum(c), N_riemann(2 * np.pi * c)
    print(f"{c:>9.0e} {S:>18.0f} {Dc:>13.0f} {Dc - G*c:>16.1f} {NR:>13.1f} {(S-NR)/c:>12.6f}")
print(f"\n   excess / c -> gamma = {G:.6f}.  Analytically: sum_n [1/n - log(1+1/n)] = gamma,")
print("   and sum_{n<=c} floor(c/n) = D(c) = c log c + (2 gamma - 1) c + O(sqrt c), so")
print("   sum_n floor(c L_n) = c log c + (gamma - 1) c + o(c) = N_Riemann + gamma c + o(c).")

# ---- II. the rigidity obstruction ---------------------------------------
print(L.rule("II. best possible phases: how many levels cannot sit on a zero"))
print("""For each ladder the phase is optimised independently (the ladders are
decoupled, so the joint optimum is the product of the per-ladder optima).  A
level counts as matched if it lies within eps of a zero.  Ladders with spacing
larger than T contribute at most one level and can always dodge, so they are
free; the cost comes from the ladders that are forced to fire repeatedly.\n""")


def best_phases(T, zeros, eps, ngrid=720):
    """Minimum number of spurious levels in (0, T], optimising each phase."""
    c = T / (2 * np.pi)
    zs = zeros[zeros <= T]
    spurious = matched = count = 0
    n = 1
    while True:
        Ln = np.log1p(1.0 / n)
        if c * Ln < 1.0:                      # at most one level: cost-free
            break
        s = 2 * np.pi / Ln
        phi = np.linspace(0.0, s, ngrid, endpoint=False)
        M = int(np.floor(c * Ln)) + 2
        lev = phi[:, None] + s * np.arange(M)[None, :]
        live = (lev > 0) & (lev <= T)
        idx = np.clip(np.searchsorted(zs, lev), 1, zs.size - 1)
        near = np.minimum(np.abs(lev - zs[idx]), np.abs(lev - zs[idx - 1]))
        hit = live & (near <= eps)
        sp = live.sum(1) - hit.sum(1)
        j = int(np.argmin(sp * 1000 - hit.sum(1)))   # fewest spurious, then most hits
        spurious += int(sp[j]); matched += int(hit[j].sum()); count += int(live[j].sum())
        n += 1
    return spurious, matched, count, n - 1


print(f"{'T':>9} {'zeros<=T':>9} {'eps':>6} {'min levels':>11} {'matched':>8} "
      f"{'spurious':>9} {'bound gamma*c':>14} {'spurious/T':>11}")
for k in (50, 100, 200, 500):
    T = gam[k - 1]
    c = T / (2 * np.pi)
    for eps in (0.02, 0.10):
        sp, mt, ct, nl = best_phases(T, gam, eps)
        print(f"{T:>9.2f} {k:>9d} {eps:>6.2f} {ct:>11d} {mt:>8d} {sp:>9d} "
              f"{G*c:>14.1f} {sp/T:>11.4f}")

print("""
The number of levels that cannot be placed on any zero grows linearly in T,
tracking the gamma c bound, and it does so under the BEST possible choice of
every phase.  Loosening eps by a factor of five barely helps: the ladders are
rigid, so a ladder that fires k times in (0, T] can align with at most a couple
of zeros no matter where it starts.""")

print(L.rule("STEP 5 VERDICT"))
print(f"""Theorem (density obstruction).  For every choice of extension phases
{{theta_n}} the decoupled self-adjoint extension satisfies

    N_theta(T)  >=  sum_{{n>=1}} floor(T L_n / 2 pi)
                 =  N_Riemann(T) + (gamma / 2 pi) T + o(T),

so it carries at least (gamma/2pi) T ~ {G/(2*np.pi):.4f} T levels below height T that
are not zeros.  No choice of phases can make this spectrum the zero set.  The
excess is exactly Euler-Mascheroni, and the counting function is the Dirichlet
divisor summatory function D(T/2pi) in disguise: what this construction knows
about is divisors, not zeros.

The rigidity obstruction is independent and stronger in practice: arithmetic
progressions cannot track a sequence with GUE fluctuations, so even the forced
levels mostly miss.

SCOPE, stated honestly.  This covers the DECOUPLED family - all self-adjoint
extensions that do not couple the intervals, which is the U(1) family of
quasi-periodic conditions, one phase per interval.  Extensions that COUPLE the
intervals are a strictly larger family (the quantum-graph setting, where the
Berry-Keating operator's extensions have been completely classified together
with their Weyl asymptotics).  The bound above does not apply to them, and
whether an analogous obstruction holds there is open.  That is the honest
frontier of this line, and it is where a real theorem would have to live.""")
