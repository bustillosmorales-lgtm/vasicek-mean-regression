"""Step 4 - compare the candidate spectra with the first Riemann zeros.

Step 3 produced exactly two spectra that are not trivially continuous:

  (M1) the free extension in a Berry-Keating box, k in [1, K].  Perfectly
       arithmetic ladder, spacing 4 pi / log K in the E = 2 gamma variable.
       Tuning K = T / 2 pi reproduces the mean density of the zeros at height T.

  (M2) the decoupled extension of the BBM metric above an infrared cutoff.
       Genuinely discrete, with N(E) = (E/4pi)[log(E/4pi) - 1 + gamma] + o(E),
       i.e. the correct E log E Weyl law.

Both reproduce a counting function.  This script asks the question that
counting functions cannot answer: are these the zeros?

Three tests, in increasing severity:
  1. the counting function N(E);
  2. the individual levels, side by side with the zeros;
  3. the nearest-neighbour spacing distribution of the unfolded spectra, which
     is the standard discriminator: Poisson (uncorrelated), GUE (the zeros),
     or a picket fence (rigid ladder).
"""

import sys, os, json
import numpy as np
import mpmath as mp
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lib_bbm as L

HERE = os.path.dirname(os.path.abspath(__file__))
NZ = 500
print(__doc__)

# ---- the zeros -----------------------------------------------------------
cache = os.path.join(HERE, "zeta_zeros.json")
if os.path.exists(cache):
    gam = np.array(json.load(open(cache)))
else:
    mp.mp.dps = 25
    gam = np.array([float(mp.im(mp.zetazero(n))) for n in range(1, NZ + 1)])
    json.dump(gam.tolist(), open(cache, "w"))
gam = gam[:NZ]
print(f"first {NZ} nontrivial zeros, gamma_1 = {gam[0]:.9f} ... gamma_{NZ} = {gam[-1]:.6f}")


def N_smooth(g):
    """Riemann-von Mangoldt smooth counting of zeros with 0 < gamma' <= g."""
    return (g / (2 * np.pi)) * np.log(g / (2 * np.pi)) - g / (2 * np.pi) + 7.0 / 8.0


# ---- test 1: counting functions -----------------------------------------
print(L.rule("1. counting functions, in the zero variable gamma = E/2"))
T = gam[-1]
K = T / (2 * np.pi)                       # Berry-Keating box tuned at height T
print(f"Berry-Keating box tuned at T = {T:.4f}:  K = T/2pi = {K:.4f}, log K = {np.log(K):.6f}")
print(f"ladder spacing in gamma: 2 pi / log K = {2*np.pi/np.log(K):.6f}")
print(f"mean zero spacing near T:            {2*np.pi/np.log(T/(2*np.pi)):.6f}   (identical by construction)\n")

print(f"{'gamma':>10} {'N_true':>8} {'N_smooth':>10} {'M1 ladder':>10} {'M2 decoupled':>13}")


def N_dec_gamma(g):
    """Counting of the decoupled spectrum, expressed in gamma = E/2."""
    E = 2.0 * g
    n = np.arange(1, max(10, int(E)) + 10)
    return np.floor(E * np.log1p(1.0 / n) / (4 * np.pi) + 1e-12).clip(0).sum()


for g in (50.0, 100.0, 200.0, 300.0, T):
    ntrue = int(np.sum(gam <= g))
    print(f"{g:>10.2f} {ntrue:>8d} {N_smooth(g):>10.2f} "
          f"{g*np.log(K)/(2*np.pi):>10.2f} {N_dec_gamma(g):>13.0f}")
print("""
M1 matches by construction - the box was tuned to make it match.  M2 matches to
leading order with an excess density gamma/2pi per unit height.  Neither is
evidence of anything: a counting function is one number per energy, and both
models were built to have a logarithmic density.""")

# ---- test 2: the individual levels ---------------------------------------
print(L.rule("2. the levels themselves"))
lad_all = 2 * np.pi * np.arange(1, 4000) / np.log(K)
lad = lad_all[:12]
dec = []
for nn in range(1, 4000):
    s_ = 4 * np.pi / np.log1p(1.0 / nn)
    if s_ <= 2 * gam[11]:
        dec.extend(s_ * np.arange(1, int(2 * gam[11] / s_) + 1))
dec = np.sort(np.array(dec))[:12] / 2.0
print(f"{'n':>3} {'zero gamma_n':>15} {'M1 ladder':>13} {'rel err':>9} {'M2 decoupled':>15} {'rel err':>9}")
for i in range(12):
    print(f"{i+1:>3d} {gam[i]:>15.6f} {lad[i]:>13.6f} {abs(lad[i]-gam[i])/gam[i]:>9.1%} "
          f"{dec[i]:>15.6f} {abs(dec[i]-gam[i])/gam[i]:>9.1%}")
print("""
At the bottom of the spectrum the ladder is hopeless, and for a structural
reason: a fixed box has CONSTANT density, while the density of zeros grows like
log(gamma).  A single box can only be right at one height.  So look at the
height it was tuned to.\n""")

j0 = int(np.searchsorted(lad_all, gam[-12]))
print(f"{'zero gamma_n':>15} {'M1 ladder near T':>18} {'rel err':>9} "
      f"{'zero spacing':>14} {'ladder spacing':>16}")
for i in range(12):
    zn, ln_ = gam[-12 + i], lad_all[j0 + i]
    ds_z = gam[-11 + i] - zn if i < 11 else np.nan
    ds_l = lad_all[j0 + i + 1] - ln_
    print(f"{zn:>15.6f} {ln_:>18.6f} {abs(ln_-zn)/zn:>9.2%} "
          f"{ds_z:>14.6f} {ds_l:>16.6f}")
print("""
Tuned at its own height the ladder is within a percent or two in position -
that is the density agreeing, nothing more.  The spacings give it away: the
ladder's are all identical to the last digit, the zeros' vary by a factor of
several.  M2 is off by tens of percent and its levels depend on extension
phases that were set to zero by fiat.""")

# ---- test 3: spacing statistics ------------------------------------------
print(L.rule("3. nearest-neighbour spacings of the unfolded spectra"))


def unfold_spacings(levels, counting):
    x = np.array([counting(v) for v in levels])
    s = np.diff(x)
    return s[s > 0]


s_zero = unfold_spacings(gam, N_smooth)
s_lad = np.full(400, 1.0)                                  # a ladder unfolds to exactly 1

Emax = 4.0e4
dec_lev = []
for nn in range(1, int(Emax) + 2):
    s_ = 4 * np.pi / np.log1p(1.0 / nn)
    if s_ <= Emax:
        dec_lev.extend(s_ * np.arange(1, int(Emax / s_) + 1))
dec_lev = np.sort(np.array(dec_lev))
dec_lev = dec_lev[dec_lev > 0.2 * Emax]                    # use the asymptotic regime
def unfold_local(levels, half=200):
    """Unfold by the local mean density, estimated on a centred window."""
    out = []
    for i in range(half, levels.size - half - 1):
        rho = (2.0 * half) / (levels[i + half] - levels[i - half])
        out.append((levels[i + 1] - levels[i]) * rho)
    return np.array(out)


s_dec = unfold_local(dec_lev)
s_dec = s_dec[s_dec > 0]

# reference laws
def cdf_poisson(s):  return 1.0 - np.exp(-s)
def cdf_gue(s):
    ss = np.atleast_1d(s).astype(float)
    grid = np.linspace(0, 6, 60001)
    pdf = (32 / np.pi ** 2) * grid ** 2 * np.exp(-4 * grid ** 2 / np.pi)
    cdf = np.concatenate([[0.0], np.cumsum((pdf[1:] + pdf[:-1]) / 2 * np.diff(grid))])
    return np.interp(ss, grid, cdf / cdf[-1])

print(f"{'spectrum':>26} {'#s':>7} {'mean':>7} {'var':>7} {'P(s<0.3)':>9} "
      f"{'KS vs GUE':>21} {'KS vs Poisson':>21}")
for label, sv in (("Riemann zeros", s_zero),
                  ("M1 Berry-Keating ladder", s_lad),
                  ("M2 decoupled extension", s_dec)):
    sv = sv / sv.mean()
    kg, pg = stats.kstest(sv, cdf_gue)
    kp, pp = stats.kstest(sv, cdf_poisson)
    print(f"{label:>26} {sv.size:>7d} {sv.mean():>7.4f} {sv.var():>7.4f} "
          f"{np.mean(sv < 0.3):>9.4f} {kg:>10.4f} (p={pg:>7.1e}) "
          f"{kp:>10.4f} (p={pp:>7.1e})")
print(f"{'reference: GUE':>26} {'-':>7} {1.0:>7.4f} {0.1800:>7.4f} {0.0134:>9.4f}")
print(f"{'reference: Poisson':>26} {'-':>7} {1.0:>7.4f} {1.0000:>7.4f} {0.2592:>9.4f}")
print(f"{'reference: picket fence':>26} {'-':>7} {1.0:>7.4f} {0.0000:>7.4f} {0.0:>9.4f}")
print("\nP(s < 0.3) is the level-repulsion probe: GUE suppresses close pairs")
print("quadratically, Poisson does not suppress them at all.")

print(L.rule("STEP 4 VERDICT"))
print("""The zeros show the GUE level repulsion they are known for: spacing variance
near 0.18 and small s suppressed.  The two candidate spectra sit at the two
opposite extremes of the classification:

  * the Berry-Keating ladder is a picket fence, variance 0 - maximally rigid,
    no fluctuation at all, so it cannot be a zero sequence;
  * the decoupled extension is a superposition of many uncorrelated arithmetic
    ladders and shows no level repulsion at all: variance above 1 and close
    pairs MORE frequent than Poisson, because ladders with commensurable
    spacings pile levels on top of each other.  It overshoots the randomness
    the zeros do not have in the first place.

Matching a counting function is cheap, and both models do it.  Neither carries
the correlations.  Reproducing N(E) means reproducing one smooth function;
reproducing the zeros means reproducing a spectrum whose fluctuations are
rigid in a very specific way, and nothing in the metric-completion construction
supplies them - because, as step 3 showed, the metric is a gauge factor that
drops out of the eigenvalue equation entirely.""")
