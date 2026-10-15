"""Part I residuals are pivotal: r^Q_i = (z_i - zbar)/sigma_ML with z iid N(0,1),
whatever alpha and theta.  So the across-replicate SD of each moment summary is a
function of n alone and can be simulated directly.  Moments follow SAS PROC MEANS
defaults (VARDEF=DF, adjusted G1 and G2)."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(20261013)
R = 200000
M = 5000


def g1g2(x):
    n = x.shape[1]
    m = x.mean(1, keepdims=True)
    d = x - m
    s2 = (d ** 2).sum(1) / (n - 1)
    m2 = (d ** 2).mean(1); m3 = (d ** 3).mean(1); m4 = (d ** 4).mean(1)
    g1 = m3 / m2 ** 1.5
    G1 = g1 * np.sqrt(n * (n - 1)) / (n - 2)
    g2 = m4 / m2 ** 2 - 3
    G2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * g2 + 6)
    return m[:, 0], s2, G1, G2


for n in (30, 50, 100, 150, 200):
    z = rng.standard_normal((R, n))
    sig = z.std(1, ddof=0, keepdims=True)
    rq = (z - z.mean(1, keepdims=True)) / sig
    rcs = -stats.norm.logsf(rq)
    out = []
    for nm, r in (('Q', rq), ('CS', rcs)):
        mu, s2, G1, G2 = g1g2(r)
        out.append('%s mean %.4f/%.5f var %.4f/%.5f skew %.4f/%.5f EK %.4f/%.5f' % (
            nm, mu.mean(), mu.std() / np.sqrt(M), s2.mean(), s2.std() / np.sqrt(M),
            G1.mean(), G1.std() / np.sqrt(M), G2.mean(), G2.std() / np.sqrt(M)))
    print(n, ' | '.join(out))
