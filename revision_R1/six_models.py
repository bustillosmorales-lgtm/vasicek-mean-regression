"""Six unit regression models with logit(E[Y_i]) = x_i'beta and one constant
dispersion parameter, for an arbitrary design matrix.  The densities, the mean
inversions and the optimizer are those of verification_model_comparison/fit_all.py
(the code behind the table of six models in the article), generalized from four
regression coefficients to any number k of them."""
import sys
import numpy as np
from scipy import stats, optimize, special

import os
# fit_all.py of the application (densities, mean inversions, optimizer)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'oecd'))
import fit_all as F

expit, lg, bisect = F.expit, F.lg, F.bisect


def ll_beta(p, y, X):
    mu = np.clip(expit(X @ p[:-1]), 1e-12, 1 - 1e-12)
    phi = np.exp(p[-1])
    return (special.gammaln(phi) - special.gammaln(mu * phi) - special.gammaln((1 - mu) * phi)
            + (mu * phi - 1) * lg(y) + ((1 - mu) * phi - 1) * lg(1 - y))


def ll_vasicek(p, y, X):
    al = np.clip(expit(X @ p[:-1]), 1e-12, 1 - 1e-12)
    th = expit(p[-1])
    q = stats.norm.ppf(y)
    a = stats.norm.ppf(al)
    return (0.5 * np.log(1 - th) - 0.5 * np.log(th) + 0.5 * q ** 2
            - (np.sqrt(1 - th) * q - a) ** 2 / (2 * th))


def ll_simplex(p, y, X):
    mu = np.clip(expit(X @ p[:-1]), 1e-9, 1 - 1e-9)
    s2 = np.exp(p[-1])
    d = (y - mu) ** 2 / (y * (1 - y) * mu ** 2 * (1 - mu) ** 2)
    return -0.5 * np.log(2 * np.pi * s2) - 1.5 * lg(y * (1 - y)) - d / (2 * s2)


def ll_logitnormal(p, y, X):
    s = np.exp(p[-1])
    mu = np.clip(expit(X @ p[:-1]), 1e-9, 1 - 1e-9)
    m = bisect(lambda v: F.logitnormal_mean(v, s), -60.0, 60.0, mu)
    return stats.norm.logpdf(lg(y / (1 - y)), m, s) - lg(y * (1 - y))


def ll_kumaraswamy(p, y, X):
    mu = np.clip(expit(X @ p[:-1]), 1e-9, 1 - 1e-9)
    b = np.exp(p[-1])
    la = bisect(lambda v: F.kumaraswamy_mean(np.exp(v), b), -20.0, 20.0, mu)
    return F._kuma_ll(np.exp(la), b, y)


def ll_unitweibull(p, y, X):
    mu = np.clip(expit(X @ p[:-1]), 1e-9, 1 - 1e-9)
    b = np.exp(p[-1])
    la = bisect(lambda v: F.unitweibull_mean(np.exp(v), b), -20.0, 20.0, mu)
    return F._uw_ll(np.exp(la), b, y)


MODELS = {"Beta": ll_beta, "Vasicek": ll_vasicek, "Simplex": ll_simplex,
          "Logit-normal": ll_logitnormal, "Kumaraswamy": ll_kumaraswamy,
          "Unit-Weibull": ll_unitweibull}


def fit(loglik, y, X, p0, tries=3):
    best = None
    for j in range(tries):
        start = p0 if j == 0 else p0 + 0.5 * np.random.default_rng(j).standard_normal(len(p0))
        nll = lambda p: -np.sum(loglik(p, y, X))
        r = optimize.minimize(nll, start, method="BFGS", options=dict(gtol=1e-8, maxiter=5000))
        r2 = optimize.minimize(nll, r.x, method="Nelder-Mead",
                               options=dict(maxiter=40000, maxfev=40000, xatol=1e-10, fatol=1e-12))
        r = r2 if r2.fun < r.fun else r
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun


def compare(y, X):
    n, k = X.shape
    p0 = np.append(np.linalg.lstsq(X, np.log(y / (1 - y)), rcond=None)[0], 0.0)
    fits = {nm: fit(f, y, X, p0.copy()) for nm, f in MODELS.items()}
    aic = {nm: -2 * fits[nm][1] + 2 * (k + 1) for nm in MODELS}
    best = min(aic.values())
    w = {nm: np.exp(-(aic[nm] - best) / 2) for nm in MODELS}
    S = sum(w.values())
    lv = ll_vasicek(fits["Vasicek"][0], y, X)
    vu = {}
    for nm in MODELS:
        if nm == "Vasicek":
            continue
        d = lv - MODELS[nm](fits[nm][0], y, X)
        z = np.sqrt(n) * d.mean() / d.std(ddof=1)
        vu[nm] = (float(z), float(2 * stats.norm.sf(abs(z))))
    rank = sorted(MODELS, key=lambda m: aic[m])
    return dict(n=n, k=k + 1, ll={m: float(fits[m][1]) for m in MODELS},
                aic={m: float(aic[m]) for m in MODELS},
                weight={m: float(w[m] / S) for m in MODELS}, rank=rank, vuong=vu,
                theta_vas=float(expit(fits["Vasicek"][0][-1])))
