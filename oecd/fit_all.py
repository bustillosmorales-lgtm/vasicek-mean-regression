"""
Referee re-analysis of the OECD application in "Vasicek mean regression:
Estimation and diagnostics".

Fits six unit-interval regression models to the same data, with the SAME
structure the manuscript uses: four regression coefficients on the linear
predictor (intercept + 3 covariates, logit link) plus one constant
shape/dispersion parameter.  k = 5 for every model.

Dependencies: numpy, scipy only.
Run:  python fit_all.py

WHAT IS LINKED TO THE LINEAR PREDICTOR
--------------------------------------
The manuscript's Vasicek model is a MEAN model: E(Y_i) = alpha_i and
logit(alpha_i) = x_i'beta.  To keep the comparison like for like, the primary
table below links logit(E(Y_i)) = x_i'beta for ALL six families.  For beta,
simplex and Vasicek the mean is already a natural parameter.  For the
logit-normal, Kumaraswamy and unit-Weibull families the mean has no closed
form in the natural parameters, so the auxiliary shape parameter is recovered
from the target mean by vectorised bisection at every likelihood evaluation.

A secondary table repeats the exercise with logit(median(Y_i)) = x_i'beta for
the three families whose median IS closed form, because that is the
parameterisation in which they are usually published.  Both tables are
reported so the reader can see the ranking is not an artefact of the choice.

COMPARABILITY OF THE LOG-LIKELIHOODS
------------------------------------
Every log-likelihood below is the density of the SAME response Y on (0,1)
with respect to Lebesgue measure.  Nothing is fitted on a transformed scale.
In particular the logit-normal includes its Jacobian term -log(y(1-y)); without
it the value would not be comparable and the model would be flattered.
check_densities() verifies numerically that each FITTED density integrates to
one over (0,1), which catches a missing or wrong Jacobian.
"""

import numpy as np
from scipy import stats, optimize, special, integrate

import os as _os
# la ruta se resuelve contra la carpeta del script, para que tambien
# funcione ejecutandolo desde la raiz del repositorio
CSV = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)),
                    "oecd_bli_2017.csv")

# ----------------------------------------------------------------- helpers
def expit(v):
    return 1.0 / (1.0 + np.exp(-v))

def lg(v):
    return np.log(np.clip(v, 1e-300, None))

def bisect(f, lo, hi, target, iters=200):
    """Vectorised bisection: solve f(p) = target for increasing f."""
    lo = np.full_like(target, lo, dtype=float)
    hi = np.full_like(target, hi, dtype=float)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        go_up = f(mid) < target
        lo = np.where(go_up, mid, lo)
        hi = np.where(go_up, hi, mid)
    return 0.5 * (lo + hi)

# Gauss-Hermite (logit-normal mean) and Gauss-Laguerre (unit-Weibull mean)
_GH_X, _GH_W = np.polynomial.hermite.hermgauss(80)
_GL_X, _GL_W = np.polynomial.laguerre.laggauss(120)

def logitnormal_mean(m, s):
    """E[expit(m + s Z)], Z ~ N(0,1)."""
    m = np.atleast_1d(m)[:, None]
    return (expit(m + s * np.sqrt(2.0) * _GH_X) * _GH_W).sum(1) / np.sqrt(np.pi)

def kumaraswamy_mean(a, b):
    """E[Y] = b * B(1 + 1/a, b) for Kumaraswamy(a, b)."""
    return np.exp(np.log(b) + special.gammaln(1.0 + 1.0 / a) + special.gammaln(b)
                  - special.gammaln(1.0 + 1.0 / a + b))

def unitweibull_mean(a, b):
    """E[Y] = int_0^inf exp(-t) exp(-(t/a)^(1/b)) dt for F(y)=exp(-a(-log y)^b)."""
    a = np.atleast_1d(a)[:, None]
    return (np.exp(-np.power(_GL_X / a, 1.0 / b)) * _GL_W).sum(1)

# --------------------------------------------------- log-densities of Y
# Each returns the pointwise log density of the observed y on (0,1).

def ll_beta(p, y, X):
    mu = np.clip(expit(X @ p[:4]), 1e-12, 1 - 1e-12)
    phi = np.exp(p[4])
    return (special.gammaln(phi) - special.gammaln(mu * phi)
            - special.gammaln((1 - mu) * phi)
            + (mu * phi - 1) * lg(y) + ((1 - mu) * phi - 1) * lg(1 - y))

def ll_vasicek(p, y, X):
    al = np.clip(expit(X @ p[:4]), 1e-12, 1 - 1e-12)
    th = expit(p[4])
    q = stats.norm.ppf(y)
    a = stats.norm.ppf(al)
    return (0.5 * np.log(1 - th) - 0.5 * np.log(th) + 0.5 * q ** 2
            - (np.sqrt(1 - th) * q - a) ** 2 / (2 * th))

def ll_simplex(p, y, X):
    mu = np.clip(expit(X @ p[:4]), 1e-9, 1 - 1e-9)
    s2 = np.exp(p[4])
    d = (y - mu) ** 2 / (y * (1 - y) * mu ** 2 * (1 - mu) ** 2)
    return -0.5 * np.log(2 * np.pi * s2) - 1.5 * lg(y * (1 - y)) - d / (2 * s2)

# ---- logit-normal.  NOTE the Jacobian term -log(y(1-y)); it is required.
def ll_logitnormal_median(p, y, X):
    """logit(median) = x'beta  (equivalently, the mean of logit(Y) is x'beta)."""
    s = np.exp(p[4])
    return stats.norm.logpdf(lg(y / (1 - y)), X @ p[:4], s) - lg(y * (1 - y))

def ll_logitnormal_mean(p, y, X):
    """logit(E[Y]) = x'beta; solve for the normal location m."""
    s = np.exp(p[4])
    mu = np.clip(expit(X @ p[:4]), 1e-9, 1 - 1e-9)
    m = bisect(lambda v: logitnormal_mean(v, s), -60.0, 60.0, mu)
    return stats.norm.logpdf(lg(y / (1 - y)), m, s) - lg(y * (1 - y))

def _kuma_ll(a, b, y):
    return np.log(a) + np.log(b) + (a - 1) * lg(y) + (b - 1) * lg(1 - y ** a)

def ll_kumaraswamy_median(p, y, X):
    m = np.clip(expit(X @ p[:4]), 1e-9, 1 - 1e-9)
    b = np.exp(p[4])
    a = np.log1p(-0.5 ** (1.0 / b)) / lg(m)        # median(Y) = m
    return _kuma_ll(a, b, y)

def ll_kumaraswamy_mean(p, y, X):
    mu = np.clip(expit(X @ p[:4]), 1e-9, 1 - 1e-9)
    b = np.exp(p[4])
    la = bisect(lambda v: kumaraswamy_mean(np.exp(v), b), -20.0, 20.0, mu)
    return _kuma_ll(np.exp(la), b, y)

def _uw_ll(a, b, y):
    u = -lg(y)
    return np.log(a) + np.log(b) - np.log(y) + (b - 1) * np.log(u) - a * u ** b

def ll_unitweibull_median(p, y, X):
    m = np.clip(expit(X @ p[:4]), 1e-9, 1 - 1e-9)
    b = np.exp(p[4])
    a = -np.log(0.5) / (-lg(m)) ** b               # median(Y) = m
    return _uw_ll(a, b, y)

def ll_unitweibull_mean(p, y, X):
    mu = np.clip(expit(X @ p[:4]), 1e-9, 1 - 1e-9)
    b = np.exp(p[4])
    la = bisect(lambda v: unitweibull_mean(np.exp(v), b), -20.0, 20.0, mu)
    return _uw_ll(np.exp(la), b, y)

MEAN_MODELS = {
    "Beta":            ll_beta,
    "Vasicek":         ll_vasicek,
    "Simplex":         ll_simplex,
    "Logit-normal":    ll_logitnormal_mean,
    "Kumaraswamy":     ll_kumaraswamy_mean,
    "Unit-Weibull":    ll_unitweibull_mean,
}
MEDIAN_MODELS = {
    "Logit-normal":    ll_logitnormal_median,
    "Kumaraswamy":     ll_kumaraswamy_median,
    "Unit-Weibull":    ll_unitweibull_median,
}

# ------------------------------------------------------------------- fitting
def fit(loglik, y, X, p0, tries=4):
    best = None
    for j in range(tries):
        start = p0 if j == 0 else p0 * (1 + 0.25 * np.random.default_rng(j).standard_normal(len(p0)))
        r = optimize.minimize(lambda p: -np.sum(loglik(p, y, X)), start,
                              method="Nelder-Mead",
                              options=dict(maxiter=60000, maxfev=60000,
                                           xatol=1e-10, fatol=1e-10))
        if best is None or r.fun < best.fun:
            best = r
    return best.x, -best.fun

def hessian_se(nll, p, eps=1e-5):
    k = len(p)
    H = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            pp = p.copy(); pp[i] += eps; pp[j] += eps; f1 = nll(pp)
            pp = p.copy(); pp[i] += eps; pp[j] -= eps; f2 = nll(pp)
            pp = p.copy(); pp[i] -= eps; pp[j] += eps; f3 = nll(pp)
            pp = p.copy(); pp[i] -= eps; pp[j] -= eps; f4 = nll(pp)
            H[i, j] = (f1 - f2 - f3 + f4) / (4 * eps * eps)
    return np.sqrt(np.diag(np.linalg.inv(H)))

def table(models, fits, n, title):
    print("\n" + "=" * 88)
    print(title)
    print("=" * 88)
    aic = {nm: -2 * fits[nm][1] + 2 * 5 for nm in models}
    best = min(aic.values())
    w = {nm: np.exp(-(aic[nm] - best) / 2) for nm in models}
    S = sum(w.values())
    print(f"{'Model':16s} {'logLik':>10} {'k':>3} {'AIC':>11} {'BIC':>11} "
          f"{'dAIC':>8} {'w_AIC':>8} {'dAIC vs Vas':>12}")
    ref = aic.get("Vasicek", np.nan)
    for nm in sorted(models, key=lambda z: aic[z]):
        ll = fits[nm][1]
        print(f"{nm:16s} {ll:10.4f} {5:3d} {aic[nm]:11.4f} "
              f"{-2*ll + 5*np.log(n):11.4f} {aic[nm]-best:8.4f} "
              f"{w[nm]/S:8.4f} {aic[nm]-ref:12.4f}")
    return aic

def vuong(fits, models, y, X, n):
    print(f"\n{'Vuong test vs the Vasicek model (non-nested, equal k)':<60}")
    lv = ll_vasicek(fits["Vasicek"][0], y, X)
    for nm in models:
        if nm == "Vasicek":
            continue
        d = lv - models[nm](fits[nm][0], y, X)
        if d.std(ddof=1) < 1e-12:
            continue
        z = np.sqrt(n) * d.mean() / d.std(ddof=1)
        p = 2 * (1 - stats.norm.cdf(abs(z)))
        verdict = ("Vasicek better" if z > 1.96 else
                   "OTHER model better" if z < -1.96 else "not distinguishable")
        print(f"   Vasicek vs {nm:16s} z = {z:+8.4f}   p = {p:.4f}   {verdict}")

def check_densities(models, fits, y, X):
    """Every fitted conditional density must integrate to 1 over (0,1)."""
    print("\n" + "=" * 88)
    print("DENSITY CHECK: numerical integral over (0,1) of each FITTED conditional density")
    print("(catches a missing/incorrect Jacobian; all values must be 1.000000)")
    print("=" * 88)
    idx = [0, 21, 37]                     # Australia, Mexico, South Africa
    print(f"{'Model':16s} " + " ".join(f"{'obs '+str(i):>14}" for i in idx))
    for nm in models:
        p = fits[nm][0]
        vals = []
        for i in idx:
            f = lambda t: np.exp(models[nm](p, np.array([t]), X[i:i+1]))[0]
            vals.append(integrate.quad(f, 1e-10, 1 - 1e-10, limit=400)[0])
        print(f"{nm:16s} " + " ".join(f"{v:14.6f}" for v in vals))

# ---------------------------------------------------------------------- main
def main():
    raw = np.genfromtxt(CSV, delimiter=",", names=True, dtype=None, encoding="utf-8")
    y = raw["educ"].astype(float)                     # already divided by 100
    X = np.column_stack([np.ones(len(y)), raw["homicide"].astype(float),
                         raw["dwellings"].astype(float),
                         raw["labour_insec"].astype(float)])  # already /100
    n = len(y)
    print(f"n = {n}   y range [{y.min():.2f}, {y.max():.2f}]")
    print("design: intercept + homicide rate + dwellings w/o basic facilities "
          "+ labour market insecurity")

    p0 = np.append(np.linalg.lstsq(X, np.log(y / (1 - y)), rcond=None)[0], 0.0)

    # ---- (a) does the reconstruction reproduce the manuscript's Vasicek fit?
    print("\n" + "=" * 88)
    print("CHECK (a): reproduction of the manuscript's Table 2 / Table 3, Vasicek column")
    print("=" * 88)
    pv, llv = fit(ll_vasicek, y, X, p0.copy())
    se = hessian_se(lambda q: -np.sum(ll_vasicek(q, y, X)), pv)
    names = ["Intercept", "Homicide rate", "Dwellings w/o basic fac.",
             "Labour market insecurity", "gamma_0 (logit theta)"]
    paper = [(1.9778, 0.1528), (-0.0690, 0.0186), (0.0495, 0.0192),
             (-11.8295, 2.2846), (-2.0710, None)]
    print(f"{'parameter':28s} {'estimate':>10} {'paper':>10}   {'SE':>9} {'paper SE':>9}")
    for k in range(5):
        ps = f"{paper[k][1]:9.4f}" if paper[k][1] is not None else "        -"
        print(f"{names[k]:28s} {pv[k]:10.4f} {paper[k][0]:10.4f}   {se[k]:9.4f} {ps}")
    print(f"\n  theta_hat = {expit(pv[4]):.4f}   (manuscript 0.1119)")
    print(f"  logLik    = {llv:.4f}   (manuscript 38.3997)")
    print(f"  AIC       = {-2*llv+10:.4f}   (manuscript -66.7993)")
    ok = (abs(llv - 38.3997) < 5e-4) and (abs(expit(pv[4]) - 0.1119) < 5e-5)
    print(f"  ==> reconstruction {'VALIDATED' if ok else 'DOES NOT MATCH -- STOP'}")

    # ---- primary, like-for-like: every family parameterised by the MEAN
    fits_mean = {nm: fit(f, y, X, p0.copy()) for nm, f in MEAN_MODELS.items()}
    table(MEAN_MODELS, fits_mean, n,
          "PRIMARY TABLE -- all six families with logit(E[Y_i]) = x_i'beta (like for like)")
    vuong(fits_mean, MEAN_MODELS, y, X, n)
    check_densities(MEAN_MODELS, fits_mean, y, X)

    # ---- secondary: the three families in their usual median parameterisation
    mixed = dict(MEDIAN_MODELS)
    mixed["Vasicek"] = ll_vasicek
    mixed["Beta"] = ll_beta
    mixed["Simplex"] = ll_simplex
    fits_med = {nm: fit(f, y, X, p0.copy()) for nm, f in mixed.items()}
    table(mixed, fits_med, n,
          "SECONDARY TABLE -- logit-normal / Kumaraswamy / unit-Weibull linked at the "
          "MEDIAN\n(beta, simplex, Vasicek remain mean-linked; NOT like for like, "
          "shown for transparency)")
    vuong(fits_med, mixed, y, X, n)

    print("\n" + "=" * 88)
    print("The manuscript compares the Vasicek model against ONE competitor (beta) and")
    print("reports an Akaike weight of 0.8532 in that two-model set. Both tables above")
    print("use the competitor set from the authors' own Mazucheli et al. (2022) paper.")
    print("=" * 88)

if __name__ == "__main__":
    main()
