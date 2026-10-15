"""Core routines for the misspecification study of the Vasicek mean regression.

WHAT IS COMPARED.  The quantile residual and the generalized Cox-Snell residual
are related by r^GCS = -log(1 - Phi(r^Q)), a strictly increasing bijection, so
they carry the same information and no rank-based criterion can separate them.
What this study compares is therefore two GOODNESS-OF-FIT TESTS applied to one
residual: Shapiro-Wilk on the normal scale, and Anderson-Darling against the
unit exponential on the Cox-Snell scale.

HOW THE ALTERNATIVES ARE BUILT.  Two matchings are provided, and the contrast
between them is what separates shape from variance:

  "centre"  dispersion fixed at the value that matches the Vasicek variance at
            the centre of the design.  Away from the centre the variance
            functions differ, so a diagnostic can respond to either shape or
            variance.
  "full"    dispersion varies with the mean so that the variance matches the
            Vasicek variance at EVERY observation.  Only the shape of the
            conditional density then differs.

A third generator, "purevar", draws genuine Vasicek responses with a
mean-dependent shape parameter chosen to reproduce a competitor's variance
function.  Its conditional density is Vasicek everywhere, so any rejection it
produces is a response to the variance function alone.

Because the target variance v(mu) = Var_Vasicek(mu, theta) is a deterministic
function of the mean, every dispersion parameter is a deterministic function of
the mean too.  All of them are tabulated once per theta and interpolated.
"""
import numpy as np
from scipy import stats, optimize, special

# --------------------------------------------------------------------------
# Vasicek
# --------------------------------------------------------------------------
_GHZ, _GHW = np.polynomial.hermite_e.hermegauss(160)
_GHW = _GHW / _GHW.sum()


def vas_var(alpha, theta):
    """Var of the Vasicek at mean alpha and shape theta. Vectorised over both."""
    alpha = np.atleast_1d(np.asarray(alpha, float))
    theta = np.broadcast_to(np.asarray(theta, float), alpha.shape)
    a = stats.norm.ppf(np.clip(alpha, 1e-12, 1 - 1e-12))[:, None]
    th = theta[:, None]
    y = stats.norm.cdf((a + np.sqrt(th) * _GHZ) / np.sqrt(1 - th))
    m = (_GHW * y).sum(1, keepdims=True)
    return (_GHW * (y - m) ** 2).sum(1)


def vas_rvs(alpha, theta, rng):
    a = stats.norm.ppf(alpha)
    e = rng.standard_normal(np.shape(alpha))
    return stats.norm.cdf((a + np.sqrt(theta) * e) / np.sqrt(1 - theta))


def theta_for_var(mu, v_target, lo=1e-4, hi=0.999):
    """Shape that gives the Vasicek variance v_target at mean mu (vectorised)."""
    lo = np.full_like(mu, lo)
    hi = np.full_like(mu, hi)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        up = vas_var(mu, mid) < v_target
        lo = np.where(up, mid, lo)
        hi = np.where(up, hi, mid)
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------
# ML fit of the Vasicek mean regression, constant shape
# --------------------------------------------------------------------------
def _nll(p, z, X):
    k = X.shape[1]
    b, g = p[:k], p[-1]
    n = z.size
    th = min(max(1.0 / (1.0 + np.exp(-g)), 1e-10), 1 - 1e-10)
    et = np.clip(1.0 / (1.0 + np.exp(-(X @ b))), 1e-12, 1 - 1e-12)
    a = stats.norm.ppf(et)
    t = (np.sqrt(1 - th) * z - a) / np.sqrt(th)
    f = 0.5 * n * (np.log(th) - np.log1p(-th)) + 0.5 * np.sum(t * t)
    da_db = (et * (1 - et) / stats.norm.pdf(a))[:, None] * X
    gb = (t * (-1.0 / np.sqrt(th))) @ da_db
    dt_dth = -z / (2 * np.sqrt(th * (1 - th))) - t / (2 * th)
    dth = 0.5 * n * (1 / th + 1 / (1 - th)) + np.sum(t * dt_dth)
    return f / n, np.concatenate([gb, [dth * th * (1 - th)]]) / n


_STARTS = ((0.0, 0.0, 0.0), (0.0, 1.5, -1.0), (0.0, -1.0, 1.0))


def vas_fit(y, X, starts=_STARTS, want_se=False):
    """Returns (quantile residuals, loglik, converged, params, se_or_None)."""
    z = stats.norm.ppf(np.clip(y, 1e-12, 1 - 1e-12))
    n = z.size
    best = None
    for s in starts:
        r = optimize.minimize(_nll, np.array(s, float), args=(z, X), jac=True,
                              method="BFGS", options=dict(gtol=1e-9, maxiter=3000))
        if best is None or r.fun < best.fun:
            best = r
    ok = bool(np.max(np.abs(best.jac)) < 1e-6) and np.all(np.isfinite(best.x))
    k = X.shape[1]
    b, g = best.x[:k], best.x[-1]
    th = 1.0 / (1.0 + np.exp(-g))
    a = stats.norm.ppf(np.clip(1.0 / (1.0 + np.exp(-(X @ b))), 1e-12, 1 - 1e-12))
    rq = (np.sqrt(1 - th) * z - a) / np.sqrt(th)
    ll = -best.fun * n + 0.5 * np.sum(z ** 2)
    se = None
    if want_se:
        # observed information of the 1/n-scaled objective, by central differences
        h = 1e-5
        H = np.empty((k + 1, k + 1))
        for i in range(k + 1):
            ep = np.zeros(k + 1); ep[i] = h
            gp = _nll(best.x + ep, z, X)[1]
            gm = _nll(best.x - ep, z, X)[1]
            H[i] = (gp - gm) / (2 * h)
        H = 0.5 * (H + H.T) * n
        try:
            se = np.sqrt(np.diag(np.linalg.inv(H)))
        except np.linalg.LinAlgError:
            se = np.full(k + 1, np.nan)
    return rq, float(ll), ok, best.x, se


# --------------------------------------------------------------------------
# the two goodness-of-fit tests
# --------------------------------------------------------------------------
def a2_exp(x):
    x = np.sort(x[x > 0])
    n = x.size
    if n < 5:
        return np.nan
    u = np.clip(1 - np.exp(-x / x.mean()), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return -n - np.sum((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))) / n


def gof_pair(rq):
    """(Shapiro-Wilk W on the normal scale, Anderson-Darling A2 on the Cox-Snell
    scale).  Both are computed from the same ordered sample."""
    return float(stats.shapiro(rq).statistic), float(a2_exp(-stats.norm.logsf(rq)))


# --------------------------------------------------------------------------
# alternatives
# --------------------------------------------------------------------------
def simplex_logpdf(y, mu, sigma):
    d = (y - mu) ** 2 / (y * (1 - y) * mu ** 2 * (1 - mu) ** 2)
    return -0.5 * np.log(2 * np.pi * sigma ** 2 * (y * (1 - y)) ** 3) - d / (2 * sigma ** 2)


def _probit_grid(ny=6001, zmax=8.5):
    """y grid equally spaced in the PROBIT scale: that is the scale the tests
    act on, so it is the scale that has to be resolved."""
    z = np.linspace(-zmax, zmax, ny)
    return stats.norm.cdf(z), z


def simplex_var(mu, sigma, ny=6001):
    """Variance of the simplex. Vectorised over mu, sigma scalar or matching."""
    mu = np.atleast_1d(np.asarray(mu, float))[:, None]
    sigma = np.atleast_1d(np.asarray(sigma, float))
    if sigma.size == 1:
        sigma = np.full(mu.shape[0], sigma[0])
    sigma = sigma[:, None]
    y, _ = _probit_grid(ny)
    y = y[None, :]
    with np.errstate(over="ignore", under="ignore", divide="ignore", invalid="ignore"):
        lf = simplex_logpdf(y, mu, sigma)
        f = np.nan_to_num(np.exp(lf - np.nanmax(lf, axis=1, keepdims=True)))
    yy = y[0]
    m0 = np.trapezoid(f, yy, axis=1)
    m1 = np.trapezoid(f * yy, yy, axis=1) / m0
    m2 = np.trapezoid(f * yy * yy, yy, axis=1) / m0
    return m2 - m1 ** 2


def kw_a_of(mu, b):
    """Kumaraswamy first shape from the mean, given the second."""
    a = np.exp(np.linspace(np.log(1e-3), np.log(200.0), 4000))
    m = b * special.beta(1 + 1.0 / a, b)
    return np.interp(mu, m, a)


def kw_var(mu, b):
    a = kw_a_of(mu, b)
    return b * special.beta(1 + 2.0 / a, b) - mu ** 2


# --------------------------------------------------------------------------
# log-densities, for the likelihood-ratio benchmark and for the exact
# Neyman-Pearson ceiling
# --------------------------------------------------------------------------
def vas_logpdf(y, alpha, theta):
    a = stats.norm.ppf(np.clip(alpha, 1e-12, 1 - 1e-12))
    z = stats.norm.ppf(np.clip(y, 1e-12, 1 - 1e-12))
    t = (np.sqrt(1 - theta) * z - a) / np.sqrt(theta)
    return 0.5 * np.log((1 - theta) / theta) + stats.norm.logpdf(t) - stats.norm.logpdf(z)


def kw_logpdf(y, a, b):
    return np.log(a) + np.log(b) + (a - 1) * np.log(y) + (b - 1) * np.log1p(-y ** a)


def _neg_beta(p, y, X):
    mu = np.clip(1 / (1 + np.exp(-(X @ p[:2]))), 1e-9, 1 - 1e-9)
    ph = np.exp(np.clip(p[2], -20, 20))
    return -np.sum(stats.beta.logpdf(y, mu * ph, (1 - mu) * ph))


def _neg_simplex(p, y, X):
    mu = np.clip(1 / (1 + np.exp(-(X @ p[:2]))), 1e-9, 1 - 1e-9)
    s2 = np.exp(np.clip(p[2], -20, 20))
    d = (y - mu) ** 2 / (y * (1 - y) * mu ** 2 * (1 - mu) ** 2)
    return -np.sum(-0.5 * np.log(2 * np.pi * s2 * (y * (1 - y)) ** 3) - d / (2 * s2))


def _neg_kw(p, y, X):
    mu = np.clip(1 / (1 + np.exp(-(X @ p[:2]))), 1e-9, 1 - 1e-9)
    b = np.exp(np.clip(p[2], -10, 10))
    a = kw_a_of(mu, b)
    return -np.sum(kw_logpdf(y, a, b))


def ll_alt_max(y, X, alt, d):
    """ML fit of the alternative with a CONSTANT dispersion, as an analyst
    would fit it.  Returns (maximised loglik, converged)."""
    y = np.clip(y, 1e-9, 1 - 1e-9)
    neg, s0 = {"beta": (_neg_beta, np.log(max(d.phi_c, 1e-3))),
               "simplex": (_neg_simplex, 2 * np.log(max(d.s_c, 1e-3))),
               "kw": (_neg_kw, np.log(max(d.b_c, 1e-3)))}[alt]
    best, ok = None, False
    for st in ([0.0, 1.5, s0], [0.0, 0.0, 0.0]):
        r = optimize.minimize(neg, np.array(st, float), args=(y, X), method="Nelder-Mead",
                              options=dict(maxiter=6000, maxfev=6000, xatol=1e-8, fatol=1e-8))
        if best is None or r.fun < best.fun:
            best, ok = r, bool(r.success)
    return -float(best.fun), ok


def ll_vas_true(y, mu, theta):
    """Vasicek loglik with every parameter known."""
    return float(np.sum(vas_logpdf(y, mu, theta)))


def ll_alt_true(y, mu, alt, d):
    """Alternative loglik with every parameter known, under the 'full'
    matching, i.e. the dispersion the generator actually used."""
    y = np.clip(y, 1e-12, 1 - 1e-12)
    if alt == "beta":
        ph = d._interp(d.phi_f, mu)
        return float(np.sum(stats.beta.logpdf(y, mu * ph, (1 - mu) * ph)))
    if alt == "simplex":
        sg = d._interp(d.s_f, mu)
        return float(np.sum(simplex_logpdf(y, mu, sg)))
    if alt == "kw":
        b = d._interp(d.b_f, mu)
        a = np.array([kw_a_of(np.array([m]), bb)[0] for m, bb in zip(mu, b)])
        return float(np.sum(kw_logpdf(y, a, b)))
    raise ValueError(alt)
