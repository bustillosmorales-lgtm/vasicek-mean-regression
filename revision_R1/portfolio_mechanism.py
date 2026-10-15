"""Default fractions of finite loan portfolios under the one-factor Gaussian model
of Vasicek (1987).  Pool i has N obligors, default probability PD_i with
logit(PD_i) = b0 + b1 x_i, x_i ~ U(0,1), and asset correlation rho.  Given the
systematic factor Z_i ~ N(0,1), each obligor defaults independently with
probability p_i(Z_i) = Phi((Phi^{-1}(PD_i) - sqrt(rho) Z_i)/sqrt(1 - rho)), and the
observed response is the default fraction D_i ~ Binomial(N, p_i(Z_i))/N.
As N grows D_i tends to Vas(PD_i, rho); for finite N it is a binomial mixture and
is NOT a Vasicek variable.  For each replicate the Vasicek and the beta mean
regressions (logit link, constant shape) are fitted by ML and compared by AIC,
which with equal parameter counts is the comparison of maximized log-likelihoods."""
import sys, json
import numpy as np
from scipy import stats, optimize, special

B0, B1, RHO = -2.5, 1.5, 0.10


def nll_vas(p, y, X):
    al = np.clip(1 / (1 + np.exp(-(X @ p[:-1]))), 1e-12, 1 - 1e-12)
    th = 1 / (1 + np.exp(-p[-1]))
    q = stats.norm.ppf(y); a = stats.norm.ppf(al)
    return -np.sum(0.5 * np.log(1 - th) - 0.5 * np.log(th) + 0.5 * q ** 2 - (np.sqrt(1 - th) * q - a) ** 2 / (2 * th))


def nll_beta(p, y, X):
    mu = np.clip(1 / (1 + np.exp(-(X @ p[:-1]))), 1e-12, 1 - 1e-12)
    phi = np.exp(p[-1])
    return -np.sum(special.gammaln(phi) - special.gammaln(mu * phi) - special.gammaln((1 - mu) * phi)
                   + (mu * phi - 1) * np.log(y) + ((1 - mu) * phi - 1) * np.log1p(-y))


def fit(nll, y, X, p0):
    best = None
    for s in (p0, p0 + np.array([0, 0, 1.0])):
        r = optimize.minimize(nll, s, args=(y, X), method='BFGS', options=dict(gtol=1e-8, maxiter=2000))
        r = optimize.minimize(nll, r.x, args=(y, X), method='Nelder-Mead',
                              options=dict(xatol=1e-9, fatol=1e-11, maxiter=20000, maxfev=20000))
        if best is None or r.fun < best.fun:
            best = r
    return -best.fun


def one(rng, m, N):
    x = rng.random(m)
    pd = 1 / (1 + np.exp(-(B0 + B1 * x)))
    z = rng.standard_normal(m)
    p = stats.norm.cdf((stats.norm.ppf(pd) - np.sqrt(RHO) * z) / np.sqrt(1 - RHO))
    y = rng.binomial(N, p) / N if np.isfinite(N) else p
    if np.any(y <= 0) or np.any(y >= 1):
        return None
    X = np.column_stack([np.ones(m), x])
    lz = np.log(y / (1 - y))
    p0 = np.append(np.linalg.lstsq(X, lz, rcond=None)[0], 0.0)
    return fit(nll_vas, y, X, p0) - fit(nll_beta, y, X, p0)


if __name__ == '__main__':
    R = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    out = {}
    MS = [int(sys.argv[2])] if len(sys.argv) > 2 else [40, 100]
    NS = [float(sys.argv[3])] if len(sys.argv) > 3 else [500, 2000, 10000, np.inf]
    for m in MS:
        for N in NS:
            rng = np.random.default_rng([20261013, m, 0 if not np.isfinite(N) else int(N)])
            d, drop = [], 0
            while len(d) < R:
                v = one(rng, m, N)
                if v is None:
                    drop += 1
                    continue
                d.append(v)
            d = np.array(d)
            f = float((d > 0).mean())
            out['%d|%s' % (m, N)] = dict(m=m, N=str(N), R=R, frac_vas=f, se=float(np.sqrt(f * (1 - f) / R)),
                                       median_dll=float(np.median(d)), dropped_zero=drop)
            print(m, N, 'Vasicek preferred %.3f (SE %.3f)  median dlogL %.2f  dropped %d' % (
                f, np.sqrt(f * (1 - f) / R), np.median(d), drop), flush=True)
    json.dump(out, open('portfolio_mechanism_%s.json' % '_'.join(sys.argv[2:]), 'w'), indent=1)
