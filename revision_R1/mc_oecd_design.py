"""Monte Carlo study at the covariate design of the application: three
correlated covariates, the 38 observed covariate vectors held fixed (n = 38),
and the same design replicated four times (n = 152).  The true parameters are
the Vasicek ML estimates of the application.  Each replicate is fitted by ML
(BFGS with the analytical gradient), and the Wald intervals use the expected
information K of Section 3, evaluated at the estimate."""
import csv, json, sys
import numpy as np
from scipy import stats, optimize

M = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
rows = list(csv.DictReader(open(sys.argv[2], encoding='utf-8')))
X0 = np.array([[1.0, float(r['homicide']), float(r['dwellings']), float(r['labour_insec'])] for r in rows])
BETA = np.array([1.9778475, -0.0689978, 0.0494743, -11.8295346])
G0 = -2.0710


def nll(p, z, X):
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


def K_expected(b, g, X):
    al = 1 / (1 + np.exp(-(X @ b)))
    th = 1 / (1 + np.exp(-g))
    a = stats.norm.ppf(al)
    ad = al * (1 - al)
    td = th * (1 - th)
    ph = stats.norm.pdf(a)
    Kbb = (X * (ad ** 2 / (th * ph ** 2))[:, None]).T @ X
    Kbg = X.T @ (a * ad * td / (2 * th * (1 - th) * ph))
    Kgg = np.sum((a ** 2 * th + 2) * td ** 2 / (4 * th ** 2 * (1 - th) ** 2))
    k = X.shape[1]
    K = np.zeros((k + 1, k + 1))
    K[:k, :k] = Kbb
    K[:k, k] = K[k, :k] = Kbg
    K[k, k] = Kgg
    return K


out = {}
for rep in (1, 4):
    X = np.vstack([X0] * rep)
    n = X.shape[0]
    al = 1 / (1 + np.exp(-(X @ BETA)))
    th = 1 / (1 + np.exp(-G0))
    rng = np.random.default_rng([20261013, n])
    true = np.append(BETA, G0)
    est = np.empty((M, 5)); se = np.empty((M, 5)); ok = np.empty(M, bool)
    p0 = np.append(np.linalg.lstsq(X, np.log(al / (1 - al)), rcond=None)[0], 0.0)
    for m in range(M):
        z = (stats.norm.ppf(al) + np.sqrt(th) * rng.standard_normal(n)) / np.sqrt(1 - th)
        y = stats.norm.cdf(z)
        zz = stats.norm.ppf(np.clip(y, 1e-15, 1 - 1e-15))
        start = np.append(np.linalg.lstsq(X, np.log(y / (1 - y)), rcond=None)[0], 0.0)
        r = optimize.minimize(nll, start, args=(zz, X), jac=True, method='BFGS',
                              options=dict(gtol=1e-10, maxiter=5000))
        ok[m] = np.max(np.abs(r.jac)) < 1e-6 and np.all(np.isfinite(r.x))
        est[m] = r.x
        K = K_expected(r.x[:4], r.x[4], X)
        se[m] = np.sqrt(np.diag(np.linalg.inv(K)))
    e = est - true
    bias = e.mean(0); rmse = np.sqrt((e ** 2).mean(0))
    cp = ((np.abs(e) <= 1.959964 * se)).mean(0)
    sd = e.std(0, ddof=1)
    res = dict(n=n, M=M, converged=int(ok.sum()), bias=bias.tolist(), rmse=rmse.tolist(), cp=cp.tolist(),
               mcse_bias=(sd / np.sqrt(M)).tolist(), mcse_cp=np.sqrt(cp * (1 - cp) / M).tolist(),
               rel_bias=(bias / np.abs(true)).tolist())
    out[str(n)] = res
    names = ['beta0', 'beta1(HR)', 'beta2(DWBF)', 'beta3(LMI)', 'gamma0']
    print('n=%d converged %d/%d' % (n, ok.sum(), M))
    for k in range(5):
        print('  %-11s bias %+.4f (MCSE %.4f) RMSE %.4f CP %.4f (MCSE %.4f)' % (
            names[k], bias[k], sd[k] / np.sqrt(M), rmse[k], cp[k], np.sqrt(cp[k] * (1 - cp[k]) / M)))
json.dump(out, open('mc_oecd_design.json', 'w'), indent=1)
