"""Estudio de mala especificacion, con los dos tests CALIBRADOS contra el nulo.

En la primera corrida el Anderson-Darling sobre el residuo Cox-Snell daba error de
tipo uno de 0.002 a 0.007 con el valor critico de tabla: ese valor no vale para
residuos ajustados, y comparar potencias asi es injusto con el Cox-Snell. Aqui el
umbral de CADA prueba se toma del percentil 95 de su propia distribucion bajo el
modelo correcto, de modo que las dos tienen 5 por ciento exacto por construccion y
la comparacion de potencia es sobre la misma base.
"""
import numpy as np
from scipy import stats, optimize, special
import sys

RNG = np.random.default_rng(20260811)
M = int(sys.argv[1]) if len(sys.argv) > 1 else 800
NS = [30, 50, 100, 200]
B0, B1, THETA = 0.0, 1.5, 0.25

def vas_var(al, th):
    z, w = np.polynomial.hermite_e.hermegauss(120)
    a = stats.norm.ppf(al)
    y = stats.norm.cdf((a + np.sqrt(th) * z) / np.sqrt(1 - th))
    w = w / w.sum()
    m = (w * y).sum()
    return float((w * (y - m) ** 2).sum())

def gen_vasicek(al, th, rng):
    a = stats.norm.ppf(al)
    return stats.norm.cdf((a + np.sqrt(th) * rng.standard_normal(al.size)) / np.sqrt(1 - th))

def gen_beta(al, phi, rng):
    return rng.beta(al * phi, (1 - al) * phi)

def gen_kw(al, b, rng):
    out = np.empty_like(al)
    for i, m in enumerate(al):
        a = optimize.brentq(lambda a: b * special.beta(1 + 1 / a, b) - m, 1e-3, 60)
        out[i] = (1 - (1 - rng.random()) ** (1 / b)) ** (1 / a)
    return out

def gen_simplex(al, sg, rng):
    yy = np.linspace(1e-4, 1 - 1e-4, 1500)
    out = np.empty_like(al)
    for i, m in enumerate(al):
        d = (yy - m) ** 2 / (yy * (1 - yy) * m ** 2 * (1 - m) ** 2)
        lg = -0.5 * np.log((yy * (1 - yy)) ** 3) - d / (2 * sg ** 2)
        p = np.exp(lg - lg.max()); p /= p.sum()
        out[i] = rng.choice(yy, p=p)
    return out

def fit_resid(y, X):
    z = stats.norm.ppf(np.clip(y, 1e-12, 1 - 1e-12))
    def nll(p):
        b, g = p[:X.shape[1]], p[-1]
        th = 1 / (1 + np.exp(-g))
        if not (1e-6 < th < 1 - 1e-6):
            return 1e10
        a = stats.norm.ppf(np.clip(1 / (1 + np.exp(-(X @ b))), 1e-12, 1 - 1e-12))
        t = (np.sqrt(1 - th) * z - a) / np.sqrt(th)
        return -np.sum(0.5 * np.log(1 - th) - 0.5 * np.log(th) - 0.5 * t ** 2)
    r = optimize.minimize(nll, np.zeros(X.shape[1] + 1), method="Nelder-Mead",
                          options=dict(maxiter=8000, xatol=1e-8, fatol=1e-8))
    b, g = r.x[:X.shape[1]], r.x[-1]
    th = 1 / (1 + np.exp(-g))
    a = stats.norm.ppf(np.clip(1 / (1 + np.exp(-(X @ b))), 1e-12, 1 - 1e-12))
    return (np.sqrt(1 - th) * z - a) / np.sqrt(th), r.success

def a2_exp(x):
    x = np.sort(x[x > 0]); n = x.size
    if n < 5:
        return np.nan
    u = np.clip(1 - np.exp(-x / x.mean()), 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    return -n - np.sum((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))) / n

def stats_pair(rq):
    w = stats.shapiro(rq).statistic          # chico = mala normalidad
    a2 = a2_exp(-stats.norm.logsf(rq))       # grande = mala exponencialidad
    return w, a2

print(f"M = {M}   theta = {THETA}   umbrales calibrados al 5 por ciento bajo el modelo correcto\n")
hdr = (f"{'verdad':<13}{'n':>5}{'asim rQ':>9}{'curt rQ':>9}"
       f"{'potencia rQ':>13}{'potencia rGCS':>15}")
print(hdr); print("-" * len(hdr))

for n in NS:
    x = RNG.random(n)
    X = np.column_stack([np.ones(n), x])
    al = 1 / (1 + np.exp(-(B0 + B1 * x)))
    am = 1 / (1 + np.exp(-(B0 + B1 * 0.5)))
    vt = vas_var(am, THETA)
    phi = am * (1 - am) / vt - 1
    bkw = optimize.brentq(
        lambda b: (lambda a: b * special.beta(1 + 2 / a, b)
                   - (b * special.beta(1 + 1 / a, b)) ** 2 - vt)(
            optimize.brentq(lambda a: b * special.beta(1 + 1 / a, b) - am, 1e-3, 60)), 0.3, 40)
    sg = np.sqrt(vt / (am ** 3 * (1 - am) ** 3))

    # ---- nulo: se calibran los dos umbrales
    W0, A0 = [], []
    for _ in range(M):
        rq, _ = fit_resid(np.clip(gen_vasicek(al, THETA, RNG), 1e-6, 1 - 1e-6), X)
        if not np.all(np.isfinite(rq)):
            continue
        w, a2 = stats_pair(rq)
        W0.append(w); A0.append(a2)
    cw = np.percentile(W0, 5)        # se rechaza si W < cw
    ca = np.percentile(A0, 95)       # se rechaza si A2 > ca

    for truth, gen in (("Vasicek", lambda: gen_vasicek(al, THETA, RNG)),
                       ("beta", lambda: gen_beta(al, phi, RNG)),
                       ("Kumaraswamy", lambda: gen_kw(al, bkw, RNG)),
                       ("simplex", lambda: gen_simplex(al, sg, RNG))):
        sk, ku, rw, ra, ok = [], [], 0, 0, 0
        for _ in range(M):
            rq, _ = fit_resid(np.clip(gen(), 1e-6, 1 - 1e-6), X)
            if not np.all(np.isfinite(rq)):
                continue
            ok += 1
            d = rq - rq.mean(); s2 = (d ** 2).sum() / n
            g1 = np.sqrt(n * (n - 1)) / (n - 2) * ((d ** 3).sum() / n) / s2 ** 1.5
            b2 = ((d ** 4).sum() / n) / s2 ** 2
            g2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (b2 - 3) + 6)
            sk.append(g1); ku.append(g2)
            w, a2 = stats_pair(rq)
            rw += w < cw
            ra += a2 > ca
        print(f"{truth:<13}{n:>5}{np.mean(sk):>9.4f}{np.mean(ku):>9.4f}"
              f"{rw / ok:>12.3f}{ra / ok:>14.3f}")
    print()
