"""Estudio de mala especificacion para la regresion Vasicek por la media.

Lo que le falta al paper: el estudio de residuos solo genera datos Vasicek y ajusta
Vasicek, de modo que mide la calibracion bajo la nula y nunca la POTENCIA. Aqui la
respuesta se genera de otras familias y SIEMPRE se ajusta Vasicek, para medir si los
dos diagnosticos detectan la falla.

Diseno: logit(alpha_i) = 0 + 1.5 x_i, con x_i uniforme en (0,1) y forma constante.
Para que la comparacion sea justa, cada competidor se calibra a la MISMA media
condicional y a la misma varianza condicional en x = 0.5 que el Vasicek con theta.

Se reporta, para cada residuo, los cuatro momentos y la tasa de rechazo al cinco por
ciento de una prueba formal de bondad de ajuste: Shapiro-Wilk contra la normal para
el residuo cuantil, y Anderson-Darling contra la exponencial para el Cox-Snell.
Bajo el modelo correcto esa tasa es el error de tipo uno; bajo los otros, la potencia.
"""
import numpy as np
from scipy import stats, optimize, special
import sys

RNG = np.random.default_rng(20260811)
M = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
NS = [30, 50, 100, 200]
B0, B1, THETA = 0.0, 1.5, 0.25

# ---------------------------------------------------------------- utilidades
def vas_var(al, th):
    """Var(Y) de la Vasicek por integracion de la representacion estocastica."""
    z, w = np.polynomial.hermite_e.hermegauss(120)
    a = stats.norm.ppf(al)
    y = stats.norm.cdf((a + np.sqrt(th) * z) / np.sqrt(1 - th))
    w = w / w.sum()
    m = (w * y).sum()
    return float((w * (y - m) ** 2).sum())

def gen_vasicek(al, th, rng):
    a = stats.norm.ppf(al)
    return stats.norm.cdf((a + np.sqrt(th) * rng.standard_normal(al.size))
                          / np.sqrt(1 - th))

def gen_beta(al, phi, rng):
    return rng.beta(al * phi, (1 - al) * phi)

def gen_kw(al, b, rng):
    """Kumaraswamy con media fijada: se resuelve el parametro a para cada alpha."""
    out = np.empty_like(al)
    for i, m in enumerate(al):
        f = lambda a: b * special.beta(1 + 1 / a, b) - m
        a = optimize.brentq(f, 1e-3, 60)
        u = rng.random()
        out[i] = (1 - (1 - u) ** (1 / b)) ** (1 / a)
    return out

def gen_simplex(al, sg, rng):
    """Simplex por rechazo sobre una malla fina (n chico, es barato)."""
    yy = np.linspace(1e-4, 1 - 1e-4, 2000)
    out = np.empty_like(al)
    for i, m in enumerate(al):
        d = (yy - m) ** 2 / (yy * (1 - yy) * m ** 2 * (1 - m) ** 2)
        lg = -0.5 * np.log(2 * np.pi * sg ** 2 * (yy * (1 - yy)) ** 3) - d / (2 * sg ** 2)
        p = np.exp(lg - lg.max()); p /= p.sum()
        out[i] = rng.choice(yy, p=p)
    return out

# ---------------------------------------------------------------- ajuste ML
def fit_vasicek(y, X):
    z = stats.norm.ppf(np.clip(y, 1e-12, 1 - 1e-12))
    def nll(p):
        b, g = p[:X.shape[1]], p[-1]
        th = 1 / (1 + np.exp(-g))
        if not (1e-6 < th < 1 - 1e-6):
            return 1e10
        a = stats.norm.ppf(np.clip(1 / (1 + np.exp(-(X @ b))), 1e-12, 1 - 1e-12))
        t = (np.sqrt(1 - th) * z - a) / np.sqrt(th)
        return -np.sum(0.5 * np.log(1 - th) - 0.5 * np.log(th) - 0.5 * t ** 2)
    p0 = np.zeros(X.shape[1] + 1)
    r = optimize.minimize(nll, p0, method="Nelder-Mead",
                          options=dict(maxiter=8000, xatol=1e-8, fatol=1e-8))
    b, g = r.x[:X.shape[1]], r.x[-1]
    th = 1 / (1 + np.exp(-g))
    a = stats.norm.ppf(np.clip(1 / (1 + np.exp(-(X @ b))), 1e-12, 1 - 1e-12))
    rq = (np.sqrt(1 - th) * z - a) / np.sqrt(th)
    return rq, r.success

def ad_exp(x):
    """Anderson-Darling contra Exp(1) con la media estimada, estadistico A2."""
    x = np.sort(x[x > 0]); n = x.size
    if n < 5:
        return np.nan
    u = 1 - np.exp(-x / x.mean())
    u = np.clip(u, 1e-12, 1 - 1e-12)
    i = np.arange(1, n + 1)
    a2 = -n - np.sum((2 * i - 1) * (np.log(u) + np.log(1 - u[::-1]))) / n
    return a2 * (1 + 0.6 / n)          # correccion de Stephens para Exp

AD_CRIT = 1.321                        # 5 por ciento, Exp con media estimada

# ---------------------------------------------------------------- simulacion
def moments(r):
    n = r.size
    m = r.mean(); d = r - m
    s2 = (d ** 2).sum() / n
    g1 = np.sqrt(n * (n - 1)) / (n - 2) * ((d ** 3).sum() / n) / s2 ** 1.5
    b2 = ((d ** 4).sum() / n) / s2 ** 2
    g2 = (n - 1) / ((n - 2) * (n - 3)) * ((n + 1) * (b2 - 3) + 6)
    return m, (d ** 2).sum() / (n - 1), g1, g2

print(f"M = {M} replicaciones   theta = {THETA}   logit(alpha) = {B0} + {B1} x\n")
hdr = f"{'verdad':<13}{'n':>5}{'media rQ':>10}{'var rQ':>9}{'asim rQ':>9}{'curt rQ':>9}" \
      f"{'rechazo SW':>12}{'rechazo AD':>12}"
print(hdr); print("-" * len(hdr))

for truth in ("Vasicek", "beta", "Kumaraswamy", "simplex"):
    for n in NS:
        x = RNG.random(n)
        X = np.column_stack([np.ones(n), x])
        al = 1 / (1 + np.exp(-(B0 + B1 * x)))
        v_target = np.array([vas_var(a, THETA) for a in al])
        # calibracion de cada competidor a la misma media y varianza en x = 0.5
        am = 1 / (1 + np.exp(-(B0 + B1 * 0.5)))
        vt = vas_var(am, THETA)
        phi = am * (1 - am) / vt - 1
        bkw = optimize.brentq(
            lambda b: (lambda a: b * special.beta(1 + 2 / a, b)
                       - (b * special.beta(1 + 1 / a, b)) ** 2 - vt)(
                optimize.brentq(lambda a: b * special.beta(1 + 1 / a, b) - am, 1e-3, 60)),
            0.3, 40)
        sg = np.sqrt(vt / (am ** 3 * (1 - am) ** 3)) if vt > 0 else 1.0

        mm, rej_sw, rej_ad, ok = [], 0, 0, 0
        for _ in range(M):
            if truth == "Vasicek":
                y = gen_vasicek(al, THETA, RNG)
            elif truth == "beta":
                y = gen_beta(al, phi, RNG)
            elif truth == "Kumaraswamy":
                y = gen_kw(al, bkw, RNG)
            else:
                y = gen_simplex(al, sg, RNG)
            y = np.clip(y, 1e-6, 1 - 1e-6)
            rq, good = fit_vasicek(y, X)
            if not np.all(np.isfinite(rq)):
                continue
            ok += 1
            mm.append(moments(rq))
            rej_sw += stats.shapiro(rq).pvalue < 0.05
            rgcs = -stats.norm.logsf(rq)
            a2 = ad_exp(rgcs)
            rej_ad += (a2 > AD_CRIT) if np.isfinite(a2) else 0
        mm = np.array(mm).mean(axis=0)
        print(f"{truth:<13}{n:>5}{mm[0]:>10.4f}{mm[1]:>9.4f}{mm[2]:>9.4f}{mm[3]:>9.4f}"
              f"{rej_sw / ok:>11.3f}{rej_ad / ok:>12.3f}")
    print()
