"""Faster mean inversion for the six-model comparison.  six_models.py inverts the
mean of the logit-normal, Kumaraswamy and unit-Weibull models by 200 bisection
steps at every evaluation of the likelihood.  Here the increasing map v -> mean
is tabulated once per evaluation on a grid of the same bracket, inverted by
interpolation, and polished by Newton steps with a central-difference
derivative, kept inside the bracket.  check() compares the result with the
bisection of six_models.py."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import six_models as S

F = S.F
GRID = 401


def invert(f, lo, hi, target, newton=6):
    g = np.linspace(lo, hi, GRID)
    fg = f(g)
    fg = np.maximum.accumulate(fg)            # guard against flat tails
    v = np.interp(target, fg, g)
    h = 1e-6
    for _ in range(newton):
        d = (f(v + h) - f(v - h)) / (2 * h)
        step = np.where(d > 0, (f(v) - target) / np.where(d > 0, d, 1.0), 0.0)
        v = np.clip(v - step, lo, hi)
    return v


def ll_logitnormal(p, y, X):
    s = np.exp(p[-1])
    mu = np.clip(S.expit(X @ p[:-1]), 1e-9, 1 - 1e-9)
    m = invert(lambda v: F.logitnormal_mean(v, s), -60.0, 60.0, mu)
    return S.stats.norm.logpdf(S.lg(y / (1 - y)), m, s) - S.lg(y * (1 - y))


def ll_kumaraswamy(p, y, X):
    mu = np.clip(S.expit(X @ p[:-1]), 1e-9, 1 - 1e-9)
    b = np.exp(p[-1])
    la = invert(lambda v: F.kumaraswamy_mean(np.exp(v), b), -20.0, 20.0, mu)
    return F._kuma_ll(np.exp(la), b, y)


def ll_unitweibull(p, y, X):
    mu = np.clip(S.expit(X @ p[:-1]), 1e-9, 1 - 1e-9)
    b = np.exp(p[-1])
    la = invert(lambda v: F.unitweibull_mean(np.exp(v), b), -20.0, 20.0, mu)
    return F._uw_ll(np.exp(la), b, y)


FAST = {"Logit-normal": ll_logitnormal, "Kumaraswamy": ll_kumaraswamy, "Unit-Weibull": ll_unitweibull}


def use_fast():
    S.MODELS.update(FAST)


def check(seed=0, n=100):
    """Largest |loglik difference| between fast and bisection versions at random parameters."""
    rng = np.random.default_rng(seed)
    X = np.column_stack([np.ones(n), rng.random(n)])
    y = np.clip(rng.beta(2, 20, n), 1e-4, 1 - 1e-4)
    slow = {"Logit-normal": S.ll_logitnormal, "Kumaraswamy": S.ll_kumaraswamy, "Unit-Weibull": S.ll_unitweibull}
    worst = {}
    for nm in FAST:
        w = 0.0
        for _ in range(40):
            p = np.array([-2.5 + rng.normal(0, 1), rng.normal(0, 2), rng.normal(0, 1.5)])
            a, b = FAST[nm](p, y, X).sum(), slow[nm](p, y, X).sum()
            w = max(w, abs(a - b))
        worst[nm] = w
    return worst


if __name__ == '__main__':
    print(check())
