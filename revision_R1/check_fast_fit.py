"""Maximized log-likelihood with the bisection (six_models) and the fast inversion,
on one portfolio sample.  Usage: python check_fast_fit.py model m N k  (k-th sample)."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import six_models as S
import fast_invert as FI
from portfolio_six import draw

nm, m, N, k = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
rng = np.random.default_rng([20261013, m, 0 if not np.isfinite(N) else int(N)])
got = 0
while True:
    d = draw(rng, m, N)
    if d is None:
        continue
    if got == k:
        break
    got += 1
y, X = d
p0 = np.append(np.linalg.lstsq(X, np.log(y / (1 - y)), rcond=None)[0], 0.0)
slow = {"Logit-normal": S.ll_logitnormal, "Kumaraswamy": S.ll_kumaraswamy, "Unit-Weibull": S.ll_unitweibull}[nm]
t = time.time(); a = S.fit(slow, y, X, p0.copy()); ts = time.time() - t
t = time.time(); b = S.fit(FI.FAST[nm], y, X, p0.copy()); tf = time.time() - t
r = dict(model=nm, m=m, N=str(N), k=k, ll_slow=a[1], ll_fast=b[1], diff=a[1] - b[1], t_slow=ts, t_fast=tf,
         ll_fast_at_slow=float(FI.FAST[nm](a[0], y, X).sum()))
print(json.dumps(r), flush=True)
