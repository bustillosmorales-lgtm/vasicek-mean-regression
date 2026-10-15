"""Six-model comparison on the portfolio default fractions of portfolio_mechanism.py.
The samples are the same as in that study (same generator, same seeds, same
order of draws); each is fitted by the six models of six_models.py, all with a
logit link and three parameters, and the model with the lowest AIC is recorded.
Usage: python portfolio_six.py R m N"""
import sys, json, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy import stats
import six_models as S
from portfolio_mechanism import B0, B1, RHO


def draw(rng, m, N):
    x = rng.random(m)
    pd = 1 / (1 + np.exp(-(B0 + B1 * x)))
    z = rng.standard_normal(m)
    p = stats.norm.cdf((stats.norm.ppf(pd) - np.sqrt(RHO) * z) / np.sqrt(1 - RHO))
    y = rng.binomial(N, p) / N if np.isfinite(N) else p
    if np.any(y <= 0) or np.any(y >= 1):
        return None
    return y, np.column_stack([np.ones(m), x])


if __name__ == '__main__':
    # usage: R m N shard nshards -- fits samples j < R with j % nshards == shard, in the
    # draw order of portfolio_mechanism.py, so the samples are those of Table 14
    import fast_invert
    fast_invert.use_fast()
    R, m, N, sh, ns = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    rng = np.random.default_rng([20261013, m, 0 if not np.isfinite(N) else int(N)])
    j, rows = 0, []
    while j < R:
        d = draw(rng, m, N)
        if d is None:
            continue
        if j % ns == sh:
            res = S.compare(*d)
            rows.append(dict(j=j, rank=res['rank'], aic=res['aic']))
            json.dump(rows, open('ps6_%d_%s_%d.json' % (m, sys.argv[3], sh), 'w'))
        j += 1
    print('done', m, N, sh, len(rows), flush=True)
