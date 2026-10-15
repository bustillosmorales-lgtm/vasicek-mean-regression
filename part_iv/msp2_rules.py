"""Two different rules get confused in the closing sentence of Part IV.

  calibrated LR   reject the Vasicek when D = l_alt - l_vas exceeds a threshold
                  simulated under the null at level alpha.  A proper size-alpha
                  test, and the one Table B reports.
  selection rule  prefer the alternative when D > 0, i.e. the AIC/BIC comparison
                  the application section actually performs (equal parameter
                  counts, so the criteria agree with D).  Its size is whatever
                  it happens to be.

They are not the same rule and they do not have the same size.  This script
measures both, and the share of the calibrated rule's rejections that fall in
the region where the Vasicek has the HIGHER maximised log-likelihood -- where
the calibrated test rejects a model that fits the sample better.
"""
import json, sys, time, warnings
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import msp2_core as C
import msp2_design as DD

warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")

B0, B1, ALPHA = 0.0, 1.5, 0.05
THETAS = [float(v) for v in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0.10", "0.25"])]
NLIST = [int(v) for v in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["100", "200"])]
MLR = int(sys.argv[3]) if len(sys.argv) > 3 else 3000
MASTER = 20260812
_D = {}


def _init():
    for th in THETAS:
        _D[th] = DD.Design(th, B0, B1)


def _blk(task):
    th, n, fam, alt, seed, cnt = task
    rng = np.random.default_rng(seed)
    D = np.empty(cnt)
    for i in range(cnt):
        x = rng.random(n)
        X = np.column_stack([np.ones(n), x])
        mu = 1 / (1 + np.exp(-(B0 + B1 * x)))
        y = np.clip(_D[th].draw(alt if fam == "alt" else "vasicek", "full", mu, rng)
                    if fam == "alt" else C.vas_rvs(mu, th, rng), 1e-9, 1 - 1e-9)
        _, llv, _, _, _ = C.vas_fit(y, X)
        lla, _ = C.ll_alt_max(y, X, alt, _D[th])
        D[i] = lla - llv
    return D


def _chunk(th, n, fam, alt, ss, total, nw):
    per = max(1, total // nw)
    seeds = ss.spawn(nw)
    out, left = [], total
    for k in range(nw):
        c = min(per, left) if k < nw - 1 else left
        left -= c
        if c > 0:
            out.append((th, n, fam, alt, seeds[k], c))
    return out


def main():
    nw = 20
    t0 = time.time()
    res = {}
    with ProcessPoolExecutor(max_workers=nw, initializer=_init) as pool:
        _init()
        print("%-6s %-4s %-8s %9s %9s %9s %9s %9s"
              % ("theta", "n", "alt", "crit", "LR pow", "AIC pow", "AIC size", "neg-D%"))
        for th in THETAS:
            for n in NLIST:
                for alt in ("beta", "kw", "simplex"):
                    s0 = np.random.SeedSequence([MASTER, int(th * 1e6), n, 7, len(alt)])
                    s1 = np.random.SeedSequence([MASTER, int(th * 1e6), n, 8, len(alt)])
                    d0 = np.concatenate(list(pool.map(_blk, _chunk(th, n, "null", alt, s0, MLR, nw))))
                    d1 = np.concatenate(list(pool.map(_blk, _chunk(th, n, "alt", alt, s1, MLR, nw))))
                    crit = float(np.percentile(d0, 100 * (1 - ALPHA)))
                    rej = d1 > crit
                    r = dict(crit=crit,
                             lr_power=float(rej.mean()),
                             aic_power=float((d1 > 0).mean()),
                             aic_size=float((d0 > 0).mean()),
                             neg_share=float((rej & (d1 < 0)).sum() / max(rej.sum(), 1)))
                    res["%g|%d|%s" % (th, n, alt)] = r
                    print("%-6.2f %-4d %-8s %9.4f %9.4f %9.4f %9.4f %9.4f"
                          % (th, n, alt, crit, r["lr_power"], r["aic_power"],
                             r["aic_size"], r["neg_share"]), flush=True)
    res["_elapsed_s"] = time.time() - t0
    json.dump(res, open("msp2_rules.json", "w"), indent=1)
    print("\nwritten msp2_rules.json (%.0f s)" % res["_elapsed_s"])


if __name__ == "__main__":
    main()
