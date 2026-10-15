"""Misspecification study, rebuilt.

Phase A   size and power of the two goodness-of-fit TESTS, plus what the
          undetected misspecification costs the fitted model
Phase B   the same alternatives against a likelihood-ratio benchmark and
          against the exact Neyman-Pearson ceiling

Generators per cell:
  vasicek            the null
  beta/kw/simplex    matched in the variance at EVERY observation ("full"),
                     so that only the shape of the density differs
  beta_c/kw_c/simplex_c   matched at the centre only, run at one theta so the
                     difference isolates how much signal is variance function
  purevar            genuine Vasicek data with a competitor's variance
                     function: zero shape difference, so any rejection it
                     produces is a response to the variance function alone
"""
import json, sys, time, warnings
import numpy as np

# the simplex density underflows at the ends of the probit grid by construction;
# those cells contribute nothing and are zeroed downstream
warnings.filterwarnings("ignore", category=RuntimeWarning)
np.seterr(all="ignore")
from scipy import stats
from concurrent.futures import ProcessPoolExecutor
import msp2_core as C
import msp2_design as DD

B0, B1 = 0.0, 1.5
ALPHA = 0.05
BETA1_TRUE = B1

THETAS = [float(v) for v in (sys.argv[1].split(",") if len(sys.argv) > 1 else ["0.05", "0.10", "0.25", "0.50"])]
NLIST = [int(v) for v in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["30", "100", "200"])]
N0 = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
M = int(sys.argv[4]) if len(sys.argv) > 4 else 5000
MLR = int(sys.argv[5]) if len(sys.argv) > 5 else 2000
MASTER = int(sys.argv[6]) if len(sys.argv) > 6 else 20260812
CENTRE_AT = 0.25          # the theta at which the centre-matched arm is run

FULL = ("beta", "kw", "simplex")
CENTRE = ("beta_c", "kw_c", "simplex_c")
_D = {}


def _init():
    for th in THETAS:
        _D[th] = DD.Design(th, B0, B1)


def _gen(th, fam, mu, rng):
    d = _D[th]
    if fam.endswith("_c"):
        return d.draw(fam[:-2], "centre", mu, rng)
    return d.draw(fam, "full", mu, rng)


def _block_A(task):
    """-> Shapiro W, AD A2, converged, beta1 hat, its se, theta hat, loglik."""
    th, n, fam, seed, cnt = task
    rng = np.random.default_rng(seed)
    W = np.empty(cnt); A = np.empty(cnt); ok = np.empty(cnt, bool)
    b1 = np.empty(cnt); s1 = np.empty(cnt); thh = np.empty(cnt)
    for i in range(cnt):
        x = rng.random(n)
        X = np.column_stack([np.ones(n), x])
        mu = 1 / (1 + np.exp(-(B0 + B1 * x)))
        y = np.clip(_gen(th, fam, mu, rng), 1e-9, 1 - 1e-9)
        rq, _, c, p, se = C.vas_fit(y, X, want_se=True)
        W[i], A[i] = C.gof_pair(rq)
        ok[i] = c
        b1[i] = p[1]
        s1[i] = se[1] if se is not None else np.nan
        thh[i] = 1 / (1 + np.exp(-p[2]))
    return W, A, ok, b1, s1, thh


def _block_B(task):
    """-> (loglik alt - loglik vasicek, exact NP statistic, both fits converged)."""
    th, n, fam, alt, seed, cnt = task
    rng = np.random.default_rng(seed)
    D = np.empty(cnt); NP = np.empty(cnt); ok = np.empty(cnt, bool)
    d = _D[th]
    for i in range(cnt):
        x = rng.random(n)
        X = np.column_stack([np.ones(n), x])
        mu = 1 / (1 + np.exp(-(B0 + B1 * x)))
        y = np.clip(_gen(th, fam, mu, rng), 1e-9, 1 - 1e-9)
        _, llv, cv, _, _ = C.vas_fit(y, X)
        lla, ca = C.ll_alt_max(y, X, alt, d)
        D[i] = lla - llv
        NP[i] = C.ll_alt_true(y, mu, alt, d) - C.ll_vas_true(y, mu, th)
        ok[i] = bool(cv and ca)
    return D, NP, ok


def _chunk(th, n, fam, ss, total, nw, extra=()):
    per = max(1, total // nw)
    seeds = ss.spawn(nw)
    out, left = [], total
    for k in range(nw):
        c = min(per, left) if k < nw - 1 else left
        left -= c
        if c > 0:
            out.append((th, n, fam) + tuple(extra) + (seeds[k], c))
    return out


def mcnemar(d1, d2):
    b = int(((d1 == 1) & (d2 == 0)).sum())
    c = int(((d1 == 0) & (d2 == 1)).sum())
    return b, c, float(stats.binomtest(b, b + c, 0.5).pvalue if b + c > 0 else 1.0)


def main():
    nw = 20
    t0 = time.time()
    res = {"design": dict(beta0=B0, beta1=B1, alpha=ALPHA, N0=N0, M=M, MLR=MLR,
                          master=MASTER, thetas=THETAS, ns=NLIST, centre_at=CENTRE_AT),
           "cells": {}, "lrt": {}}
    with ProcessPoolExecutor(max_workers=nw, initializer=_init) as pool:
        _init()
        res["design"]["designs"] = {str(t): _D[t].summary() for t in THETAS}
        for th in THETAS:
            fams = ("vasicek",) + FULL + (("purevar",) if _D[th].feasible_pv else ())
            if abs(th - CENTRE_AT) < 1e-12:
                fams = fams + CENTRE
            for n in NLIST:
                key = "%g|%d" % (th, n)
                sn = np.random.SeedSequence([MASTER, int(th * 1e6), n, 1])
                se_ = np.random.SeedSequence([MASTER, int(th * 1e6), n, 2])
                out = list(pool.map(_block_A, _chunk(th, n, "vasicek", sn, N0, nw)))
                W0 = np.concatenate([o[0] for o in out]); A0 = np.concatenate([o[1] for o in out])
                cW = float(np.percentile(W0, 100 * ALPHA))
                cA = float(np.percentile(A0, 100 * (1 - ALPHA)))
                cell = {"theta": th, "n": n, "cW": cW, "cA": cA,
                        "N0_used": int(W0.size), "families": {}}
                for fam in fams:
                    out = list(pool.map(_block_A, _chunk(th, n, fam, se_, M, nw)))
                    W = np.concatenate([o[0] for o in out]); A = np.concatenate([o[1] for o in out])
                    cc = np.concatenate([o[2] for o in out])
                    b1 = np.concatenate([o[3] for o in out]); s1 = np.concatenate([o[4] for o in out])
                    thh = np.concatenate([o[5] for o in out])
                    d1 = (W < cW).astype(int); d2 = (A > cA).astype(int)
                    b, c, p = mcnemar(d1, d2)
                    lo, hi = b1 - 1.96 * s1, b1 + 1.96 * s1
                    cell["families"][fam] = dict(
                        pow_SW=float(d1.mean()), pow_AD=float(d2.mean()),
                        mcnemar_b=b, mcnemar_c=c, mcnemar_p=p,
                        rej_AD_table=float((A * (1 + 0.6 / n) > 1.321).mean()),
                        conv=float(cc.mean()), M_used=int(W.size),
                        bias_beta1=float(b1.mean() - BETA1_TRUE), sd_beta1=float(b1.std(ddof=1)),
                        cover_beta1=float(((lo <= BETA1_TRUE) & (BETA1_TRUE <= hi)).mean()),
                        mean_theta_hat=float(thh.mean()))
                    f = cell["families"][fam]
                    print("[A] th=%.2f n=%-4d %-10s SW=%.4f AD=%.4f McN=%.1e "
                          "cov(b1)=%.3f bias=%+.4f th^=%.4f conv=%.4f [%.0fs]"
                          % (th, n, fam, f["pow_SW"], f["pow_AD"], p, f["cover_beta1"],
                             f["bias_beta1"], f["mean_theta_hat"], f["conv"], time.time() - t0),
                          flush=True)
                res["cells"][key] = cell

        # ---------------- Phase B ----------------
        for th in THETAS:
            for n in NLIST:
                key = "%g|%d" % (th, n)
                res["lrt"][key] = {}
                for alt in FULL:
                    s0 = np.random.SeedSequence([MASTER, int(th * 1e6), n, 3, len(alt)])
                    s1s = np.random.SeedSequence([MASTER, int(th * 1e6), n, 4, len(alt)])
                    o0 = list(pool.map(_block_B, _chunk(th, n, "vasicek", s0, MLR, nw, extra=(alt,))))
                    d0 = np.concatenate([o[0] for o in o0]); np0 = np.concatenate([o[1] for o in o0])
                    c0 = np.concatenate([o[2] for o in o0])
                    crit = float(np.percentile(d0, 100 * (1 - ALPHA)))
                    critNP = float(np.percentile(np0, 100 * (1 - ALPHA)))
                    o1 = list(pool.map(_block_B, _chunk(th, n, alt, s1s, MLR, nw, extra=(alt,))))
                    d1 = np.concatenate([o[0] for o in o1]); np1 = np.concatenate([o[1] for o in o1])
                    c1 = np.concatenate([o[2] for o in o1])
                    res["lrt"][key][alt] = dict(
                        crit=crit, power=float((d1 > crit).mean()),
                        crit_np=critNP, power_np=float((np1 > critNP).mean()),
                        conv=float(min(c0.mean(), c1.mean())), MLR=MLR)
                    r = res["lrt"][key][alt]
                    print("[B] th=%.2f n=%-4d %-8s LR=%.4f  NP ceiling=%.4f  crit=%.3f conv=%.4f [%.0fs]"
                          % (th, n, alt, r["power"], r["power_np"], crit, r["conv"], time.time() - t0),
                          flush=True)

    res["elapsed_s"] = time.time() - t0
    # one file per theta block, so an interrupted sweep keeps what it finished
    tag = ("th%g" % THETAS[0]) if len(THETAS) == 1 else "all"
    out = "msp2_results_%s.json" % tag if MASTER == 20260812 \
        else "msp2_results_%s_%d.json" % (tag, MASTER)
    json.dump(res, open(out, "w"), indent=1)
    print("\nwritten %s (%.0f s)" % (out, res["elapsed_s"]))


if __name__ == "__main__":
    main()
