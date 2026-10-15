"""Which theta values can carry the design, and how far does centre matching
differ from matching everywhere?  Run this before spending compute."""
import json
import numpy as np
from scipy import stats
import msp2_core as C
import msp2_design as D

print("theta grid feasibility, logit(E[Y]) = 1.5 x, x ~ U(0,1)\n")
out = {}
for th in (0.05, 0.10, 0.25, 0.50, 0.75):
    d = D.Design(th)
    s = d.summary()
    out[str(th)] = s
    f = s["feasible"]
    print("theta = %.2f   v(centre) = %.6f" % (th, s["v_centre"]))
    print("   feasible: beta %s  kw %s  simplex %s  purevar %s"
          % (f["beta"], f["kw"], f["simplex"], f["purevar"]))
    print("   dispersion range when matched everywhere:"
          "  phi %s   b %s   sigma %s"
          % (np.round(s["range_beta"], 4), np.round(s["range_kw"], 4),
             np.round(s["range_simplex"], 4)))
    cv = s["centre_var_ratio"]
    print("   variance ratio to the Vasicek under CENTRE matching, over the design:")
    for nm in ("beta", "kw", "simplex"):
        if nm in cv:
            print("       %-11s %.3f to %.3f" % (nm, cv[nm][0], cv[nm][1]))
    if "purevar_var_ratio" in s:
        print("   pure-variance control spans variance ratio %.3f to %.3f"
              % tuple(s["purevar_var_ratio"]))
    print()

json.dump(out, open("feasibility.json", "w"), indent=1)

# ---- does the corrected simplex sampler reproduce its own law? -----------
print("=" * 74)
print("SAMPLER CHECK at theta = 0.25, matched everywhere")
print("=" * 74)
d = D.Design(0.25)
rng = np.random.default_rng(3)
N = 400000
print("%8s %12s %12s %12s %12s %10s" % ("mu", "emp mean", "target", "emp var", "target var", "KS p"))
for mu in (d.mu_lo, d.mu_c, d.mu_hi):
    m = np.full(N, mu)
    for fam in ("simplex", "kw", "beta", "purevar"):
        y = d.draw(fam, "full", m, rng)
        tv = float(C.vas_var(np.array([mu]), 0.25)[0])
        # KS against the law the sampler is meant to have (simplex only)
        p = np.nan
        if fam == "simplex":
            sig = d._interp(d.s_f, m)[0]
            yy = d.yg
            lf = C.simplex_logpdf(yy, mu, sig)
            f = np.nan_to_num(np.exp(lf - np.nanmax(lf)))
            c = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(yy))])
            c /= c[-1]
            sub = y[:20000]
            p = stats.kstest(np.interp(sub, yy, c), "uniform").pvalue
        tag = fam if fam != "purevar" else "purevar*"
        print("%8.4f %12.6f %12.6f %12.8f %12.8f %10s"
              % (mu, y.mean(), mu, y.var(), tv, ("%.3f" % p) if p == p else "-")
              + ("   " + tag))
print("\n* purevar is a Vasicek with a mean-dependent shape: its variance is")
print("  deliberately NOT the target, it reproduces the Kumaraswamy's instead.")

# ---- tails: the defect the previous sampler had --------------------------
print("\n" + "=" * 74)
print("TAIL BEHAVIOUR of the corrected simplex sampler (theta=0.25, centre mean)")
print("=" * 74)
y = d.draw("simplex", "full", np.full(2000000, d.mu_c), rng)
z = stats.norm.ppf(np.clip(y, 1e-300, 1 - 1e-16))
sig = float(d._interp(d.s_f, np.array([d.mu_c]))[0])
yy = d.yg
lf = C.simplex_logpdf(yy, d.mu_c, sig)
f = np.nan_to_num(np.exp(lf - np.nanmax(lf)))
c = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(yy))])
c /= c[-1]
print("   P(|probit(Y)| > 4)      empirical %.3e   true %.3e"
      % ((np.abs(z) > 4).mean(),
         np.interp(stats.norm.cdf(-4), yy, c) + 1 - np.interp(stats.norm.cdf(4), yy, c)))
print("   P(Y < 0.03)             empirical %.3e   true %.3e"
      % ((y < 0.03).mean(), np.interp(0.03, yy, c)))
print("   min probit %.3f   max probit %.3f" % (z.min(), z.max()))
