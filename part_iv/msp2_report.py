"""Merge the per-theta blocks and build the tables Part IV will carry."""
import glob
import json
import numpy as np

R = {"cells": {}, "lrt": {}, "design": None}
for f in sorted(glob.glob("msp2_results_th*.json")):
    b = json.load(open(f))
    R["cells"].update(b["cells"])
    R["lrt"].update(b["lrt"])
    if R["design"] is None:
        R["design"] = b["design"]
    else:
        R["design"]["designs"].update(b["design"]["designs"])
        R["design"]["thetas"] = sorted(set(R["design"]["thetas"] + b["design"]["thetas"]))
json.dump(R, open("msp2_results.json", "w"), indent=1)

D = R["design"]
M = D["M"]
KEY = lambda s: (float(s.split("|")[0]), int(s.split("|")[1]))
CELLS = sorted(R["cells"], key=KEY)
NAME = {"vasicek": "Vasicek (null)", "beta": "Beta", "kw": "Kumaraswamy",
        "simplex": "Simplex", "purevar": "Vasicek, wrong variance function",
        "beta_c": "Beta, centre only", "kw_c": "Kumaraswamy, centre only",
        "simplex_c": "Simplex, centre only"}
se = lambda p: np.sqrt(p * (1 - p) / M)

print("theta blocks merged: %s" % D["thetas"])
print("N0 = %d, M = %d, MLR = %d, master = %d\n" % (D["N0"], D["M"], D["MLR"], D["master"]))

# ---------------------------------------------------------------- TABLE A
print("=" * 104)
print("TABLE A  size and power of the two tests at the 5%% level, alternatives matched in the")
print("         variance at EVERY observation, so that only the density shape differs")
print("=" * 104)
print("%6s %5s %-32s %8s %8s %11s %12s" %
      ("theta", "n", "generating law", "SW", "AD", "McNemar", "AD, tabulated"))
for k in CELLS:
    c = R["cells"][k]
    for fam in ("vasicek", "beta", "kw", "simplex", "purevar"):
        if fam not in c["families"]:
            continue
        f = c["families"][fam]
        print("%6.2f %5d %-32s %8.4f %8.4f %11.1e %12.4f"
              % (c["theta"], c["n"], NAME[fam], f["pow_SW"], f["pow_AD"],
                 f["mcnemar_p"], f["rej_AD_table"]))
    print()

# ---------------------------------------------------------------- TABLE B
print("=" * 104)
print("TABLE B  how much of the power is the variance function, not the shape")
print("         (centre matching against matching everywhere, theta = %g)" % D["centre_at"])
print("=" * 104)
print("%5s %-12s %10s %10s %10s %10s %9s" %
      ("n", "family", "SW centre", "SW full", "excess c.", "excess f.", "share"))
for k in CELLS:
    c = R["cells"][k]
    if "beta_c" not in c["families"]:
        continue
    sz = c["families"]["vasicek"]["pow_SW"]
    for fam in ("beta", "kw", "simplex"):
        a = c["families"][fam + "_c"]["pow_SW"]
        b = c["families"][fam]["pow_SW"]
        ea, eb = a - sz, b - sz
        sh = (1 - eb / ea) if ea > 0 else np.nan
        print("%5d %-12s %10.4f %10.4f %10.4f %10.4f %8.1f%%"
              % (c["n"], NAME[fam], a, b, ea, eb, 100 * sh))
print("\n'excess' is the rejection rate above the size of the same cell; 'share' is the")
print("fraction of the centre-matched excess that disappears once the variance function")
print("is matched everywhere, i.e. the part that was never about the shape.")

# ---------------------------------------------------------------- TABLE C
print("\n" + "=" * 104)
print("TABLE C  what the undetected misspecification costs the fitted model")
print("=" * 104)
print("%6s %5s %-32s %9s %9s %9s %9s %9s" %
      ("theta", "n", "generating law", "bias b1", "sd b1", "cover", "theta hat", "power SW"))
for k in CELLS:
    c = R["cells"][k]
    for fam in ("vasicek", "beta", "kw", "simplex"):
        if fam not in c["families"]:
            continue
        f = c["families"][fam]
        print("%6.2f %5d %-32s %+9.4f %9.4f %9.4f %9.4f %9.4f"
              % (c["theta"], c["n"], NAME[fam], f["bias_beta1"], f["sd_beta1"],
                 f["cover_beta1"], f["mean_theta_hat"], f["pow_SW"]))
    print()

# ---------------------------------------------------------------- TABLE D
print("=" * 104)
print("TABLE D  the two tests against a likelihood ratio benchmark and against the exact")
print("         Neyman-Pearson ceiling, which no procedure can beat")
print("=" * 104)
print("%6s %5s %-12s %10s %10s %10s %12s %12s" %
      ("theta", "n", "alternative", "best test", "LR", "NP ceiling", "test/ceiling", "LR/ceiling"))
rows = []
for k in sorted(R["lrt"], key=KEY):
    c = R["cells"][k]
    for alt in ("beta", "kw", "simplex"):
        if alt not in R["lrt"][k]:
            continue
        r = R["lrt"][k][alt]
        f = c["families"][alt]
        best = max(f["pow_SW"], f["pow_AD"])
        rows.append((c["theta"], c["n"], alt, best, r["power"], r["power_np"]))
        print("%6.2f %5d %-12s %10.4f %10.4f %10.4f %12.2f %12.2f"
              % (c["theta"], c["n"], NAME[alt], best, r["power"], r["power_np"],
                 best / r["power_np"] if r["power_np"] > 0 else np.nan,
                 r["power"] / r["power_np"] if r["power_np"] > 0 else np.nan))
sh = [b / c for _, _, _, b, _, c in rows if c > 0]
print("\nshare of the available signal recovered by the better test: %.2f to %.2f"
      % (min(sh), max(sh)))

# ---------------------------------------------------------------- checks
print("\n" + "=" * 104)
print("CHECKS")
print("=" * 104)
conv = [R["cells"][k]["families"][f]["conv"] for k in CELLS for f in R["cells"][k]["families"]]
print("  minimum convergence over the %d cells of Table A: %.4f" % (len(conv), min(conv)))
lc = [R["lrt"][k][a]["conv"] for k in R["lrt"] for a in R["lrt"][k]]
print("  minimum convergence over the %d cells of the benchmark: %.4f" % (len(lc), min(lc)))
sizes = [(R["cells"][k]["families"]["vasicek"]["pow_SW"],
          R["cells"][k]["families"]["vasicek"]["pow_AD"]) for k in CELLS]
z = [abs(s - 0.05) / se(s) for pair in sizes for s in pair]
print("  empirical size: %.4f to %.4f, worst deviation %.2f simulation errors"
      % (min(min(p) for p in sizes), max(max(p) for p in sizes), max(z)))
pv = [R["cells"][k]["families"]["purevar"]["pow_SW"] for k in CELLS
      if "purevar" in R["cells"][k]["families"]]
pvA = [R["cells"][k]["families"]["purevar"]["pow_AD"] for k in CELLS
       if "purevar" in R["cells"][k]["families"]]
print("  pure-variance control (no shape difference at all): SW %.4f to %.4f, AD %.4f to %.4f"
      % (min(pv), max(pv), min(pvA), max(pvA)))
adt = [R["cells"][k]["families"]["vasicek"]["rej_AD_table"] for k in CELLS]
print("  AD read against its tabulated critical value, under the null: %.4f to %.4f"
      % (min(adt), max(adt)))
