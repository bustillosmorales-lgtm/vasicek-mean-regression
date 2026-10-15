"""How close the 'full' matching of Part IV gets to the Vasicek variance, and
where the bisected shape of the pure-variance control lies in its interval.
Evaluated at 400 means spread over the design range, off the tabulation nodes."""
import sys, json
import numpy as np
import os
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else
                os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'part_iv'))
import msp2_core as C
import msp2_design as DD

out = {}
for th in (0.05, 0.10, 0.25, 0.50, 0.75):
    d = DD.Design(th, 0.0, 1.5)
    mu = np.linspace(d.mu_lo, d.mu_hi, 400)[1:-1] + 0.37 * d.dmu
    mu = mu[mu < d.mu_hi]
    v = C.vas_var(mu, th)
    phi = d._interp(d.phi_f, mu)
    vb = mu * (1 - mu) / (1 + phi)
    b = d._interp(d.b_f, mu)
    vk = np.array([C.kw_var(np.array([m]), bb)[0] for m, bb in zip(mu, b)])
    s = d._interp(d.s_f, mu)
    vs = np.array([C.simplex_var(np.array([m]), ss, 6001)[0] for m, ss in zip(mu, s)])
    r = dict(beta=float(np.max(np.abs(vb / v - 1))), kw=float(np.max(np.abs(vk / v - 1))),
             simplex=float(np.max(np.abs(vs / v - 1))))
    if d.th_pv is not None:
        r['th_pv_range'] = [float(d.th_pv.min()), float(d.th_pv.max())]
        vpv_target = C.kw_var(d.mug, d.b_c)
        r['pv_relerr_nodes'] = float(np.max(np.abs(C.vas_var(d.mug, d.th_pv) / vpv_target - 1)))
    r['b_f_range'] = [float(np.nanmin(d.b_f)), float(np.nanmax(d.b_f))]
    r['s_f_range'] = [float(np.nanmin(d.s_f)), float(np.nanmax(d.s_f))]
    out[str(th)] = r
    print(th, json.dumps(r))
json.dump(out, open('matching_accuracy.json', 'w'), indent=1)
