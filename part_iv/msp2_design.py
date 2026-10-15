"""Design objects: everything that depends on theta is tabulated once.

Each Design owns, on a grid of the conditional mean:
  v(mu)        the Vasicek variance to be matched
  phi_c, b_c, s_c     dispersions matched at the CENTRE only (scalars)
  phi_f(mu), b_f(mu), s_f(mu)   dispersions matched at EVERY mean
  th_pv(mu)    the Vasicek shape reproducing a competitor's variance function
and a quantile table for the simplex, tabulated on a PROBIT-spaced grid of the
uniform argument so that the tails are resolved on the scale the tests act on.
"""
import numpy as np
from scipy import stats, optimize
import msp2_core as C


class Design:
    def __init__(self, theta, b0=0.0, b1=1.5, nmu=241, nu=2401, zu=8.0, ny=6001):
        self.theta, self.b0, self.b1 = theta, b0, b1
        self.mu_lo = 1 / (1 + np.exp(-b0))
        self.mu_hi = 1 / (1 + np.exp(-(b0 + b1)))
        self.mu_c = 1 / (1 + np.exp(-(b0 + b1 * 0.5)))
        pad = 1e-4
        self.mug = np.linspace(self.mu_lo - pad, self.mu_hi + pad, nmu)
        self.dmu = self.mug[1] - self.mug[0]
        self.vg = C.vas_var(self.mug, theta)
        self.v_c = float(C.vas_var(np.array([self.mu_c]), theta)[0])

        # ---- beta -------------------------------------------------------
        self.phi_f = self.mug * (1 - self.mug) / self.vg - 1.0
        self.phi_c = self.mu_c * (1 - self.mu_c) / self.v_c - 1.0
        self.feasible_beta = bool(np.all(self.phi_f > 0))

        # ---- Kumaraswamy: solve b at each mean --------------------------
        self.bg = np.exp(np.linspace(np.log(0.05), np.log(60.0), 900))
        VK = np.empty((nmu, self.bg.size))
        for k, b in enumerate(self.bg):
            VK[:, k] = C.kw_var(self.mug, b)
        self.b_f = self._invert(VK, self.bg, self.vg)
        self.b_c = self._invert(VK, self.bg, np.full(nmu, self.v_c))[nmu // 2]
        self.feasible_kw = bool(np.all(np.isfinite(self.b_f)))

        # ---- simplex: solve sigma at each mean --------------------------
        # a coarser grid suffices to INVERT the variance; the fine grid is kept
        # for the quantile tables, which is where accuracy is actually needed
        self.sg = np.exp(np.linspace(np.log(0.05), np.log(30.0), 320))
        VS = np.empty((nmu, self.sg.size))
        for k, s in enumerate(self.sg):
            VS[:, k] = C.simplex_var(self.mug, s, 2001)
        self.s_f = self._invert(VS, self.sg, self.vg)
        self.s_c = self._invert(VS, self.sg, np.full(nmu, self.v_c))[nmu // 2]
        self.feasible_smp = bool(np.all(np.isfinite(self.s_f)))

        # ---- pure-variance control: the centre-matched Kumaraswamy's
        #      variance function, reproduced by a genuine Vasicek ----------
        if np.isfinite(self.b_c):
            v_pv = C.kw_var(self.mug, self.b_c)
            self.th_pv = C.theta_for_var(self.mug, v_pv)
            self.feasible_pv = bool(np.all(np.isfinite(self.th_pv)))
            self.v_pv_ratio = v_pv / self.vg
        else:
            self.th_pv, self.feasible_pv, self.v_pv_ratio = None, False, None

        # ---- simplex quantile tables (centre and full) -------------------
        self.zu = np.linspace(-zu, zu, nu)
        self.ug = stats.norm.cdf(self.zu)
        self.yg, _ = C._probit_grid(ny)
        self.Q_f = self._qtable(self.s_f) if self.feasible_smp else None
        self.Q_c = self._qtable(np.full(nmu, self.s_c)) if np.isfinite(self.s_c) else None

    # ------------------------------------------------------------------
    @staticmethod
    def _invert(V, par, target):
        """For each row of V (monotone in par), find par with V = target."""
        out = np.full(V.shape[0], np.nan)
        for i in range(V.shape[0]):
            v = V[i]
            inc = v[-1] > v[0]
            vv, pp = (v, par) if inc else (v[::-1], par[::-1])
            if not (vv[0] <= target[i] <= vv[-1]):
                continue
            out[i] = np.interp(target[i], vv, pp)
        return out

    def _qtable(self, sig):
        """Quantiles in the probit scale, on the probit-spaced u grid."""
        Q = np.empty((self.mug.size, self.ug.size))
        zy = stats.norm.ppf(np.clip(self.yg, 1e-300, 1 - 1e-16))
        for j in range(self.mug.size):
            lf = C.simplex_logpdf(self.yg, self.mug[j], sig[j])
            f = np.nan_to_num(np.exp(lf - np.nanmax(lf)))
            c = np.concatenate([[0.0], np.cumsum(0.5 * (f[1:] + f[:-1]) * np.diff(self.yg))])
            c /= c[-1]
            keep = np.concatenate([[True], np.diff(c) > 0])
            Q[j] = np.interp(self.ug, c[keep], zy[keep])
        return Q

    # ------------------------------------------------------------------
    def _idx(self, mu):
        j = np.clip(((mu - self.mug[0]) / self.dmu).astype(int), 0, self.mug.size - 2)
        return j, (mu - self.mug[j]) / self.dmu

    def _interp(self, tab, mu):
        j, w = self._idx(mu)
        return tab[j] * (1 - w) + tab[j + 1] * w

    def draw(self, fam, match, mu, rng):
        """fam in vasicek/beta/kw/simplex/purevar; match in centre/full."""
        if fam == "vasicek":
            return C.vas_rvs(mu, self.theta, rng)
        if fam == "purevar":
            return C.vas_rvs(mu, self._interp(self.th_pv, mu), rng)
        if fam == "beta":
            phi = np.full_like(mu, self.phi_c) if match == "centre" else self._interp(self.phi_f, mu)
            return rng.beta(mu * phi, (1 - mu) * phi)
        if fam == "kw":
            b = np.full_like(mu, self.b_c) if match == "centre" else self._interp(self.b_f, mu)
            a = np.array([C.kw_a_of(np.array([m]), bb)[0] for m, bb in zip(mu, b)]) \
                if match == "full" else C.kw_a_of(mu, self.b_c)
            u = rng.random(mu.shape)
            return (1 - (1 - u) ** (1.0 / b)) ** (1.0 / a)
        if fam == "simplex":
            Q = self.Q_c if match == "centre" else self.Q_f
            u = rng.random(mu.shape)
            zt = stats.norm.ppf(np.clip(u, 1e-16, 1 - 1e-16))
            p = np.clip((zt - self.zu[0]) / (self.zu[1] - self.zu[0]), 0, self.ug.size - 1.000001)
            iu = p.astype(int)
            wu = p - iu
            j, wm = self._idx(mu)
            q0 = Q[j, iu] * (1 - wu) + Q[j, iu + 1] * wu
            q1 = Q[j + 1, iu] * (1 - wu) + Q[j + 1, iu + 1] * wu
            return stats.norm.cdf(q0 * (1 - wm) + q1 * wm)
        raise ValueError(fam)

    def summary(self):
        r = dict(theta=self.theta, v_centre=self.v_c,
                 phi_c=float(self.phi_c), b_c=float(self.b_c), s_c=float(self.s_c),
                 feasible=dict(beta=self.feasible_beta, kw=self.feasible_kw,
                               simplex=self.feasible_smp, purevar=self.feasible_pv))
        for nm, tab in (("beta", self.phi_f), ("kw", self.b_f), ("simplex", self.s_f)):
            r["range_" + nm] = [float(np.nanmin(tab)), float(np.nanmax(tab))]
        if self.v_pv_ratio is not None:
            r["purevar_var_ratio"] = [float(self.v_pv_ratio.min()), float(self.v_pv_ratio.max())]
        # how far the CENTRE matching is from matching everywhere
        rc = {}
        for nm, cval, fn in (("beta", self.phi_c, lambda p: self.mug * (1 - self.mug) / (1 + p)),
                             ("kw", self.b_c, lambda b: C.kw_var(self.mug, b)),
                             ("simplex", self.s_c, lambda s: C.simplex_var(self.mug, s))):
            if np.isfinite(cval):
                ratio = fn(cval) / self.vg
                rc[nm] = [float(ratio.min()), float(ratio.max())]
        r["centre_var_ratio"] = rc
        return r
