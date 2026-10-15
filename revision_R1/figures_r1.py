"""Figures of the revision.
fig_mean_*.pdf, fig_var_mean.pdf : Figure 1 redrawn with the density f(y; alpha, theta)
    itself on the vertical axis, and the panels in increasing order of alpha.
fig_oecd_data.pdf : the data of the application and the two fitted models."""
import csv, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = sys.argv[1]
plt.rcParams.update({'font.family': 'serif', 'font.size': 11, 'axes.spines.top': False,
                     'axes.spines.right': False, 'pdf.fonttype': 42})
THETAS = (0.10, 0.25, 0.50, 0.75, 0.90)
COLS = ('black', '#DF536B', '#61D04F', '#2297E6', '#28E2E5')


def vas_pdf(y, a, t):
    z = stats.norm.ppf(y)
    return np.sqrt((1 - t) / t) * np.exp(0.5 * (z ** 2 - ((np.sqrt(1 - t) * z - stats.norm.ppf(a)) / np.sqrt(t)) ** 2))


def vas_var(a, t):
    q = stats.norm.ppf(a)
    return np.array([stats.multivariate_normal(mean=[0, 0], cov=[[1, t], [t, 1]]).cdf([qq, qq]) for qq in q]) - a ** 2


y = np.linspace(1e-4, 1 - 1e-4, 4001)
for k, a in enumerate((0.10, 0.25, 0.50, 0.75, 0.90), start=1):
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    for t, c in zip(THETAS, COLS):
        ax.plot(y, vas_pdf(y, a, t), color=c, lw=1.4, label=r'$\theta=%.2f$' % t)
    ax.set_ylim(0, 6)
    ax.set_xlim(0, 1)
    ax.set_xlabel(r'$y$')
    ax.set_ylabel(r'$f(y;\alpha,\theta)$, $\alpha=%.2f$' % a)
    ax.grid(ls=':', lw=0.5)
    if k == 1:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig('%s/fig_mean_%d.pdf' % (OUT, k))
    plt.close(fig)

aa = np.linspace(0.001, 0.999, 300)
fig, ax = plt.subplots(figsize=(3.4, 3.2))
for t, c in zip(THETAS, COLS):
    ax.plot(aa, vas_var(aa, t), color=c, lw=1.4)
ax.set_xlim(0, 1)
ax.set_ylim(0, 0.2)
ax.set_xlabel(r'$\alpha$')
ax.set_ylabel(r'Var$(Y)$')
ax.grid(ls=':', lw=0.5)
fig.tight_layout()
fig.savefig('%s/fig_var_mean.pdf' % OUT)
plt.close(fig)

# ---------------------------------------------------------------- application
rows = list(csv.DictReader(open(sys.argv[2], encoding='utf-8')))
g = lambda c: np.array([float(r[c]) for r in rows])
yo, mb, mv, rb, rv, sb, tv = (g(c) for c in ('y', 'mu_beta', 'mu_vas', 'rq_beta', 'rq_vas', 'sigma_beta', 'theta_vas'))
phi = 1 / sb ** 2 - 1
grid = np.linspace(0.2, 0.999, 1500)
fb = np.mean([stats.beta.pdf(grid, m * p, (1 - m) * p) for m, p in zip(mb, phi)], axis=0)
fv = np.mean([vas_pdf(grid, m, t) for m, t in zip(mv, tv)], axis=0)

fig, axs = plt.subplots(1, 3, figsize=(10.5, 3.3))
ax = axs[0]
ax.hist(yo, bins=np.arange(0.3, 1.001, 0.05), density=True, color='0.85', edgecolor='0.4', lw=0.6)
ax.plot(grid, fb, color='#2297E6', lw=1.5, ls='--', label='Beta')
ax.plot(grid, fv, color='#DF536B', lw=1.5, label='Vasicek')
ax.set_xlabel('Educational attainment')
ax.set_ylabel('Density')
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.set_title('(a)', fontsize=11)
ax = axs[1]
ax.plot([0.3, 1], [0.3, 1], color='0.6', lw=0.8)
ax.scatter(mb, yo, s=16, facecolors='none', edgecolors='#2297E6', label='Beta')
ax.scatter(mv, yo, s=14, marker='x', color='#DF536B', lw=0.9, label='Vasicek')
ax.set_xlabel('Fitted mean')
ax.set_ylabel('Observed')
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.set_title('(b)', fontsize=11)
ax = axs[2]
q = stats.norm.ppf((np.arange(1, len(yo) + 1) - 0.375) / (len(yo) + 0.25))
ax.plot([-2.6, 2.6], [-2.6, 2.6], color='0.6', lw=0.8)
ax.scatter(q, np.sort(rb), s=16, facecolors='none', edgecolors='#2297E6', label='Beta')
ax.scatter(q, np.sort(rv), s=14, marker='x', color='#DF536B', lw=0.9, label='Vasicek')
ax.set_xlabel('Standard normal quantile')
ax.set_ylabel('Quantile residual')
ax.legend(frameon=False, fontsize=9, loc='upper left')
ax.set_title('(c)', fontsize=11)
for ax in axs:
    ax.grid(ls=':', lw=0.5)
fig.tight_layout()
fig.savefig('%s/fig_oecd_data.pdf' % OUT)
fig.savefig('%s/fig_oecd_data.png' % OUT, dpi=110)
print('ok', yo.min(), yo.max(), np.trapezoid(fv, grid), np.trapezoid(fb, grid))
