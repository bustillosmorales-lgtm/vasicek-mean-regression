# Metric completion of the BBM Hamiltonian — does the spectrum discretise?

Working through the four steps, with the numerics carrying the weight in step 3.

Reference under examination: Kejun Liu, *Metric completion of the Bender–Brody–Müller
Hamiltonian: dilation spectrum and missing eigenstates*,
[arXiv:2607.19067](https://arxiv.org/abs/2607.19067) (21 July 2026). The original of
Bender–Brody–Müller is [PRL 118, 130201 (2017)](https://link.aps.org/doi/10.1103/PhysRevLett.118.130201) /
[arXiv:1608.03679](https://arxiv.org/abs/1608.03679).

> **Provenance.** `arxiv.org` is blocked by this session's network egress policy, so the 2026
> paper could not be read directly. Its statements were recovered from search-engine summaries
> of the abstract and then re-derived and verified independently here. Every claim in these notes
> is checked by the scripts; none rests on the summaries. See the note at the end of
> `notes/step1_coercivity.md` for the one item that still needs the original.

## Short answer

**No. The spectrum does not discretise, and no choice of metric can make it.**

In the dilation representation the metric is a pure gauge factor along the dilation flow: it
cancels out of the eigenvalue equation *and* out of the normalisation. It can only change which
states are normalisable, never where the spectrum is.

One construction does produce a discrete spectrum — a particular self-adjoint extension made
possible by the zeros of the BBM symbol, plus an infrared cutoff — and it even carries the
correct $E\log E$ Weyl law for free. It still is not the zeros: its levels are free parameters,
and its level statistics are the opposite of what the zeros show.

## Running it

```
python bbm/step1_coercivity.py      # where coercivity fails, quantitatively
python bbm/step2_dilation.py        # the dilation representation; Hurwitz eigenstates (mpmath)
python bbm/step3_metric_scan.py     # the metric scan — the main experiment
python bbm/step4_zeta_compare.py    # comparison with the first 500 zeros
```

Needs `numpy`, `scipy`, `mpmath`. Saved output is in `results_step*.txt`; the analytic write-ups
of steps 1 and 2 are in `notes/`.

## Step 1 — where coercivity fails

$\Delta=1-e^{-i\hat p}$ has symbol $1-e^{-ik}$, so the candidate metric is the Fourier multiplier

$$\eta_0=\Delta^\dagger\Delta=4\sin^2(\hat p/2).$$

The symbol has a **double zero at every $k\in 2\pi\mathbb Z$**. Hence $\eta_0>0$ with trivial
kernel but $\inf\operatorname{spec}\eta_0=0$: positive, not coercive. Measured on $(0,X]$:

| $X$ | $\lambda_{\min}(\eta_0)$ | $(\pi/2X)^2$ | ratio | $\max\|\psi\|/\|\Delta\psi\|$ | $2X/\pi$ |
|---|---|---|---|---|---|
| 32 | 2.3355e-03 | 2.4096e-03 | 0.969 | 20.69 | 20.37 |
| 128 | 1.4943e-04 | 1.5060e-04 | 0.992 | 81.81 | 81.49 |
| 512 | 9.3940e-06 | 9.4124e-06 | 0.998 | 326.27 | 325.95 |

The near-null direction is the slowly varying $k\to0$ mode: the $\eta$-norm only sees unit
differences, so it is blind to anything varying slowly on the scale of the shift. Consequently
the completion $\mathcal H_{\eta_0}=\{\psi:\Delta\psi\in L^2\}$ is **not** a subspace of $L^2$ —
the constant function is already in it — and $\Delta^{-1}=\sum_{n\ge0}e^{-in\hat p}$ is unbounded,
so $H=\Delta^{-1}A\Delta$ is not a bounded similarity transform of the dilation generator.

## Step 2 — the BBM operator in the dilation representation

With $u=\log x$ and $(U\psi)(u)=e^{u/2}\psi(e^u)$, the symmetrised $A=2D=\hat x\hat p+\hat p\hat x$
becomes $-2i\,\mathrm d/\mathrm du$, and on the completion $V:\psi\mapsto\Delta\psi$ is unitary
with $VHV^{-1}=A$. So **$H$ is multiplication by the dilation variable**: spectrum $\mathbb R$,
simple, purely absolutely continuous.

Verified numerically:

* $A\,x^{-z}=E\,x^{-z}$ with $E=i(2z-1)$, residual $\sim10^{-13}$;
* the box spectrum is the ladder $4\pi n/L_u$, to $10^{-14}$;
* $\Delta\psi_z=x^{-z}$ for $\psi_z=-\zeta(z,\cdot+1)$, to $10^{-32}$ (mpmath, 30 dps);
* $\psi_z(0)=-\zeta(z)$, so the BBM boundary condition *is* $\zeta(z)=0$, giving $E=-2\gamma$;
* $\|\psi_z\|^2_{\eta_0}=\int_a^b\mathrm dx/x=\log(b/a)$ to twelve digits — **divergent**.

The BBM eigenstates are the delta-normalised generalised eigenfunctions of an absolutely
continuous spectrum. The divergence is only logarithmic, and $\operatorname{Re}z=\tfrac12$ is
exactly the borderline exponent: the construction misses by a hair, on the critical line.

## Step 3 — alternative metrics (the main experiment)

Momentum representation, $v=\log k$: $A=2i\,\mathrm d/\mathrm dv$ and any metric $\eta=f(k)$ is
multiplication by $g(v)^2$, $g=f^{1/2}$. Then $H_g=g^{-1}Ag$ and, for **every** $g$,

$$\psi_E(v)=g(v)^{-1}e^{-iEv/2},\qquad \|\psi_E\|_{\eta}=\|e^{-iEv/2}\| .$$

**A. Similarity invariance.** Eight metrics — BBM, a coercive regularisation, $k^2$, $k^4$,
$1+k^2$, $e^{-1/k^2}$, $4\sin^2(k/2)(1+k^2)$, trivial — condition numbers from $1$ to $1.6\times10^5$:
$\max|\operatorname{spec}H_g-\operatorname{spec}A| \le 1.4\times10^{-11}$ in every case, imaginary
parts $\le10^{-11}$. The predicted eigenfunctions give residuals $\sim10^{-12}$ and an
$\eta$-norm of $\sqrt{L_v}$ **independent of the metric**.

**B. The singular limit.** $g=2|\sin(k/2)|$ vanishes at $k=2\pi n$. Functions blowing up like
$1/g$ there still have finite $\eta$-norm, so each interval $(2\pi n,2\pi(n+1))$ supports its own
solutions for *every* real $E$ — verified: $\|\psi_E\|^2_\eta=\log(1+1/n)$ exactly. That is the
deficiency $(\infty,\infty)$ and the adjoint with every real eigenvalue of infinite multiplicity.

The decoupled extension, above an infrared cutoff, **is genuinely discrete**, and

$$N_{\text{dec}}(E)=\frac{E}{4\pi}\Big[\log\frac{E}{4\pi}-1+\gamma\Big]+o(E),
\qquad
N_{\text{Riemann}}(E)=\frac{E}{4\pi}\Big[\log\frac{E}{4\pi}-1\Big]+\tfrac78 .$$

The leading $E\log E$ term is exactly right; the excess density is the Euler–Mascheroni constant.
Measured $(N_{\text{dec}}-N_R)/E \to 0.045930$ against $\gamma/4\pi=0.045933$. The Berry–Keating
counting law falls out of nothing but the arithmetic of the zero set $k=2\pi n$ — and each
interval carries a free extension phase $\theta_n$, so the levels are parameters, not outputs.

**C. Box scaling.** The only honest discreteness test: every level slides like $4\pi/\log(\text{box})$,
identically for all metrics, over sixteen orders of magnitude in box size. Nothing converges.

**Control.** A metric diagonal in position, $\eta=x^{2a}$, gives $H_w=-2i\,\mathrm d/\mathrm du-2ia$:
it translates the spectrum off the real axis rather than confining it — verified pointwise to
$10^{-11}$. It is not a metric for $H$ at all.

## Step 4 — against the first 500 zeros

Two candidates survive step 3: **M1**, the free extension in a Berry–Keating box tuned to height
$T$ (arithmetic ladder); **M2**, the decoupled extension (correct Weyl law). Both reproduce a
counting function — both were built to. The spacing statistics decide:

| spectrum | #s | var | $P(s<0.3)$ | KS vs GUE | KS vs Poisson |
|---|---|---|---|---|---|
| Riemann zeros | 499 | 0.1397 | 0.0120 | 0.0497 (p = 0.16) | 0.3312 (p = 3e-49) |
| M1 Berry–Keating ladder | 400 | 0.0000 | 0.0000 | 0.5331 (p = 1e-106) | 0.6321 (p = 2e-155) |
| M2 decoupled extension | 20084 | 1.2247 | 0.3781 | 0.3655 (p ≈ 0) | 0.1454 (p ≈ 0) |
| *GUE* | | 0.1800 | 0.0134 | | |
| *Poisson* | | 1.0000 | 0.2592 | | |
| *picket fence* | | 0.0000 | 0.0000 | | |

The zeros are consistent with GUE and overwhelmingly reject Poisson. The two candidates sit at
the opposite extremes: M1 is a perfectly rigid picket fence, M2 has *less* rigidity than Poisson
(commensurable ladders pile levels up). Neither reproduces a single zero — M1 tuned at its own
height gets positions to within 0.1% purely because the density agrees, while its spacings are
identical to the last digit and the zeros' vary by a factor of several.

## What this leaves

The obstruction is structural, not technical. In the dilation representation the metric is a
gauge factor that drops out of the eigenvalue equation, so discreteness — a statement about
compactness of the resolvent — is untouched by it. What does control the level density is the
*length of the dilation box*, i.e. the phase-space cutoff, and matching a density is cheap.

Anything that is going to produce the zeros has to supply their **correlations**, and none of the
mechanisms here does. The natural next targets are the ones that break the arithmetic ladder
rather than re-weight it: a genuine potential on the dilation flow, or a boundary condition
coupling the intervals instead of decoupling them. Whether any of those can be made
self-adjoint with the right Weyl law is open, and is not a question about metrics.
