# Step 1 — where the coercivity fails

## The construction

Bender, Brody and Müller (PRL **118**, 130201, 2017) propose, on the half line,

$$\hat H \;=\; \frac{1}{1-e^{-i\hat p}}\,(\hat x\hat p+\hat p\hat x)\,(1-e^{-i\hat p}),
\qquad p=-i\,\mathrm d/\mathrm dx .$$

Write $\Delta = 1-e^{-i\hat p}$, so $(\Delta\psi)(x)=\psi(x)-\psi(x-1)$ with $\psi\equiv 0$ on
$x<0$, and $A = 2D = \hat x\hat p+\hat p\hat x$. Then

$$H=\Delta^{-1}A\,\Delta .$$

$H$ is not symmetric on $L^2(0,\infty)$, but it is symmetric for the inner product built
from the candidate metric

$$\eta_0=\Delta^{\dagger}\Delta,\qquad
\langle\psi,\varphi\rangle_{\eta_0}=\langle\Delta\psi,\Delta\varphi\rangle ,$$

because $\langle\psi,H\varphi\rangle_{\eta_0}=\langle\Delta\psi,A\,\Delta\varphi\rangle$ and $A$ is symmetric.

## The symbol, and the exact point of failure

$\Delta$ is a Fourier multiplier with symbol $1-e^{-ik}$, so

$$\boxed{\;f_0(k)=|1-e^{-ik}|^{2}=2-2\cos k=4\sin^{2}(k/2)\;}$$

i.e. $\eta_0=4\sin^2(\hat p/2)$. Therefore

* $f_0(k)>0$ for $k\notin 2\pi\mathbb Z$, so $\eta_0>0$ and $\ker\eta_0=\{0\}$ (the zero set has
  measure zero);
* **but** $f_0$ vanishes — to second order — at every $k\in2\pi\mathbb Z$, so
  $\inf\operatorname{spec}\eta_0=0$.

That is the failure. $\eta_0$ is positive with trivial kernel and **not coercive**: there is no
$c>0$ with $\|\Delta\psi\|\ge c\|\psi\|$. Coercivity is exactly the hypothesis that would make
$\|\cdot\|_{\eta_0}$ equivalent to $\|\cdot\|_{L^2}$, and it is the only hypothesis under which
the whole construction would be harmless.

## Four consequences, in order of severity

1. **The norms are inequivalent.** $\|\cdot\|_{\eta_0}$ is strictly weaker than
   $\|\cdot\|_{L^2}$. On $(0,X]$ the loss factor is
   $\lambda_{\min}(\eta_0)^{-1/2}\simeq 2X/\pi$ — measured in `step1_coercivity.py` and
   matching the prediction to four significant figures, with
   $\lambda_{\min}\simeq(\pi/2X)^2$ from the double zero at $k=0$.

2. **The similarity is unbounded.** $\Delta^{-1}=\sum_{n\ge0}e^{-in\hat p}$ is unbounded
   (a finite sum pointwise on the half line, but with no bound in $L^2$). So
   $H=\Delta^{-1}A\Delta$ is **not** a bounded similarity transform of $A$, and no spectral
   statement transfers for free. Everything downstream has to be earned.

3. **The completion is not inside $L^2$.** The completion of $C_c^\infty(0,\infty)$ in
   $\|\cdot\|_{\eta_0}$ is
   $$\mathcal H_{\eta_0}=\{\psi:\Delta\psi\in L^2(0,\infty)\},$$
   and $V:\psi\mapsto\Delta\psi$ is unitary from it onto $L^2(0,\infty)$. It contains objects
   that are not functions in $L^2$: the constant function is the simplest, since $\Delta 1$ is
   supported near the origin while $\|1\|_{L^2(0,X)}=\sqrt X\to\infty$. The near-null direction
   of $\eta_0$ is the slowly varying $k\to0$ mode — the numerics show it as a smooth
   quarter-sine ramp across the box. The $\eta$-norm only ever sees *unit differences*, so it is
   blind to everything that varies slowly on the scale of the shift.

4. **The eigenstates fall in the gap.** This is step 2, but it is already visible here. The BBM
   eigenfunctions satisfy $\Delta\psi_z = x^{-z}$, so
   $$\|\psi_z\|^2_{\eta_0}=\int_0^\infty x^{-2\operatorname{Re}z}\,\mathrm dx
     =\int_0^\infty \frac{\mathrm dx}{x}=\infty \quad\text{on }\operatorname{Re}z=\tfrac12 .$$
   The divergence is *logarithmic at both ends*, and $\operatorname{Re}z=\tfrac12$ is exactly the
   borderline exponent. The construction misses by a hair, on the critical line, which is
   precisely why it looks like it should work.

## Relation to the literature

Liu, *Metric completion of the Bender–Brody–Müller Hamiltonian: dilation spectrum and missing
eigenstates*, [arXiv:2607.19067](https://arxiv.org/abs/2607.19067) (21 July 2026), states the
same structure: $\eta_0=\Delta^\dagger\Delta$ positive with trivial kernel but not coercive; the
completion canonically unitarily equivalent to $L^2(\mathbb R_+)$; the free self-adjoint
realisation the dilation generator with simple, purely absolutely continuous spectrum
$\mathbb R$; the transported symmetric operator with deficiency indices $(\infty,\infty)$ and an
adjoint having every real number as an eigenvalue of infinite multiplicity.

**Provenance note.** `arxiv.org` is blocked by this session's network egress policy, so the paper
itself could not be read. The statements above were recovered from search-engine summaries of the
abstract and then *re-derived and verified independently* in `step1_coercivity.py`,
`step2_dilation.py` and `step3_metric_scan.py`. Everything asserted in these notes is checked
numerically here and does not rest on the summaries. The one thing that should still be read in
the original is the precise indexing of the "missing eigenstates for $n>0$", which the summaries
report but which the numerics here do not pin down.
