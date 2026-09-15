# Step 2 — the BBM operator in the dilation representation

## The change of variables

Put $u=\log x$ and

$$(U\psi)(u)=e^{u/2}\psi(e^{u}),$$

which is unitary $L^2(\mathbb R_+,\mathrm dx)\to L^2(\mathbb R,\mathrm du)$. Under $U$,

$$A \;=\; 2D \;=\; \hat x\hat p+\hat p\hat x \;=\; -i\Big(x\frac{\mathrm d}{\mathrm dx}+\frac{\mathrm d}{\mathrm dx}x\Big)
\;\longmapsto\; -2i\,\frac{\mathrm d}{\mathrm du}.$$

So $A$ is the generator of dilations; in the Fourier variable conjugate to $u$ (equivalently, on
the Mellin transform along the critical line) it is **multiplication by $2\xi$**: spectrum
$\mathbb R$, simple, purely absolutely continuous. Generalised eigenfunctions:

$$A\,x^{-z}=i(2z-1)\,x^{-z},\qquad
E=i(2z-1)\in\mathbb R \iff \operatorname{Re}z=\tfrac12,\qquad z=\tfrac12-\tfrac{iE}{2}.$$

## Where the BBM Hamiltonian goes

On the completion $\mathcal H_{\eta_0}$ of step 1, $V:\psi\mapsto\Delta\psi$ is unitary onto
$L^2(\mathbb R_+)$ and

$$\boxed{\;V\,H\,V^{-1}=A\;}$$

so **in the dilation representation $H$ is multiplication by the dilation variable.** This is the
"clean spectrum" the construction promises, and it is clean: $\operatorname{spec}H=\mathbb R$,
simple, purely absolutely continuous. There is no point spectrum to find.

A second, equally useful form is the momentum representation. With $v=\log k$ the same
computation gives $A=2i\,\mathrm d/\mathrm dv$, and any metric $\eta=f(\hat p)$ becomes
*multiplication by $g(v)^2$*, $g=f^{1/2}$. That is the form step 3 uses, and it is what makes the
no-go visible: $H_g=g^{-1}Ag=2i\,\mathrm d/\mathrm dv+2i(\log g)'$, a pure gauge along the
dilation flow.

## The eigenfunctions, and the boundary condition

BBM take $\psi_z(x)=-\zeta(z,x+1)$ (Hurwitz zeta). The Hurwitz difference identity
$\zeta(z,a)-\zeta(z,a+1)=a^{-z}$ gives **exactly**

$$\Delta\psi_z=\psi_z(x)-\psi_z(x-1)=x^{-z},$$

verified to $10^{-32}$ in `step2_dilation.py`. Since $\psi_z(0)=-\zeta(z,1)=-\zeta(z)$, the
boundary condition $\psi_z(0)=0$ **is** the equation $\zeta(z)=0$. With $z=\tfrac12+i\gamma$ a
nontrivial zero, the attached eigenvalue is

$$E=i(2z-1)=-2\gamma .$$

That is the whole seduction of the construction, and it is genuine as far as it goes: the
eigenvalue equation and the boundary condition really do select the zeros.

## Why it does not close

$$\|\psi_z\|^2_{\eta_0}=\|\Delta\psi_z\|^2=\int_0^\infty\big|x^{-z}\big|^2\mathrm dx
=\int_0^\infty x^{-2\operatorname{Re}z}\,\mathrm dx=\int_0^\infty\frac{\mathrm dx}{x}=\infty$$

for every $z$ on the critical line. Numerically, $\int_a^b|\Delta\psi_z|^2 = \log(b/a)$ to twelve
digits for $z=\rho_1$. The $\psi_z$ are the **delta-normalised generalised eigenfunctions of an
absolutely continuous spectrum**, not eigenvectors, and they are not elements of the completion
of the very metric that was built to host them.

Two details worth keeping:

* The divergence is only *logarithmic*, and symmetric between $x\to0$ and $x\to\infty$.
  $\operatorname{Re}z=\tfrac12$ is exactly the exponent at which $x^{-z}$ fails to be square
  integrable at both ends simultaneously. The construction fails at the margin, on the critical
  line, which is exactly why it is hard to see.
* $\psi_z$ is not in $L^2$ either: $\psi_z(x)\sim x^{1-z}/(z-1)$, so $|\psi_z|^2\sim x$.
  There is no space in which these are eigenvectors.

## What a finite computation will show you instead

On a dilation box of length $L_u$ the operator $-2i\,\mathrm d/\mathrm du$ with quasi-periodic
boundary conditions has the **arithmetic ladder**

$$E_n=\frac{2(2\pi n+\theta)}{L_u},\qquad\text{spacing } \frac{4\pi}{L_u},$$

reproduced to $10^{-14}$ by the spectral discretisation. The spacing goes to zero only like
$1/\log(\text{box})$, so at any box size a computer can hold, the spectrum *looks* discrete.
This is the trap, and it is the reason step 3 insists on the box-scaling diagnostic rather than
on eyeballing a finite spectrum.

It is also the origin of the coincidence that drives the whole programme: the mean spacing of
Riemann zeros at height $T$ is $2\pi/\log(T/2\pi)$, which in the variable $E=2\gamma$ is
$4\pi/\log(T/2\pi)$ — the same law, matching exactly when the box satisfies $b/a=T/2\pi$. That is
the Berry–Keating phase-space cutoff.
