# Step 5 — rigidity: no choice of extension phases works

## The loophole step 3 left open

The decoupled self-adjoint extension of the BBM operator is genuinely discrete and carries the
correct $E\log E$ Weyl law. But each interval $I_n=(2\pi n,2\pi(n+1))$ carries its own extension
phase $\theta_n\in[0,2\pi)$ — a first-order operator on an interval has exactly the $U(1)$ family
of quasi-periodic conditions $\psi(b)=e^{i\theta}\psi(a)$ — so the levels looked like free
parameters. With infinitely many of them, could one fit the zeros?

In the zero variable $\gamma=E/2$, interval $n$ contributes the arithmetic progression

$$\gamma^{(n)}_m=\frac{2\pi m+\theta_n}{L_n},\qquad L_n=\log\Big(1+\frac1n\Big),
\qquad s_n=\frac{2\pi}{L_n}.$$

## Obstruction I — density, uniform in $\theta$

Write $c=T/2\pi$. The levels of ladder $n$ in $(0,T]$ number

* $\theta_n=0$: $\;\lfloor cL_n\rfloor$;
* $0<\theta_n<2\pi$: $\;\lfloor cL_n-\theta_n/2\pi\rfloor+1\;\ge\;\lfloor cL_n\rfloor$.

So $\lfloor cL_n\rfloor$ is a lower bound **for every phase**, and therefore

$$N_\theta(T)\;\ge\;\sum_{n\ge1}\Big\lfloor \frac{TL_n}{2\pi}\Big\rfloor
\qquad\text{for every }\{\theta_n\}.$$

Now evaluate the bound. Since $\sum_{n\ge1}\big[\tfrac1n-\log(1+\tfrac1n)\big]=\gamma$ and the
Dirichlet divisor summatory function satisfies
$D(c)=\sum_{n\le c}\lfloor c/n\rfloor=c\log c+(2\gamma-1)c+O(\sqrt c)$,

$$\sum_{n\ge1}\lfloor cL_n\rfloor \;=\; D(c)-\gamma c+o(c) \;=\; c\log c+(\gamma-1)c+o(c).$$

Against $N_{\text{Riemann}}(T)=c\log c-c+\tfrac78$:

$$\boxed{\;N_\theta(T)\;\ge\;N_{\text{Riemann}}(T)+\frac{\gamma}{2\pi}\,T+o(T)
\quad\text{for every }\{\theta_n\}.\;}$$

Numerically, $\big(\sum\lfloor cL_n\rfloor-N_R\big)/c\to0.577231$ at $c=10^6$ against
$\gamma=0.577216$, and the $\theta$-uniformity was checked against 500 random phase vectors at
three values of $c$.

**The construction carries at least $(\gamma/2\pi)T\approx0.0919\,T$ levels below height $T$ that
are not zeros, whatever the phases.** It cannot be made into the zero set.

The identity is worth stating on its own: the counting function of this spectrum is the Dirichlet
divisor summatory function in disguise. What the construction knows about is **divisors**, not
zeros — which is what one should have expected from a zero set $k\in2\pi\mathbb Z$ whose interval
lengths are $\log(1+1/n)$.

## Obstruction II — rigidity

Independent, and much stronger in practice. Each ladder is an arithmetic progression; the zeros
are not. Optimising every phase independently (legitimate, since the ladders are decoupled, so the
joint optimum factorises), the minimum number of levels that cannot be placed within $\varepsilon$
of any zero is:

| $T$ | zeros $\le T$ | $\varepsilon$ | min levels | matched | spurious | density bound $\gamma c$ | spurious/$T$ |
|---|---|---|---|---|---|---|---|
| 143.11 | 50 | 0.02 | 66 | 28 | 38 | 13.1 | 0.266 |
| 236.52 | 100 | 0.02 | 127 | 50 | 77 | 21.7 | 0.326 |
| 396.38 | 200 | 0.02 | 247 | 96 | 151 | 36.4 | 0.381 |
| 811.18 | 500 | 0.02 | 605 | 221 | 384 | 74.5 | 0.473 |
| 811.18 | 500 | 0.10 | 640 | 370 | 270 | 74.5 | 0.333 |

Loosening $\varepsilon$ by a factor of five barely helps. A ladder that fires $k$ times in $(0,T]$
aligns with at most a couple of zeros no matter where it starts, so the spurious count runs about
five times the density bound.

## Scope — stated honestly

This covers the **decoupled** family: all self-adjoint extensions that do not couple the
intervals, i.e. one phase per interval. Extensions that **couple** the intervals form a strictly
larger family — the quantum-graph setting, where the self-adjoint extensions of the Berry–Keating
operator have already been completely classified together with their secular equation, trace
formula and Weyl asymptotics. The bound above does not apply to them.

Whether an analogous obstruction holds for the coupled family is **open**, and it is the only
place in this line of work where a genuinely new theorem could live.
