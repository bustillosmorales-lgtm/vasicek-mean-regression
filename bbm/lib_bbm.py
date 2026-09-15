"""Shared numerics for the Bender-Brody-Muller metric-completion study.

Conventions fixed once and used everywhere:

    p = -i d/dx                       momentum on the half line
    A = 2D = x p + p x = -i(x d/dx + d/dx x)      dilation generator (Hermitian)
    Delta = 1 - exp(-i p)             (Delta psi)(x) = psi(x) - psi(x-1)
    H = Delta^{-1} A Delta            the BBM Hamiltonian
    eta_0 = Delta^dagger Delta        the BBM candidate metric, symbol 4 sin^2(k/2)

Generalised eigenfunctions of A:  A x^{-z} = i(2z-1) x^{-z}, so the eigenvalue
E is real exactly when Re z = 1/2, with z = 1/2 - iE/2.
BBM eigenfunctions: psi_z(x) = -zeta(z, x+1), which obey Delta psi_z = x^{-z}.
"""

import numpy as np

# --------------------------------------------------------------------------
# x representation: uniform grid with spacing h = 1/m, so the unit shift that
# defines Delta is an exact index shift (no interpolation error anywhere).
# --------------------------------------------------------------------------

def x_grid(n_cells, m):
    """Uniform grid on (0, X] with X = n_cells and spacing h = 1/m."""
    N = n_cells * m
    h = 1.0 / m
    return h * np.arange(1, N + 1), h


def shift_matrix(N, m):
    """(S psi)_j = psi_{j-m}, i.e. psi(x-1), with psi = 0 for x <= 0."""
    S = np.zeros((N, N))
    if m < N:
        S[m:, :N - m] = np.eye(N - m)
    return S


def delta_matrix(N, m):
    """Delta = 1 - exp(-ip) on the grid.  Unit lower triangular => invertible."""
    return np.eye(N) - shift_matrix(N, m)


def first_derivative(N, h):
    """Centred difference with Dirichlet ends.  Real and antisymmetric."""
    d = np.zeros((N, N))
    i = np.arange(N - 1)
    d[i, i + 1] = 0.5 / h
    d[i + 1, i] = -0.5 / h
    return d


def dilation_matrix(x, h):
    """A = 2D = -i(x d/dx + d/dx x).  Exactly Hermitian on the grid."""
    d = first_derivative(x.size, h)
    X = np.diag(x)
    return -1j * (X @ d + d @ X)


# --------------------------------------------------------------------------
# Dilation representation.  u = log x, (U psi)(u) = e^{u/2} psi(e^u) is unitary
# L^2(R_+, dx) -> L^2(R, du) and carries A = 2D into -2i d/du.
# --------------------------------------------------------------------------

def u_grid(u0, u1, n):
    """Uniform periodic grid on [u0, u1)."""
    return np.linspace(u0, u1, n, endpoint=False), (u1 - u0) / n


def spectral_derivative_matrix(n, L):
    """d/du on a periodic grid of n points and period L, Fourier spectral."""
    j = np.arange(n)
    xi = 2.0 * np.pi * np.fft.fftfreq(n, d=L / n)          # Fourier wavenumbers
    F = np.exp(-2j * np.pi * np.outer(j, j) / n)           # DFT
    Finv = np.conj(F) / n
    return np.real_if_close(Finv @ (np.diag(1j * xi) @ F), tol=1e6)


def dilation_matrix_u(n, L):
    """A = -2i d/du on a periodic u-box of length L.  Hermitian, spectral."""
    Dm = spectral_derivative_matrix(n, L).astype(complex)
    A = -2j * Dm
    return 0.5 * (A + A.conj().T)


def free_ladder(L, n_levels, theta=0.0):
    """Spectrum of -2i d/du with psi(u+L) = e^{i theta} psi(u):  2(2 pi n + theta)/L."""
    n = np.arange(-(n_levels // 2), n_levels - n_levels // 2)
    return np.sort(2.0 * (2.0 * np.pi * n + theta) / L)


# --------------------------------------------------------------------------
# Metric symbols f(k) >= 0.  eta = f(p_hat); Delta_f = f^{1/2}(p_hat) up to a
# unitary phase, which is the only thing the metric can see.
# --------------------------------------------------------------------------

METRICS = {
    "bbm":        (lambda k: 4.0 * np.sin(k / 2.0) ** 2,          "4 sin^2(k/2)  (Delta = 1 - e^{-ik})"),
    "bbm_reg":    (lambda k: 4.0 * np.sin(k / 2.0) ** 2 + 1e-2,   "4 sin^2(k/2) + 1e-2  (coercive)"),
    "unit":       (lambda k: np.ones_like(k),                     "1  (plain L^2)"),
    "k2":         (lambda k: k ** 2,                              "k^2  (double zero at k=0)"),
    "k4":         (lambda k: k ** 4,                              "k^4  (quartic zero at k=0)"),
    "sobolev":    (lambda k: 1.0 + k ** 2,                        "1 + k^2  (unbounded, coercive)"),
    "ess_zero":   (lambda k: np.exp(-1.0 / np.maximum(k ** 2, 1e-300)), "exp(-1/k^2)  (zero of infinite order)"),
    "bbm_sob":    (lambda k: 4.0 * np.sin(k / 2.0) ** 2 * (1.0 + k ** 2), "4 sin^2(k/2)(1+k^2)"),
}


def metric_toeplitz(f, N, h):
    """Toeplitz matrix of the Fourier multiplier f(k) on a grid of spacing h."""
    theta = 2.0 * np.pi * np.fft.fftfreq(2 * N)      # grid angle in [-pi, pi)
    k = theta / h
    c = np.fft.ifft(f(k)).real                        # Toeplitz symbol coefficients
    i = np.arange(N)
    return c[np.abs(i[:, None] - i[None, :])]


# --------------------------------------------------------------------------
# reporting helpers
# --------------------------------------------------------------------------

def rule(title, width=78):
    return "\n" + title + "\n" + "-" * width
