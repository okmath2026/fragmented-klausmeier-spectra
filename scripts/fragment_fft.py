"""Matrix-free Gaussian fragment calculations using block-Toeplitz FFTs.

These are float64 quadrature diagnostics, not outward-rounded certificates.
The local Gauss rule and every matrix-vector product are deterministic.
"""
import math
import time

import numpy as np


def _operator(m, D, nodes_per_patch, V):
    if not isinstance(m, int) or isinstance(m, bool) or m < 2:
        raise ValueError("m must be an integer at least two")
    if (not isinstance(nodes_per_patch, int) or isinstance(nodes_per_patch, bool)
            or nodes_per_patch < 1):
        raise ValueError("nodes_per_patch must be a positive integer")
    D, V = float(D), float(V)
    if not (math.isfinite(D) and math.isfinite(V) and D >= V > 0):
        raise ValueError("finite lengths must satisfy D >= V > 0")
    ell = V / m
    gap = (D - V) / (m - 1)
    step = ell + gap
    local_x, local_w = np.polynomial.legendre.leggauss(nodes_per_patch)
    local_x = local_x * ell / 2
    local_w = local_w * ell / 2
    r = nodes_per_patch
    # B[k,a,b] is the interaction from patch j,node b to patch j+k,node a.
    lag = np.arange(m)[:, None, None] * step
    distance = lag + local_x[None, :, None] - local_x[None, None, :]
    blocks = (np.exp(-distance * distance / 2) / math.sqrt(2 * math.pi)
              * np.sqrt(local_w[:, None] * local_w[None, :]))
    # Embed the Toeplitz block column in a real block-circulant operator.
    # Negative lags are transposed, rather than repeated positive-lag blocks.
    size = 1 << (2 * m - 2).bit_length()
    embedded = np.zeros((size, r, r))
    embedded[:m] = blocks
    positive = np.arange(1, m)
    embedded[size - positive] = blocks[1:].transpose(0, 2, 1)
    symbol = np.fft.rfft(embedded, axis=0)

    def matvec(vector):
        values = np.asarray(vector, dtype=float).reshape(m, r)
        padded = np.zeros((size, r))
        padded[:m] = values
        transformed = np.fft.rfft(padded, axis=0)
        product = np.einsum("kab,kb->ka", symbol, transformed, optimize=True)
        result = np.fft.irfft(product, n=size, axis=0)[:m]
        return result.reshape(-1)

    centres = (np.arange(m) - (m - 1) / 2) * step
    points = (centres[:, None] + local_x).reshape(-1)
    weights = np.tile(local_w, m)
    return matvec, points, weights


def fragment_fft(m, D, nodes_per_patch=4, V=2.):
    """Return a floating-point principal-loss estimate for fixed endpoints.

    Equal patches have total length V, envelope span D, and edge gap
    (D-V)/(m-1).  Per-patch Gauss nodes give a symmetric weighted matrix.
    FFT products cost O(r^2*m*log(m)) and store O(r^2*m) numbers for r
    nodes per patch; no (m*r)-by-(m*r) dense matrix is constructed.

    collatz_width and residual diagnose the floating-point matrix solve;
    neither these nor the Gauss refinement comparison are interval proofs.
    """
    matvec, points, weights = _operator(m, D, nodes_per_patch, V)
    n = len(points)
    q = np.ones(n) / math.sqrt(n)
    for iteration in range(1, 30001):
        y = matvec(q)
        if np.any(y <= 0) or not np.all(np.isfinite(y)):
            raise ArithmeticError("FFT product lost finite strict positivity")
        width = float(np.ptp(y / q))
        if width < 2e-12:
            rho = float(q @ y)
            residual = float(np.linalg.norm(y - rho * q))
            return dict(beta=1 - rho, collatz_width=width, residual=residual,
                        iterations=iteration, N=n)
        q = y / np.linalg.norm(y)
    raise RuntimeError("FFT Perron iteration did not converge")


def self_check():
    """Check the production product against dense algebra and fixed values."""
    matvec, x, w = _operator(4, 4., 4, 2.)
    distance = x[:, None] - x[None, :]
    dense = (np.sqrt(w[:, None] * w[None, :])
             * np.exp(-distance * distance / 2) / math.sqrt(2 * math.pi))
    error = 0.
    for vector in (np.arange(1, len(x) + 1) / len(x),
                   np.sin(np.arange(len(x)) + .3)):
        error = max(error, float(np.max(np.abs(matvec(vector) - dense @ vector))))
    assert error < 1e-12
    calculated = fragment_fft(4, 4., 4)["beta"]
    dense_beta = 1 - float(np.linalg.eigvalsh(dense)[-1])
    eigen_error = abs(calculated - dense_beta)
    assert eigen_error < 1e-12
    # Rounded reference values from independent dense 24-point/patch runs.
    references = {4.: .586469729548, 6.: .699830151916}
    reference_error = 0.
    for D, reference in references.items():
        for r in (4, 8):
            reference_error = max(reference_error,
                                  abs(fragment_fft(64, D, r)["beta"] - reference))
    assert reference_error < 1e-11
    return dict(dense_matvec_error=error, dense_eigenvalue_error=eigen_error,
                independent_m64_reference_error=reference_error)


if __name__ == "__main__":
    import json
    print(json.dumps(dict(checks=self_check()), indent=2))
    for D in (4., 6.):
        for m in (256, 512, 1024):
            for r in (4, 8):
                start = time.perf_counter()
                result = fragment_fft(m, D, r)
                print(json.dumps(dict(D=D, m=m, nodes_per_patch=r,
                                      seconds=time.perf_counter() - start,
                                      **result)))
