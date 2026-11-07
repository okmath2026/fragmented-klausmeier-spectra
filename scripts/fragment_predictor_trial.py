"""Envelope-only first-order Gaussian feature trial preparation.

The numerical eigendecomposition constructs trial data only.  A later
certificate must treat every saved decimal string as an exact number and
check the fragmented Gram matrix, Rayleigh value, residual and trace screen.
No fragmented eigenproblem enters the predictor preparation.

NumPy is imported only by the preparation function.  ``predictor_vector``
uses the standard library and forms q0 + q1/m exactly with Fraction.
"""
from decimal import Decimal
from fractions import Fraction
from math import exp, pi, sqrt

from validated_gaussian_rank import gaussian_gram_enclosure


def predictor_vector(trial, m, corrected=True):
    """Return the saved predictor as exact Fractions, with no rounding.

    In particular, the denominator m=70 is not rounded to a terminating
    decimal.  The returned vector is not normalised; Rayleigh/residual
    formulas must divide by its exact squared Euclidean norm.
    """
    if not isinstance(m, int) or isinstance(m, bool) or m < 2:
        raise ValueError('integer patch count m>=2 required')
    if not isinstance(corrected, bool):
        raise TypeError('corrected must be a boolean')
    q0, q1 = trial['q0'], trial['q1']
    if not q0 or len(q0) != len(q1):
        raise ValueError('nonempty equal-length predictor vectors required')
    if any(not isinstance(x, str) for x in list(q0)+list(q1)):
        raise TypeError('saved predictor coefficients must be decimal strings')
    if any(not Decimal(x).is_finite() for x in list(q0)+list(q1)):
        raise ValueError('finite saved predictor coefficients required')
    result = [Fraction(x)+(Fraction(y)/m if corrected else 0)
              for x, y in zip(q0, q1)]
    if not any(result):
        raise ValueError('the predictor vector must not vanish')
    return result


def prepare_fragment_predictor_trial(D_string, p, precision=60, V_string='2'):
    """Return (trial, diagnostics) from one envelope Gram eigendecomposition.

    Let B0=(V/D)*G_D and
        B1=(V/D)*(D-V)*(f_b f_b^T - G_D/D).
    We choose the unit top eigenvector q0 of B0 with positive first entry
    and its orthogonal first perturbation q1.  Saved strings are arbitrary
    exact proof inputs; these floating diagnostics are never inclusions.
    """
    import os
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    import numpy as np

    D, V = Fraction(D_string), Fraction(V_string)
    if not D > V > 0:
        raise ValueError('exact geometry strings must satisfy D>V>0')
    gram = gaussian_gram_enclosure(D_string, p, precision, None, V_string)
    G = np.array([[(float(lo)+float(hi))/2 for lo, hi in row]
                  for row in gram['blocks'][0]], dtype=float)
    theta, delta, length = float(V/D), float(D-V), float(D)
    B0 = theta*G
    eigenvalues, eigenvectors = np.linalg.eigh(B0)
    lambda0 = float(eigenvalues[-1])
    q0 = eigenvectors[:, -1].copy()
    if q0[0] < 0:
        q0 = -q0
    q0 /= np.linalg.norm(q0)

    b = length/2
    endpoint_all = np.empty(p)
    endpoint_all[0] = (2*pi)**(-.25)*exp(-b*b/2)
    for k in range(1, p):
        endpoint_all[k] = endpoint_all[k-1]*b/sqrt(k)
    endpoint = endpoint_all[::2]
    endpoint_derivative = (np.arange(0, p, 2)/b-b)*endpoint
    B1 = theta*delta*(np.outer(endpoint, endpoint)-G/length)
    first_coefficient = float(q0@B1@q0)
    forcing = B1@q0-first_coefficient*q0
    lower_modes = eigenvectors[:, :-1]
    separation = lambda0-eigenvalues[:-1]
    if not np.all(separation > 0):
        raise ArithmeticError('numerical envelope top mode is not separated')
    q1 = lower_modes@((lower_modes.T@forcing)/separation)
    q1 -= q0*float(q0@q1)

    # Optional diagnostics independently recover the second coefficient.
    kappa = float(V*D*(1-V/D)*(1-2*V/D)/12)
    B2 = theta*B1+2*kappa*(np.outer(endpoint_derivative, endpoint)
                         +np.outer(endpoint, endpoint_derivative))
    second_coefficient = float(q0@B2@q0+q0@B1@q1)
    residual1 = (lambda0*np.eye(len(q0))-B0)@q1-forcing
    if not (np.all(np.isfinite(q0)) and np.all(np.isfinite(q1))):
        raise ArithmeticError('trial preparation produced nonfinite coefficients')

    serial = lambda vector: [format(float(x), '.17g') for x in vector]
    trial = dict(D=D_string, V=V_string, features=p,
                 q0=serial(q0), q1=serial(q1))
    diagnostics = dict(
        certified=False, geometry='envelope only; no fragmented eigensolve',
        features=p, even_features=len(q0), moment_precision=precision,
        lambda0_float=lambda0, first_rho_coefficient_float=first_coefficient,
        second_rho_coefficient_float=second_coefficient,
        q0_norm_float=float(np.linalg.norm(q0)),
        q1_norm_float=float(np.linalg.norm(q1)),
        q0_q1_inner_product_float=float(q0@q1),
        first_vector_equation_residual_float=float(np.linalg.norm(residual1)),
        envelope_top_gap_float=float(separation[-1]),
        trial_preparation='NumPy envelope Gram eigendecomposition; saved decimal strings are exact trial inputs only')
    return trial, diagnostics
