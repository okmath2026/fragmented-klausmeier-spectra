"""Float64 preparation of positive trials for spatial-water certificates.

The output strings, rather than eigenvector accuracy, are proof inputs.
The separate standard-library certifier encloses all continuum majorants.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import math
import numpy as np

from validated_weighted_water import weighted_water_geometry, _physical_inputs


def weighted_water_trial(D_string, m, r, dv_string=".5", dw_string="1",
                         V_string="2", Amax="1.8", R="4", B=".45"):
    """Prepare a symmetric positive trial for the cell majorant, not K itself."""
    geometry = weighted_water_geometry(D_string, m, r, V_string)
    dv, dw, amax, bound, loss = map(float, _physical_inputs(dv_string, dw_string, Amax, R, B))
    h, ell = float(geometry["h"]), float(geometry["ell"])
    x = np.array([float(cell["centre"]) for cell in geometry["cells"]])
    nearest = np.array([float(cell["nearest_water_centre"]) for cell in geometry["cells"]])
    # The exp ratio avoids overflow of cosh for small water diffusivity.
    t = ell / (2 * math.sqrt(dw))
    y = nearest / math.sqrt(dw)
    ratio = (np.exp(-(t-y)) + np.exp(-(t+y))) / (1 + math.exp(-2*t))
    z = 1 - ratio
    a = dv + loss - amax * bound * z
    if np.any(a <= 0) or not np.all(np.isfinite(a)):
        raise ValueError("the requested trial requires positive cellwise a")
    distance = np.maximum(np.abs(x[:, None] - x[None, :]) - h, 0)
    symmetric = dv * h * np.exp(-distance**2/2) / math.sqrt(2*math.pi) / np.sqrt(a[:, None]*a[None, :])
    eigenvalues, vectors = np.linalg.eigh(symmetric)
    q = vectors[:, -1] / np.sqrt(a)
    if np.sum(q) < 0:
        q = -q
    q = (q + q[::-1]) / 2
    q = q / np.max(q)
    if not np.all(np.isfinite(q)) or np.any(q <= 0):
        raise ArithmeticError("positive trial preparation failed")
    strings = [str(float(value)) for value in q]
    assert strings == strings[::-1]
    matrix = symmetric * np.sqrt(a[None, :] / a[:, None])
    ratios = (matrix @ q) / q
    return strings, dict(upper_matrix_rho_float=float(eigenvalues[-1]),
        float_cw_width=float(np.ptp(ratios)),
        trial_preparation="positive float64 cell-majorant vector; accuracy not assumed")
