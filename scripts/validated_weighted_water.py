"""Spatial-water exclusion by directed positive-operator supersolutions.

Only the Python standard library is used.  A positive trial vector is an
exact collection of decimal numbers, not an asserted eigenvector.  The
piecewise-constant kernel majorant covers every point of every cell; no
midpoint quadrature or derivative error is omitted from the certificate.
"""
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from hashlib import sha256

from validated_gaussian import _Intervals, _pi_fraction_bounds
from validated_water import rational_water_certificate


def weighted_water_geometry(D_string, m, r, V_string="2"):
    """Exact rational equal-patch and component-water cell geometry."""
    if not isinstance(D_string, str) or not isinstance(V_string, str):
        raise TypeError("D and V must be exact decimal strings")
    D, V = Decimal(D_string), Decimal(V_string)
    if not D.is_finite() or not V.is_finite() or not D >= V > 0:
        raise ValueError("finite exact geometry D>=V>0 is required")
    if any(not isinstance(k, int) or isinstance(k, bool) or k < 1 for k in (m, r)) or m < 2:
        raise ValueError("m>=2 and r>=1 must be integers")
    D, V = Fraction(D), Fraction(V)
    ell = V / m
    gap = (D - V) / (m - 1)
    h = ell / r
    cells = []
    for patch in range(m):
        centre = Fraction(2 * patch - (m - 1), 2) * (ell + gap)
        for local in range(r):
            s = -ell / 2 + Fraction(2 * local + 1, 2) * h
            x = centre + s
            cells.append(dict(patch=patch, local=local, centre=x,
                              left=x - h / 2, right=x + h / 2,
                              nearest_water_centre=max(Fraction(0), abs(s) - h / 2)))
    return dict(D=D, V=V, m=m, r=r, ell=ell, gap=gap, h=h, cells=cells)


def _physical_inputs(dv_string, dw_string, Amax, R, B):
    values = (dv_string, dw_string, Amax, R, B)
    if any(not isinstance(value, str) for value in values):
        raise TypeError("physical parameters must be exact decimal strings")
    decimals = tuple(Decimal(value) for value in values)
    if any(not value.is_finite() for value in decimals):
        raise ValueError("physical parameters must be finite")
    dv, dw, amax, bound, loss = decimals
    if not (dv > 0 and dw > 0 and amax >= 0 and bound > 0 and loss > 0):
        raise ValueError("requires dv,dw,R,B>0 and Amax>=0")
    return tuple(Fraction(value) for value in decimals)


def weighted_water_certificate(D_string, m, r, q, dv_string=".5", dw_string="1",
                               precision=60, V_string="2", Amax="1.8", R="4", B=".45",
                               return_details=False):
    """Certify a component-water spatial criterion for the standard Gaussian.

    Put a(x)=dv+B-Amax*R*z(x), where z is the exact Dirichlet comparison
    profile on each patch. On each cell C_i compute a certified lower bound
    a_i>0, and majorise dv*a(x)^(-1)*K by the positive matrix

        C_ij = dv*|C_j|*sup_(C_i x C_j) J(x-y)/a_i.

    Every positive exact vector q gives the continuum upper bound
    max_i (Cq)_i/q_i. A value below one excludes nonzero stationary biomass
    in the specified class 0<=v_*<=R. A nonpositive a_i returns unavailable;
    a resolved upper bound >=1 is merely inconclusive.

    The default result is a flat CSV-ready summary. return_details=True
    returns (summary, audit), including every upper entry and rational cell.
    """
    geometry = weighted_water_geometry(D_string, m, r, V_string)
    dv, dw, amax, bound, loss = _physical_inputs(dv_string, dw_string, Amax, R, B)
    Q = amax * bound
    q = [Decimal(str(value)) for value in q]
    n = m * r
    if len(q) != n or any(not value.is_finite() or value <= 0 for value in q):
        raise ValueError("q must contain m*r finite strictly positive decimals")
    A = _Intervals(precision)
    lo, up, point = A.down, A.up, A.point
    exact = lambda value: A.fraction(value, value)
    ell, h, gap = geometry["ell"], geometry["h"], geometry["gap"]
    nearest = [cell["nearest_water_centre"] for cell in geometry["cells"]]
    caps = {distance: rational_water_certificate(ell, dw, distance, precision)
            for distance in sorted(set(nearest) | {Fraction(0)})}
    global_cap = caps[Fraction(0)]
    a = []
    for distance in nearest:
        cap = caps[distance]
        value = A.subtract(exact(dv + loss), A.multiply(exact(Q),
                    (Decimal(cap["q_lower"]), Decimal(cap["q_upper"]))))
        a.append(value[0])
    summary = dict(D=D_string, V=V_string, m=m, cells_per_patch=r, N=n,
        precision=precision, dv=dv_string, dw=dw_string, Amax=Amax, R=R, B=B,
        water_domain="component", patch_length_numerator=ell.numerator,
        patch_length_denominator=ell.denominator, gap_numerator=gap.numerator,
        gap_denominator=gap.denominator, cell_width_numerator=h.numerator,
        cell_width_denominator=h.denominator, minimum_a_lower=str(min(a)),
        water_q_lower=global_cap["q_lower"], water_q_upper=global_cap["q_upper"],
        cw_upper=None, majorant_cw_lower=None, majorant_cw_width_upper=None,
        weighted_radius_upper=None, contraction_margin_lower=None,
        kernel_eta_radius_strict=None, weighted_decision="unavailable",
        q_sha256=sha256("\n".join(map(str, q)).encode("ascii")).hexdigest(),
        exact_fraction_row_sum_check=False,
        method="component water; rational-cell Gaussian supremum; directed positive supersolution")
    audit = dict(rational_cells=[{key: str(value) if isinstance(value, Fraction) else value
                 for key, value in cell.items()} for cell in geometry["cells"]],
        cell_a_lower=[str(value) for value in a],
        water_caps=[dict(distance=str(distance), q_lower=cap["q_lower"],
                        q_upper=cap["q_upper"], quadratic_cap_upper=cap["quadratic_cap_upper"])
                    for distance, cap in caps.items()],
        row_ratio_upper=[], matrix_entries_upper=[])
    if min(a) <= 0:
        return (summary, audit) if return_details else summary
    pi = A.fraction(*_pi_fraction_bounds(precision))
    J0 = A.divide(point(1), A.sqrt(A.multiply(point(2), pi)))
    weighted_mass = A.multiply(exact(dv), exact(h))
    # All cells have the same rational width. Their minimum pair distance
    # is max(0,|x_i-x_j|-h), including touching endpoints.
    matrix, ratios, lower_ratios = [], [], []
    for i, cell_i in enumerate(geometry["cells"]):
        row, total = [], Decimal(0)
        for j, cell_j in enumerate(geometry["cells"]):
            distance = max(Fraction(0), abs(cell_i["centre"] - cell_j["centre"]) - h)
            exponent = -distance * distance / 2
            kernel = A.multiply(J0, A.exp(exact(exponent)))
            entry = up.divide(A.multiply(weighted_mass, kernel)[1], a[i])
            row.append(entry)
            total = up.add(total, up.multiply(entry, q[j]))
        ratio = up.divide(total, q[i])
        rational_sum = sum((Fraction(entry) * Fraction(trial)
                            for entry, trial in zip(row, q)), Fraction(0))
        assert rational_sum / Fraction(q[i]) <= Fraction(ratio)
        lower_ratios.append(exact(rational_sum / Fraction(q[i]))[0])
        matrix.append(row)
        ratios.append(ratio)
    c = max(ratios)
    margin = lo.subtract(Decimal(1), c)
    eta = None
    if margin > 0:
        # Young's inequality for an L1 kernel perturbation gives
        # ||S_tilde-S|| <= dv*eta/min(a). The radius is strict.
        eta = lo.divide(lo.multiply(margin, min(a)), exact(dv)[1])
    cw_lower = min(lower_ratios)
    summary.update(cw_upper=str(c), majorant_cw_lower=str(cw_lower),
        majorant_cw_width_upper=str(up.subtract(c, cw_lower)), weighted_radius_upper=str(c),
        contraction_margin_lower=str(margin), kernel_eta_radius_strict=str(eta) if eta is not None else None,
        weighted_decision="excluded" if c < 1 else "inconclusive",
        exact_fraction_row_sum_check=True)
    audit.update(row_ratio_upper=[str(value) for value in ratios],
                 matrix_entries_upper=[[str(value) for value in row] for row in matrix])
    return (summary, audit) if return_details else summary


def weighted_scalar_comparison(record, beta_lower, beta_upper):
    """Directed margin interval for the scalar test at identical parameters."""
    lower, upper = Decimal(str(beta_lower)), Decimal(str(beta_upper))
    if not (lower.is_finite() and upper.is_finite() and lower <= upper):
        raise ValueError("finite ordered spectral endpoints are required")
    A = _Intervals(int(record["precision"]))
    exact = lambda value: A.fraction(value, value)
    dv = Fraction(record["dv"])
    Q = Fraction(record["Amax"]) * Fraction(record["R"])
    loss = Fraction(record["B"])
    interval = A.subtract(A.add(A.multiply(exact(dv), (lower, upper)), exact(loss)),
                          A.multiply(exact(Q), (Decimal(record["water_q_lower"]),
                                                   Decimal(record["water_q_upper"]))))
    scalar_exact_upper = dv * Fraction(upper) + loss - Q * Fraction(record["water_q_lower"])
    assert scalar_exact_upper <= Fraction(interval[1])
    return dict(plant_beta_lower=str(lower), plant_beta_upper=str(upper),
        scalar_margin_lower=str(interval[0]), scalar_margin_upper=str(interval[1]),
        scalar_decision="excluded" if interval[0] > 0 else "inconclusive",
        strict_scalar_failure=interval[1] < 0,
        strict_weighted_improvement=interval[1] < 0 and record["weighted_decision"] == "excluded",
        exact_fraction_scalar_upper_check=True)


def weighted_water_self_check(D_string, m, r, q, dv_string=".5", dw_string="1",
                              V_string="2", Amax="1.8", R="4", B=".45"):
    """Check precision nesting, every majorant entry and ambient independence."""
    kwargs = dict(dv_string=dv_string, dw_string=dw_string, V_string=V_string,
                  Amax=Amax, R=R, B=B, return_details=True)
    coarse, audit60 = weighted_water_certificate(D_string, m, r, q, precision=60, **kwargs)
    fine, audit80 = weighted_water_certificate(D_string, m, r, q, precision=80, **kwargs)
    assert Decimal(fine["minimum_a_lower"]) >= Decimal(coarse["minimum_a_lower"])
    if coarse["cw_upper"] is not None:
        assert fine["cw_upper"] is not None
        assert Decimal(fine["cw_upper"]) <= Decimal(coarse["cw_upper"])
        for row60, row80 in zip(audit60["matrix_entries_upper"], audit80["matrix_entries_upper"]):
            assert all(Decimal(v80) <= Decimal(v60) for v60, v80 in zip(row60, row80))
    with localcontext() as alien:
        alien.prec = 2
        alien.rounding = ROUND_CEILING
        assert weighted_water_certificate(D_string, m, r, q, precision=60, **kwargs) == (coarse, audit60)
    return dict(D=D_string, V=V_string, m=m, cells_per_patch=r,
        precision_nesting_60_80=True, every_majorant_entry_nested=True,
        ambient_context_invariance=True, exact_fraction_row_sum_check=coarse["exact_fraction_row_sum_check"])
