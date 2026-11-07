"""Geometry- and symmetry-aware finite-fragment remainder certificates.

This module uses only standard-library directed interval primitives.  Inputs
are already verified continuous envelope/coefficient enclosures; no floating
eigenvalue, eigenvector, or diagnostic FFT value enters the bounds.
"""
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction

from validated_gaussian import _Intervals, _pi_fraction_bounds


def _fraction(A, value):
    exact = Fraction(value)
    return A.fraction(exact, exact)


def sharp_remainder(record, m):
    """Certify |beta_m-beta_infty-c_D/m| using the even spectral sector.

    The geometry is the fixed-endpoint equal-patch family of the manuscript.
    The returned denominator is certified positive; unlike the older bound,
    this a posteriori condition does not impose e <= gamma/4.
    """
    if not isinstance(m, int) or isinstance(m, bool) or m < 2:
        raise ValueError("m must be an integer at least two")
    A = _Intervals(int(record["precision"]))
    lo, up, point = A.down, A.up, A.point
    D, V = Fraction(record["D"]), Fraction(record.get("V", "2"))
    if D <= V or V <= 0:
        raise ValueError("requires exact geometry D > V > 0")
    delta = D - V
    dI, vI, deltaI, theta = (_fraction(A, x) for x in (D, V, delta, V/D))
    rL, rU = Decimal(record["rho_lower"]), Decimal(record["rho_upper"])
    cL, cU = Decimal(record["c_lower"]), Decimal(record["c_upper"])
    if rL <= 0 or cL <= 0 or cU < cL:
        raise ValueError("requires positive certified rho and c_D intervals")
    B1, B2 = point(record["B1_upper"]), point(record["B2_upper"])
    pi = A.fraction(*_pi_fraction_bounds(int(record["precision"])))
    sqrt_pi = A.sqrt(pi)
    # ||J||_2^2=1/(2 sqrt(pi)), ||J'||_1=2J(0),
    # ||J''||_1=4J(1); whole-line Young bounds also apply to zero extension.
    Jnorm = A.sqrt(A.divide(point(1), A.multiply(point(2), sqrt_pi)))
    D1 = min(A.multiply(dI, B1)[1], A.multiply(point(2), B2)[1])
    D2 = min(A.multiply(dI, B2)[1], A.multiply(point(4), B1)[1])
    F = up.add(up.divide(up.multiply(D1, D1), rL), D2)
    even_rest = Decimal(record["even_rest_upper"])
    even_gap = A.multiply(theta, point(lo.subtract(rL, even_rest)))[0]
    if even_gap <= 0:
        raise ArithmeticError("no positive even-sector gap established")

    def at(k):
        mm = k*(k-1)
        # Exact mean-square displacement: int d_m^2=D Delta^2/[6m(m-1)].
        moment = _fraction(A, D*delta*delta/Fraction(6*mm))
        e2 = A.multiply(A.multiply(theta, theta),
                        A.multiply(A.multiply(B1, B1),
                                   A.multiply(A.multiply(point(2), dI), moment)))[1]
        e = A.sqrt(point(e2))[1]
        # d is odd and u is even, so int d*u=0.  Therefore the quadratic
        # Taylor remainder is bounded by theta B2 ||u||_1 int d^2*u.
        # ||u||_1<=sqrt(D) and ||u||_infty<=||J||_2/rL.
        taylor = A.divide(A.multiply(theta,
                        A.multiply(B2, A.multiply(A.sqrt(dI),
                        A.multiply(Jnorm, moment)))), point(rL))[1]
        cell = A.divide(A.multiply(theta,
                        A.multiply(deltaI, A.multiply(dI, point(F)))),
                        point(4*mm))[1]
        scalar_error = up.add(taylor, cell)
        aL = lo.subtract(lo.divide(cU.copy_negate(), Decimal(k-1)), scalar_error)
        aU = up.add(up.divide(cL.copy_negate(), Decimal(k-1)), scalar_error)
        # Rayleigh gives lambda_m>=lambda_0+a.  Using min(e,-aL) is valid
        # because a>=-||E|| and a>=aL independently.
        loss = min(e, max(Decimal(0), aL.copy_negate()))
        denominator = lo.subtract(lo.subtract(even_gap, e), loss)
        if denominator <= 0:
            raise ArithmeticError("the verified even-sector Schur denominator is not positive")
        amin = (aU.copy_negate() if aU < 0 else aL if aL > 0 else Decimal(0))
        coupling2 = up.subtract(e2, lo.multiply(amin, amin))
        if coupling2 < 0:
            raise ArithmeticError("inconsistent perturbation/scalar enclosures")
        spectral = up.divide(coupling2, denominator)
        coefficient = up.divide(cU, Decimal(mm))
        total = up.add(up.add(spectral, scalar_error), coefficient)
        return dict(displacement_second_moment_upper=moment[1],
                    perturbation_hs_squared_upper=e2,
                    perturbation_hs_upper=e,
                    taylor_remainder_upper=taylor,
                    cell_cancellation_upper=cell,
                    rayleigh_shift_lower=aL, rayleigh_shift_upper=aU,
                    rayleigh_loss_upper=loss,
                    schur_denominator_lower=denominator,
                    centred_coupling_squared_upper=coupling2,
                    spectral_remainder_upper=spectral,
                    coefficient_conversion_upper=coefficient,
                    remainder_upper=total)

    first, twice = at(m), at(2*m)
    # The saved scalar is exact, and its uncertainty is propagated explicitly.
    cmid = A.near.divide(A.near.add(cL, cU), Decimal(2))
    cerr = max(up.subtract(cmid, cL), up.subtract(cU, cmid))
    beta_inf = A.subtract(point(1), A.multiply(theta, (rL, rU)))
    R = first["remainder_upper"]
    beta_m = A.add(A.add(beta_inf, A.divide((cL, cU), point(m))),
                   (R.copy_negate(), R))
    output = dict(D=record["D"], V=str(V), N=int(record["N"]), m=m,
                  precision=int(record["precision"]),
                  even_gap_lower=str(even_gap), F_young_upper=str(F),
                  gaussian_L2_upper=str(Jnorm[1]),
                  c_midpoint=str(cmid), c_midpoint_error_upper=str(cerr),
                  midpoint_coefficient_corrected_error_upper=str(
                      up.add(R, up.divide(cerr, Decimal(m)))),
                  richardson_error_upper=str(up.add(R, up.multiply(Decimal(2), twice["remainder_upper"]))),
                  beta_infty_lower=str(beta_inf[0]), beta_infty_upper=str(beta_inf[1]),
                  beta_m_lower=str(beta_m[0]), beta_m_upper=str(beta_m[1]),
                  method="even-sector Schur; displacement moment; Young derivatives; centred coupling")
    output.update({key: str(value) for key, value in first.items()})
    return output


def sharp_resolution(record, maximum=1000000):
    """Smallest m whose computed Schur denominator is positive.

    The validity test improves monotonically with m: e decreases, the scalar
    Taylor/cell error decreases, and c/(m-1) decreases.  Thus the saved integer
    also gives a sufficient resolution condition for all larger patch counts.
    """
    for m in range(2, maximum+1):
        try:
            sharp_remainder(record, m)
            return m
        except ArithmeticError as error:
            if "denominator" not in str(error):
                raise
    raise ArithmeticError("no certified resolution within the requested range")


def sharp_self_check(record, m):
    baseline = sharp_remainder(record, m)
    finer_record = dict(record, precision=int(record["precision"])+20)
    finer = sharp_remainder(finer_record, m)
    for key in ("remainder_upper", "richardson_error_upper",
                "midpoint_coefficient_corrected_error_upper"):
        assert Decimal(finer[key]) <= Decimal(baseline[key])
    assert Decimal(baseline["beta_m_lower"]) <= Decimal(finer["beta_m_lower"])
    assert Decimal(finer["beta_m_upper"]) <= Decimal(baseline["beta_m_upper"])
    with localcontext() as unrelated:
        unrelated.prec = 2
        unrelated.rounding = ROUND_CEILING
        assert sharp_remainder(record, m) == baseline
    assert Decimal(baseline["schur_denominator_lower"]) > 0
    assert Decimal(baseline["remainder_upper"]) > 0
    return {"D": record["D"], "m": m, "ambient_context_invariance": True,
            "precision_nesting_60_80": True}


def sharp_eventual_sign(record, search_maximum=256):
    """Uniform proof of beta_m>beta_infty for every m>=the returned count.

    For each tested anchor M, all norm/scalar-loss bounds decrease with m.
    The Schur denominator is therefore at least its value at M.  Discarding
    the nonnegative scalar residual subtraction then gives K_M/[m(m-1)].
    Finite integer search only selects the smallest proved threshold among
    these anchors; it is not an asserted globally optimal threshold.
    """
    if not isinstance(search_maximum, int) or search_maximum < 2:
        raise ValueError("search_maximum must be an integer at least two")
    A = _Intervals(int(record["precision"]))
    up = A.up
    cL = Decimal(record["c_lower"])
    best = None
    for M in range(2, search_maximum+1):
        try:
            row = sharp_remainder(record, M)
        except ArithmeticError as error:
            if "denominator" not in str(error):
                raise
            continue
        denominator = Decimal(row["schur_denominator_lower"])
        # Reconstruct an upward bound on the scale-independent coefficients
        # from this anchor's directed enclosures.
        scaled_spectral = up.divide(Decimal(row["perturbation_hs_squared_upper"]), denominator)
        sum_terms = scaled_spectral
        for key in ("taylor_remainder_upper", "cell_cancellation_upper", "coefficient_conversion_upper"):
            sum_terms = up.add(sum_terms, Decimal(row[key]))
        K = up.multiply(sum_terms, Decimal(M*(M-1)))
        ratio = up.divide(K, cL)
        threshold = max(M, int(ratio.to_integral_value(rounding=ROUND_FLOOR))+2)
        candidate = dict(D=record["D"], N=int(record["N"]),
                         anchor_m=M, uniform_remainder_constant_upper=str(K),
                         uniform_schur_denominator_lower=str(denominator),
                         eventual_sign_count=threshold,
                         anchor_search_maximum=search_maximum)
        if best is None or threshold < best["eventual_sign_count"]:
            best = candidate
    if best is None:
        raise ArithmeticError("no positive Schur denominator at the searched anchors")
    return best
