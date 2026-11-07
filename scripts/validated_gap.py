"""Parity-resolved, outward-rounded Gaussian spectral-gap certificates.

The odd restriction has the strictly positive kernel J(x-y)-J(x+y) on
(0,D/2).  Its principal eigenvalue is enclosed by midpoint CW inequalities
and a continuous Galerkin/complement estimate.  A separately certified
even-rest bound proves that this eigenvalue is the full second eigenvalue.
Only the optional trial-vector generator uses ordinary floating point.
"""
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from hashlib import sha256

from validated_gaussian import _Intervals, _pi_fraction_bounds


def _fraction(A, x):
    return A.fraction(x, x)


def odd_trial_vector(D_string, n, tolerance=2e-15):
    """Generate positive decimal test data; no accuracy claim is made here."""
    import math
    if not isinstance(n, int) or isinstance(n, bool) or n < 4 or n % 2:
        raise ValueError("n must be an even integer at least four")
    D = float(D_string)
    if not math.isfinite(D) or D <= 0:
        raise ValueError("D must be finite and positive")
    h = D/n
    k = n//2
    entries = [h/math.sqrt(2*math.pi)*math.exp(-(i*h)**2/2) for i in range(n)]
    matrix = [[entries[abs(i-j)]-entries[i+j+1] for j in range(k)] for i in range(k)]
    q = [(i+.5)*h for i in range(k)]
    for _ in range(1000):
        z = [sum(row[j]*q[j] for j in range(k)) for row in matrix]
        norm = math.sqrt(sum(v*v for v in z))
        z = [v/norm for v in z]
        error = max(abs(z[i]-q[i]) for i in range(k))
        q = z
        if error < tolerance:
            break
    else:
        raise ArithmeticError("odd trial power iteration did not converge")
    if any(v <= 0 for v in q):
        raise ArithmeticError("odd trial vector lost positivity")
    return list(map(str, q))


def gap_certificate(record, q_odd):
    """Refine a validated_boundary certificate using an exact odd test vector.

    ``record`` must be the output of boundary_certificate (or its saved replay).
    The saved principal interval and even-rest bound remain independent inputs.
    The method does not use a numerical second-eigenvalue solve.
    """
    D_string = record["D"]
    D = Decimal(D_string)
    n = int(record["N"])
    precision = int(record["precision"])
    if not D.is_finite() or D <= 0 or n < 4 or n % 2:
        raise ValueError("finite D>0 and even N>=4 are required")
    q = [Decimal(str(x)) for x in q_odd]
    k = n//2
    if len(q) != k or any(not x.is_finite() or x <= 0 for x in q):
        raise ValueError("odd trial data must be N/2 finite positive exact decimals")
    A = _Intervals(precision)
    lo, up, point = A.down, A.up, A.point
    one, two = point(1), point(2)
    DF = Fraction(D)
    hF = DF/n
    h, d = _fraction(A, hF), _fraction(A, DF)
    pi = A.fraction(*_pi_fraction_bounds(precision))
    J0 = A.divide(one, A.sqrt(A.multiply(two, pi)))
    B1 = A.multiply(J0, A.exp(point("-.5")))
    entries = []
    for offset in range(n):
        z = hF*offset
        entries.append(A.multiply(A.multiply(h, J0), A.exp(_fraction(A, -z*z/2))))
    cwL, cwU = None, None
    for i in range(k):
        rowL = rowU = Decimal(0)
        for j, value in enumerate(q):
            entry = A.subtract(entries[abs(i-j)], entries[i+j+1])
            if entry[0] <= 0:
                raise ArithmeticError("strict odd-kernel positivity was not resolved")
            rowL = lo.add(rowL, lo.multiply(entry[0], value))
            rowU = up.add(rowU, up.multiply(entry[1], value))
        rL, rU = lo.divide(rowL, q[i]), up.divide(rowU, q[i])
        cwL = rL if cwL is None else min(cwL, rL)
        cwU = rU if cwU is None else max(cwU, rU)
    h2 = A.multiply(h, h)
    pi2 = A.multiply(pi, pi)
    eps = A.divide(A.multiply(A.multiply(d, J0), h2), point(12))[1]
    derivative1 = min(A.multiply(d, B1)[1], A.multiply(two, J0)[1])
    derivative2 = min(A.multiply(d, J0)[1], A.multiply(point(4), B1)[1])
    a = A.divide(A.multiply(point(derivative2), h2), pi2)[1]
    b = A.divide(A.multiply(point(derivative1), h), pi)[1]
    oddL = lo.subtract(cwL, eps)
    projectedU = up.add(cwU, eps)
    # The maximum eigenvalue of the 2x2 norm majorant is a sharper alternative
    # to u+b^2/(l-a), and it remains valid without a positive l-a condition.
    discriminant = A.add(A.multiply(A.subtract(point(projectedU), point(a)),
                                    A.subtract(point(projectedU), point(a))),
                           A.multiply(point(4), A.multiply(point(b), point(b))))
    oddU = A.divide(A.add(A.add(point(projectedU), point(a)),
                              A.sqrt(discriminant)), two)[1]
    even_rest = Decimal(record["even_rest_upper"])
    if oddL <= even_rest:
        raise ArithmeticError("odd Perron/even-rest ordering is inconclusive")
    rhoL, rhoU = Decimal(record["rho_lower"]), Decimal(record["rho_upper"])
    theta = _fraction(A, Fraction(record["V"])/DF)
    gammaL = A.multiply(theta, point(lo.subtract(rhoL, oddU)))[0]
    gammaU = A.multiply(theta, point(up.subtract(rhoU, oddL)))[1]
    if gammaL <= 0 or gammaL > gammaU:
        raise ArithmeticError("refined gap enclosure is not positive")
    baselineL, baselineU = Decimal(record["gamma_lower"]), Decimal(record["gamma_upper"])
    if gammaL < baselineL or gammaU > baselineU:
        # Intersection preserves either valid certificate even for a poor trial.
        gammaL, gammaU = max(gammaL, baselineL), min(gammaU, baselineU)
    return dict(D=D_string, V=record["V"], N=n, precision=precision,
                odd_cw_lower=str(cwL), odd_cw_upper=str(cwU),
                odd_projection_error_upper=str(eps), odd_complement_a_upper=str(a),
                odd_coupling_b_upper=str(b), odd_eigenvalue_lower=str(oddL),
                odd_eigenvalue_upper=str(oddU),
                odd_even_ordering_margin_lower=str(lo.subtract(oddL, even_rest)),
                baseline_gamma_lower=record["gamma_lower"],
                baseline_gamma_upper=record["gamma_upper"],
                gamma_lower=str(gammaL), gamma_upper=str(gammaU),
                gamma_width_upper=str(up.subtract(gammaU, gammaL)),
                odd_q_sha256=sha256("\n".join(map(str, q)).encode("ascii")).hexdigest(),
                gap_method="positive odd half-kernel CW; quadratic projection; even-rest ordering")


def gap_self_check(record, q_odd):
    """Check precision nesting and ambient-context independence.

    The finer run retains the same independently valid saved principal and
    even-rest endpoints, so it tests the new odd-kernel arithmetic in isolation.
    """
    result = gap_certificate(record, q_odd)
    finer_record = dict(record, precision=int(record["precision"])+20)
    finer = gap_certificate(finer_record, q_odd)
    for lower, upper in (("odd_eigenvalue_lower", "odd_eigenvalue_upper"),
                         ("gamma_lower", "gamma_upper")):
        assert Decimal(result[lower]) <= Decimal(finer[lower])
        assert Decimal(finer[upper]) <= Decimal(result[upper])
    with localcontext() as alien:
        alien.prec = 2
        alien.rounding = ROUND_CEILING
        assert gap_certificate(record, q_odd) == result
    assert Decimal(result["odd_even_ordering_margin_lower"]) > 0
    return dict(D=record["D"], N=int(record["N"]),
                gap_precision_nesting=True,
                gap_ambient_context_invariance=True, odd_even_ordering=True)
