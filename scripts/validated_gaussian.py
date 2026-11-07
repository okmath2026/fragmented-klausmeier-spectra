"""Outward-rounded Gaussian midpoint Collatz--Wielandt certificates.

Only Python's standard library is used here.  The positive input vector need
not be an exact eigenvector: its decimal strings are treated as exact numbers.
No floating-point kernel values enter the enclosure.  Basic operations use
explicit directed Decimal contexts; exp is correctly rounded and expanded by
one adjacent representable value at each endpoint.  Square-root endpoints are
additionally verified by exact rational squaring.
"""
from decimal import (
    Context, Decimal, FloatOperation, ROUND_CEILING, ROUND_FLOOR,
    ROUND_HALF_EVEN, Subnormal, Underflow,
)
from fractions import Fraction
from hashlib import sha256

if not __debug__:
    raise RuntimeError("Run validated certificates without -O: assertion checks are required.")


class _Intervals:
    def __init__(self, precision):
        if not isinstance(precision, int) or precision < 30:
            raise ValueError("precision must be an integer at least 30")
        self.down = Context(prec=precision, rounding=ROUND_FLOOR)
        self.up = Context(prec=precision, rounding=ROUND_CEILING)
        self.near = Context(prec=precision, rounding=ROUND_HALF_EVEN)
        for context in (self.down, self.up, self.near):
            context.traps[FloatOperation] = True
            # Reject unsupported extreme geometry instead of silently losing
            # positive kernel entries through underflow/subnormal arithmetic.
            context.traps[Underflow] = True
            context.traps[Subnormal] = True

    @staticmethod
    def point(value):
        value = Decimal(value)
        return value, value

    def fraction(self, lower, upper):
        return (self.down.divide(Decimal(lower.numerator), Decimal(lower.denominator)),
                self.up.divide(Decimal(upper.numerator), Decimal(upper.denominator)))

    def add(self, a, b):
        return self.down.add(a[0], b[0]), self.up.add(a[1], b[1])

    def subtract(self, a, b):
        return self.down.subtract(a[0], b[1]), self.up.subtract(a[1], b[0])

    @staticmethod
    def negate(a):
        # Unary minus can round through the ambient Decimal context.
        return a[1].copy_negate(), a[0].copy_negate()

    def multiply(self, a, b):
        return (min(self.down.multiply(x, y) for x in a for y in b),
                max(self.up.multiply(x, y) for x in a for y in b))

    def divide(self, a, b):
        if b[0] <= 0 <= b[1]:
            raise ZeroDivisionError("interval denominator contains zero")
        return (min(self.down.divide(x, y) for x in a for y in b),
                max(self.up.divide(x, y) for x in a for y in b))

    def exp(self, a):
        lower = self.near.next_minus(self.near.exp(a[0]))
        upper = self.near.next_plus(self.near.exp(a[1]))
        if not (lower.is_finite() and upper.is_finite() and lower > 0):
            raise ArithmeticError("exp interval exceeds the supported finite range")
        return lower, upper

    def sqrt(self, a):
        if a[0] <= 0:
            raise ValueError("this implementation uses strictly positive square roots")
        lower = self.near.next_minus(self.near.sqrt(a[0]))
        upper = self.near.next_plus(self.near.sqrt(a[1]))
        # These checks prove the endpoint directions without assuming that a
        # requested FLOOR/CEILING mode is respected by Decimal.sqrt.
        while Fraction(lower) ** 2 > Fraction(a[0]):
            lower = self.near.next_minus(lower)
        while Fraction(upper) ** 2 < Fraction(a[1]):
            upper = self.near.next_plus(upper)
        assert 0 < lower <= upper
        return lower, upper


def _atan_reciprocal_bounds(denominator, terms):
    """Exact alternating-series enclosure for atan(1/denominator)."""
    partial = sum((Fraction((-1) ** k,
                            (2 * k + 1) * denominator ** (2 * k + 1))
                   for k in range(terms)), Fraction(0))
    adjacent = partial + Fraction((-1) ** terms,
                                   (2 * terms + 1) * denominator ** (2 * terms + 1))
    return min(partial, adjacent), max(partial, adjacent)


def _pi_fraction_bounds(precision):
    """Machin identity with exact rational alternating-series remainders."""
    a = _atan_reciprocal_bounds(5, precision + 20)
    b = _atan_reciprocal_bounds(239, precision // 3 + 10)
    lower, upper = 16 * a[0] - 4 * b[1], 16 * a[1] - 4 * b[0]
    assert Fraction(3) < lower < upper < Fraction(22, 7)
    return lower, upper


def _collatz_bounds(entries, q, arithmetic):
    """Directed CW evaluation for positive symmetric Toeplitz entries."""
    lower_ratios, upper_ratios = [], []
    for i in range(len(q)):
        lower_sum = upper_sum = Decimal(0)
        for j, value in enumerate(q):
            entry = entries[abs(i-j)]
            lower_sum = arithmetic.down.add(
                lower_sum, arithmetic.down.multiply(entry[0], value))
            upper_sum = arithmetic.up.add(
                upper_sum, arithmetic.up.multiply(entry[1], value))
        lower_ratios.append(arithmetic.down.divide(lower_sum, q[i]))
        upper_ratios.append(arithmetic.up.divide(upper_sum, q[i]))
    lower, upper = min(lower_ratios), max(upper_ratios)
    assert Decimal(0) < lower <= upper
    return lower, upper


def gaussian_certificate(L_string, n, q, precision=60):
    """Enclose beta_1 for the standard Gaussian on (-L,L).

    L_string is an exact positive decimal string; n is the midpoint cell count.
    q is any finite, strictly positive vector of length n.  Each q component is
    converted through str(float(component)); these strings, rather than an
    alleged exact eigenvector, define the exact CW test vector.

    The nonlinear interpretation uses Amax=1.8, B=.45, R=4 and dv=45,
    hence M=6.75 and M/dv=.15.  The optional robust test fixes eta=.005.
    Output endpoints are full-precision decimal strings, without display
    rounding.  A failed sufficient test is reported as 'inconclusive'.
    """
    if not isinstance(L_string, str):
        raise TypeError("L_string must specify exact decimal geometry")
    L = Decimal(L_string)
    if not L.is_finite() or L <= 0:
        raise ValueError("L must be finite and positive")
    if not isinstance(n, int) or isinstance(n, bool) or n < 1:
        raise ValueError("n must be a positive integer")
    q = [Decimal(str(float(value))) for value in q]
    if len(q) != n or any(not value.is_finite() or value <= 0 for value in q):
        raise ValueError("q must contain n finite, strictly positive components")
    arithmetic = _Intervals(precision)
    one, two = arithmetic.point(1), arithmetic.point(2)
    length = arithmetic.multiply(two, arithmetic.point(L))
    h = arithmetic.divide(length, arithmetic.point(n))
    h_exact = 2 * Fraction(L) / n
    pi = arithmetic.fraction(*_pi_fraction_bounds(precision))
    normalizer = arithmetic.divide(one, arithmetic.sqrt(arithmetic.multiply(two, pi)))
    lipschitz = arithmetic.multiply(
        normalizer, arithmetic.exp(arithmetic.point("-.5")))
    # S_ij = h J((i-j)h), because midpoint weights all equal h.
    entries = []
    for offset in range(n):
        # Use exact rational geometry before enclosing the negative exponent.
        exponent_exact = -(h_exact * offset) ** 2 / 2
        exponent = arithmetic.fraction(exponent_exact, exponent_exact)
        entries.append(arithmetic.multiply(
            arithmetic.multiply(h, normalizer), arithmetic.exp(exponent)))
    rho_lower, rho_upper = _collatz_bounds(entries, q, arithmetic)
    cell_bound = arithmetic.multiply(arithmetic.multiply(length, lipschitz), h)[1]
    beta_lower = arithmetic.down.subtract(
        arithmetic.down.subtract(Decimal(1), rho_upper), cell_bound)
    beta_upper = arithmetic.up.add(
        arithmetic.up.subtract(Decimal(1), rho_lower), cell_bound)
    eta_budget = arithmetic.down.subtract(beta_lower, Decimal(".15"))
    nominal_margin = arithmetic.down.subtract(
        arithmetic.down.multiply(Decimal(45), beta_lower), Decimal("6.75"))
    robust_margin = arithmetic.down.subtract(arithmetic.down.multiply(
        Decimal(45), arithmetic.down.subtract(beta_lower, Decimal(".005"))),
        Decimal("6.75"))
    # A cell-average Galerkin comparison yields a second-order enclosure.
    # B2 = sup |J''| = J(0) for the standard Gaussian.  All constants below
    # are upper bounds; the denominator is a lower bound on the block gap.
    h_squared = arithmetic.multiply(h, h)
    projection_error = arithmetic.divide(
        arithmetic.multiply(arithmetic.multiply(length, normalizer), h_squared),
        arithmetic.point(12))[1]
    complement_a = arithmetic.divide(
        arithmetic.multiply(arithmetic.multiply(length, normalizer), h_squared),
        arithmetic.multiply(pi, pi))[1]
    coupling_b = arithmetic.divide(
        arithmetic.multiply(arithmetic.multiply(length, lipschitz), h), pi)[1]
    projected_lower = arithmetic.down.subtract(rho_lower, projection_error)
    projected_upper = arithmetic.up.add(rho_upper, projection_error)
    complement_gap = arithmetic.down.subtract(projected_lower, complement_a)
    quadratic = dict(
        projection_error_upper=str(projection_error),
        complement_a_upper=str(complement_a),
        coupling_b_upper=str(coupling_b),
        complement_gap_lower=str(complement_gap),
    )
    endpoint_fields = (
        "quadratic_beta_lower", "quadratic_beta_upper", "quadratic_width_upper",
        "quadratic_eta_budget_lower", "quadratic_nominal_margin_lower",
        "quadratic_robust_margin_lower",
    )
    if complement_gap > 0:
        correction = arithmetic.up.divide(
            arithmetic.up.multiply(coupling_b, coupling_b), complement_gap)
        quadratic_lower = arithmetic.down.subtract(
            arithmetic.down.subtract(Decimal(1), projected_upper), correction)
        quadratic_upper = arithmetic.up.subtract(Decimal(1), projected_lower)
        quadratic_width = arithmetic.up.subtract(quadratic_upper, quadratic_lower)
        quadratic_eta = arithmetic.down.subtract(quadratic_lower, Decimal(".15"))
        quadratic_nominal = arithmetic.down.subtract(
            arithmetic.down.multiply(Decimal(45), quadratic_lower), Decimal("6.75"))
        quadratic_robust = arithmetic.down.subtract(arithmetic.down.multiply(
            Decimal(45), arithmetic.down.subtract(quadratic_lower, Decimal(".005"))),
            Decimal("6.75"))
        quadratic.update(zip(endpoint_fields, map(str, (
            quadratic_lower, quadratic_upper, quadratic_width, quadratic_eta,
            quadratic_nominal, quadratic_robust))))
        quadratic.update(
            quadratic_decision="excluded" if quadratic_nominal > 0 else "inconclusive",
            quadratic_robust_decision="excluded" if quadratic_robust > 0 else "inconclusive",
        )
    else:
        quadratic.update(dict.fromkeys(endpoint_fields))
        quadratic.update(quadratic_decision="unavailable",
                         quadratic_robust_decision="unavailable")
    return dict(
        L=L_string, N=n, precision=precision,
        rho_lower=str(rho_lower), rho_upper=str(rho_upper),
        beta_lower=str(beta_lower), beta_upper=str(beta_upper),
        cell_bound=str(cell_bound), eta_budget_lower=str(eta_budget),
        nominal_margin_lower=str(nominal_margin), robust_margin_lower=str(robust_margin),
        decision="excluded" if nominal_margin > 0 else "inconclusive",
        robust_decision="excluded" if robust_margin > 0 else "inconclusive",
        pi_lower=str(pi[0]), pi_upper=str(pi[1]),
        q_sha256=sha256("\n".join(map(str, q)).encode("ascii")).hexdigest(),
        arithmetic="Decimal directed FLOOR/CEILING; expanded exp; verified sqrt",
        **quadratic,
    )


def self_check():
    """Small exact and higher-precision checks; no numerical eigensolver used."""
    arithmetic = _Intervals(60)
    a = arithmetic.fraction(Fraction(1, 7), Fraction(1, 7))
    assert Fraction(a[0]) <= Fraction(1, 7) <= Fraction(a[1])
    a = arithmetic.fraction(Fraction(1, 3), Fraction(2, 3))
    b = arithmetic.fraction(Fraction(2, 7), Fraction(3, 7))
    for result, exact_lower, exact_upper in (
        (arithmetic.add(a, b), Fraction(13, 21), Fraction(23, 21)),
        (arithmetic.subtract(a, b), Fraction(-2, 21), Fraction(8, 21)),
        (arithmetic.multiply(a, b), Fraction(2, 21), Fraction(2, 7)),
        (arithmetic.divide(a, b), Fraction(7, 9), Fraction(7, 3)),
        (arithmetic.multiply(arithmetic.negate(a), b),
         Fraction(-2, 7), Fraction(-2, 21)),
    ):
        assert Fraction(result[0]) <= exact_lower <= exact_upper <= Fraction(result[1])
    # Reuse exactly the production CW loop on a matrix with eigenvalue 3.
    rho = _collatz_bounds([arithmetic.point(2), arithmetic.point(1)],
                         [Decimal(1), Decimal(1)], arithmetic)
    assert Fraction(rho[0]) <= 3 <= Fraction(rho[1])
    s = arithmetic.sqrt(arithmetic.point(2))
    assert Fraction(s[0]) ** 2 <= 2 <= Fraction(s[1]) ** 2
    e = arithmetic.exp(arithmetic.point(0))
    assert e[0] < 1 < e[1]
    pi = arithmetic.fraction(*_pi_fraction_bounds(60))
    pi80 = _Intervals(80).fraction(*_pi_fraction_bounds(80))
    assert pi[0] <= pi80[0] < pi80[1] <= pi[1]
    # A fixed exact decimal vector allows a strict precision comparison.
    a = gaussian_certificate("2", 20, [1.0] * 20, 60)
    b = gaussian_certificate("2", 20, [1.0] * 20, 80)
    for lower, upper in (("rho_lower", "rho_upper"), ("beta_lower", "beta_upper"),
                         ("quadratic_beta_lower", "quadratic_beta_upper")):
        assert Decimal(a[lower]) <= Decimal(b[lower])
        assert Decimal(b[upper]) <= Decimal(a[upper])
    assert a["quadratic_decision"] == b["quadratic_decision"]
    assert a["quadratic_robust_decision"] == b["quadratic_robust_decision"]
    return {"exact_fraction_check": True, "interval_basic_operations_check": True,
            "rational_toeplitz_CW_check": True, "verified_sqrt_check": True,
            "precision_nesting_60_80": True,
            "quadratic_precision_nesting_60_80": True}


def arithmetic_audit(precision=60):
    """Machine-readable description of constants, rounding, and local checks."""
    arithmetic = _Intervals(precision)
    pi = arithmetic.fraction(*_pi_fraction_bounds(precision))
    return dict(
        precision=precision,
        basic_rounding="explicit Context ROUND_FLOOR / ROUND_CEILING",
        extreme_inputs="Underflow and Subnormal trapped explicitly",
        exponential="correctly-rounded Decimal.exp expanded with next_minus/next_plus",
        square_root="expanded Decimal.sqrt; Fraction endpoint squares verified exactly",
        pi_identity="16*atan(1/5)-4*atan(1/239)",
        pi_series_terms_5=precision+20, pi_series_terms_239=precision//3+10,
        pi_lower=str(pi[0]), pi_upper=str(pi[1]),
        geometry="exact decimal L and Fraction midpoint separations",
        gaussian_lipschitz="exp(-1/2)/sqrt(2*pi), enclosed by positive interval product",
        test_vector="exact Decimal(str(float(q_i))); any strictly positive vector is valid",
        constants=dict(Amax="1.8", B="0.45", R="4", dv="45", M="6.75", eta="0.005"),
        documentation="https://docs.python.org/3.12/library/decimal.html",
        checks=self_check(),
    )


if __name__ == "__main__":
    import json
    print(json.dumps(arithmetic_audit(), indent=2))
