"""Independent coordinate/summation audit for selected fragment certificates.

The optimized certifier uses a block catalog and reflection-reduced rows. This
check reconstructs every node from the left domain endpoint, evaluates every
matrix entry separately, and sums the endpoint products as exact Fractions.
Only the Gaussian interval primitives (pi/exp) are shared. This is an independent
geometry and matrix-summation check, not a second interval-library implementation.
"""
from decimal import Decimal
from fractions import Fraction
from validated_gaussian import _Intervals, _pi_fraction_bounds
from validated_peaks import fragment_certificate


def independent_geometry_check(D, m, r, q, precision=60, V="2"):
    """Cross-check all full CW rows against the optimized exact-vector result.

    The Fraction endpoint sums incur no intermediate rounding; their resulting
    CW interval must be contained in the sequentially rounded optimized bracket.
    This additionally avoids depending on identical forward/reverse row sums.
    Selected small cases are recommended because every entry is recomputed.
    """
    optimized = fragment_certificate(D, m, r, q, precision, V)
    A = _Intervals(precision)
    DF, VF = Fraction(D), Fraction(V)
    ell = VF/m
    gap = (DF-VF)/(m-1)
    h = ell/r
    # This expression is deliberately independent of centered block differences.
    nodes = [-DF/2 + patch*(ell+gap) + (Fraction(cell)+Fraction(1, 2))*h
             for patch in range(m) for cell in range(r)]
    n = m*r
    assert nodes[0]-h/2 == -DF/2
    assert nodes[-1]+h/2 == DF/2
    assert n*h == VF
    assert nodes == [-x for x in nodes[::-1]]
    for i in range(n-1):
        expected = h+gap if (i+1) % r == 0 else h
        assert nodes[i+1]-nodes[i] == expected

    pi = A.fraction(*_pi_fraction_bounds(precision))
    J0 = A.divide(A.point(1), A.sqrt(A.multiply(A.point(2), pi)))
    weighted_J0 = A.multiply(A.fraction(h, h), J0)
    exact_q = [Fraction(Decimal(str(value))) for value in q]
    lower_ratios, upper_ratios = [], []
    for x, qi in zip(nodes, exact_q, strict=True):
        lower_sum = upper_sum = Fraction(0)
        for y, qj in zip(nodes, exact_q, strict=True):
            exponent = -(x-y)**2/2
            entry = A.multiply(weighted_J0, A.exp(A.fraction(exponent, exponent)))
            lower_sum += Fraction(entry[0])*qj
            upper_sum += Fraction(entry[1])*qj
        lower_ratios.append(lower_sum/qi)
        upper_ratios.append(upper_sum/qi)
    direct = A.fraction(min(lower_ratios), max(upper_ratios))
    opt_lower, opt_upper = Decimal(optimized["cw_lower"]), Decimal(optimized["cw_upper"])
    assert opt_lower <= direct[0] <= direct[1] <= opt_upper
    return dict(D=D, V=V, m=m, cells_per_patch=r, N=n, precision=precision,
                full_rows_evaluated=n, separately_evaluated_kernel_entries=n*n,
                exact_total_cell_measure=str(VF),
                full_coordinate_cw_lower=str(direct[0]),
                full_coordinate_cw_upper=str(direct[1]),
                optimized_cw_lower=str(opt_lower), optimized_cw_upper=str(opt_upper),
                full_bracket_contained_in_optimized=True,
                exact_endpoint_span_and_patch_geometry=True,
                exact_fraction_matrix_summation=True,
                q_sha256=optimized["q_sha256"],
                method="full endpoint-based coordinates; separate Gaussian entries; exact Fraction sums")
