"""Continuous Gaussian spectral and finite-count expansion-error certificates.

Only standard-library directed arithmetic enters proof replay. Saved envelope
predictors are arbitrary exact trials; their preparation is not trusted.
"""
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from hashlib import sha256
from validated_gaussian import _Intervals
from validated_fragment_principal import _aggregate_gram_enclosure
from fragment_predictor_trial import predictor_vector

if not __debug__:
    raise RuntimeError('Run proof replay without -O.')


def _exact(A, x):
    x = Fraction(x)
    return A.fraction(x, x)


def _sum(A, values):
    answer = A.point(0)
    for value in values:
        answer = A.add(answer, value)
    return answer


def _square(A, x):
    upper = max(A.up.multiply(a, a) for a in x)
    lower = Decimal(0) if x[0] <= 0 <= x[1] else min(A.down.multiply(a, a) for a in x)
    return lower, upper


def finite_trace_certificate(matrix, q_exact, precision=60):
    """PSD finite-matrix Temple enclosure; PSD is supplied by the Gram proof."""
    A = _Intervals(precision)
    n = len(matrix)
    if not n or len(q_exact) != n or any(len(row) != n for row in matrix):
        raise ValueError('nonempty square matrix and matching trial required')
    norm_exact = sum((x*x for x in q_exact), Fraction(0))
    if norm_exact <= 0:
        raise ValueError('nonzero trial required')
    q = [_exact(A, x) for x in q_exact]
    y = [_sum(A, [A.multiply(a, b) for a, b in zip(row, q)]) for row in matrix]
    rayleigh = A.divide(_sum(A, [A.multiply(a, b) for a, b in zip(q, y)]), _exact(A, norm_exact))
    trace = _sum(A, [matrix[i][i] for i in range(n)])
    residual = [A.subtract(yi, A.multiply(rayleigh, qi)) for yi, qi in zip(y, q)]
    residual_squared = A.divide(_sum(A, [_square(A, x) for x in residual]), _exact(A, norm_exact))
    separation = A.down.subtract(A.down.multiply(Decimal(2), rayleigh[0]), trace[1])
    if separation <= 0:
        raise ArithmeticError('trace screen unresolved: 2*Rayleigh_lower must exceed trace_upper')
    correction = A.up.divide(residual_squared[1], separation)
    upper = A.up.add(rayleigh[1], correction)
    return dict(rayleigh=rayleigh, trace=trace, residual=residual,
                residual_squared=residual_squared, separation=separation,
                correction=correction, finite_upper=upper, norm_exact=norm_exact)


def predictor_certificate(D_string, m, p, trial, precision=60, V_string='2', corrected=True):
    if trial['D'] != D_string or trial['V'] != V_string or trial['features'] != p:
        raise ValueError('predictor metadata must match the exact geometry and feature cutoff')
    gram = _aggregate_gram_enclosure(D_string, p, precision, m, V_string)
    q = predictor_vector(trial, m, corrected)
    finite = finite_trace_certificate(gram['blocks'][0], q, precision)
    A = _Intervals(precision)
    lower = finite['rayleigh'][0]
    upper = min(Decimal(1), A.up.add(finite['finite_upper'], gram['tail']))
    if not Decimal(0) < lower <= upper <= Decimal(1):
        raise ArithmeticError('inconsistent continuous principal enclosure')
    beta = (A.down.subtract(Decimal(1), upper), A.up.subtract(Decimal(1), lower))
    canonical = [str(x) for x in q]
    row = dict(D=D_string, V=V_string, m=m, features=p, precision=precision,
        predictor_order=1 if corrected else 0, even_features=len(q),
        rho_lower=str(lower), rho_upper=str(upper), beta_lower=str(beta[0]), beta_upper=str(beta[1]),
        beta_width_upper=str(A.up.subtract(beta[1], beta[0])),
        rayleigh_lower=str(finite['rayleigh'][0]), rayleigh_upper=str(finite['rayleigh'][1]),
        finite_trace_upper=str(finite['trace'][1]), trace_separation_lower=str(finite['separation']),
        residual_squared_upper=str(finite['residual_squared'][1]),
        quadratic_correction_upper=str(finite['correction']), feature_tail_upper=str(gram['tail']),
        q_norm_squared_exact=str(finite['norm_exact']),
        q_sha256=sha256('\n'.join(canonical).encode('ascii')).hexdigest(),
        trial_from_envelope_only=True, uses_fragment_eigensolve=False,
        uses_quantified_gap=False, uses_inertia=False, is_pointwise_certificate=True,
        uniform_cubic_tail_certified=False,
        method='exact aggregate Gaussian moments; finite PSD trace-Temple; continuous positive feature tail')
    audit = dict(summary=row, exact_trial=canonical,
        even_moments=[[str(a), str(b)] for a, b in gram['moments']],
        residual_intervals=[[str(a), str(b)] for a, b in finite['residual']],
        trace_interval=list(map(str, finite['trace'])),
        taylor_upper_degree=gram['taylor_upper_degree'])
    return row, audit


def second_remainder_certificate(row, signed):
    if row['D'] != signed['D'] or row['V'] != signed['V']:
        raise ValueError('signed coefficient geometry mismatch')
    A = _Intervals(int(row['precision']))
    m = int(row['m']); theta = Fraction(row['V'])/Fraction(row['D'])
    rho = tuple(Decimal(signed[k]) for k in ('rho_lower', 'rho_upper'))
    c = tuple(Decimal(signed[k]) for k in ('c_lower', 'c_upper'))
    d = tuple(Decimal(signed[k]) for k in ('d_lower', 'd_upper'))
    beta0 = A.subtract(A.point(1), A.multiply(_exact(A, theta), rho))
    expansion = A.add(beta0, A.add(A.multiply(_exact(A, Fraction(1, m)), c),
                                 A.multiply(_exact(A, Fraction(1, m*m)), d)))
    beta = tuple(Decimal(row[k]) for k in ('beta_lower', 'beta_upper'))
    remainder = A.subtract(beta, expansion)
    scaled = A.multiply(_exact(A, m**3), remainder)
    return dict(beta_infty_lower=str(beta0[0]), beta_infty_upper=str(beta0[1]),
        second_expansion_lower=str(expansion[0]), second_expansion_upper=str(expansion[1]),
        second_remainder_lower=str(remainder[0]), second_remainder_upper=str(remainder[1]),
        scaled_cubic_lower=str(scaled[0]), scaled_cubic_upper=str(scaled[1]),
        remainder_sign_certified_positive=remainder[0] > 0,
        signed_source_features=signed['features'], signed_source_q_sha256=signed['q_sha256'],
        finite_count_error_only=True)


def predictor_self_check(D, m, p, trial, signed, precision=60):
    first, _ = predictor_certificate(D, m, p, trial, precision)
    finer, _ = predictor_certificate(D, m, p, trial, precision+20)
    for lo, hi in [('rho_lower','rho_upper'), ('beta_lower','beta_upper')]:
        assert Decimal(first[lo]) <= Decimal(finer[lo]) <= Decimal(finer[hi]) <= Decimal(first[hi])
    remainder = second_remainder_certificate(first, signed)
    with localcontext() as alien:
        alien.prec=2;alien.rounding=ROUND_CEILING
        repeated, _ = predictor_certificate(D, m, p, trial, precision)
        assert repeated == first
        assert second_remainder_certificate(repeated, signed) == remainder
    return dict(D=D, m=m, features=p, precision_nesting=True, ambient_context_invariance=True,
                finite_count_only=True, uses_numpy_in_replay=False)
