"""Uniform cubic Gaussian fragmentation bounds on one Hilbert feature space.

Only standard-library arithmetic and the independently replayed Gaussian-rank
certificate are used. The positive trace-class weak-measure expansion removes
the C3 function-space norm conversions. It does not estimate an optimal error
constant or identify a finite-ratio optimiser.
Lower endpoints enclose the displayed positive majorant expressions; they
are not lower bounds for the actual operator norm or spectral error.
"""
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from hashlib import sha256

from validated_gaussian import _Intervals, _pi_fraction_bounds
from validated_gaussian_rank import gaussian_rank_certificate

if not __debug__:
    raise RuntimeError('Run validated certificates without -O.')


def _geometry(value, name):
    if not isinstance(value, str):
        raise TypeError(name+' must be an exact finite decimal string')
    decimal = Decimal(value)
    if not decimal.is_finite() or decimal <= 0:
        raise ValueError(name+' must be finite and positive')
    return Fraction(decimal)


def _anchor(value):
    if not isinstance(value, int) or isinstance(value, bool) or value < 2:
        raise ValueError('the anchor must be an integer at least two')
    return value


def _exact(A, value):
    value = Fraction(value)
    return A.fraction(value, value)


def _sum(A, values):
    result = A.point(0)
    for value in values:
        result = A.add(result, value)
    return result


def _build_base(D_string, p, trial, precision=60, V_string='2'):
    D, V = _geometry(D_string, 'D'), _geometry(V_string, 'V')
    if not D > V:
        raise ValueError('requires D>V>0')
    if not isinstance(precision, int) or isinstance(precision, bool) or precision < 30:
        raise ValueError('precision must be an integer at least thirty')
    spectral, rankaudit = gaussian_rank_certificate(
        D_string, p, trial, precision, None, V_string)
    theta, delta = V/D, D-V
    # This is a CHOSEN exact rational lower bound, not an interval estimate
    # of the actual even gap. The source quantities are replayed above.
    rho_lower = Fraction(spectral['rho_lower'])
    even_rest_upper = Fraction(spectral['even_rest_upper'])
    gamma = theta*(rho_lower-even_rest_upper)
    if gamma <= 0:
        raise ArithmeticError('the continuous even-sector gap must be positive')
    A = _Intervals(precision)
    pi = A.fraction(*_pi_fraction_bounds(precision))
    J0 = A.divide(A.point(1), A.sqrt(A.multiply(A.point(2), pi)))
    sqrt3, sqrt15 = A.sqrt(A.point(3)), A.sqrt(A.point(15))
    moments = [J0, A.multiply(A.point(2), J0),
        A.multiply(A.multiply(A.point(2), J0), A.add(sqrt3, A.point(1))),
        A.multiply(J0, A.add(A.multiply(A.point(2), sqrt15),
                             A.multiply(A.point(6), sqrt3)))]
    weak = [4*theta**3*(1-theta)*D,
        Fraction(2, 3)*theta*(1-theta)*V**2+V*delta/3,
        V**3*(theta/12+Fraction(1, 6))+V**2*D/6,
        2*V*D**3/3+V**3*D/48+V**4/192]
    CH = _sum(A, [A.multiply(_exact(A, c), moment)
                  for c, moment in zip(weak, moments)])
    kappa = V*D*(1-theta)*(1-2*theta)/12
    a1 = A.multiply(_exact(A, 2*theta*delta), J0)
    a2 = A.add(A.multiply(_exact(A, theta), a1),
              A.multiply(_exact(A, 2*abs(kappa)), moments[1]))
    bounds = {name: (Fraction(interval[0]), Fraction(interval[1]))
              for name, interval in [('J0', J0), ('a1', a1), ('a2', a2), ('CH', CH)]}
    bounds['s'] = (1/gamma, 1/gamma)
    assert all(0 < low <= high for low, high in bounds.values())
    canonical = '\n'.join(trial['even_q'])+'\n--odd--\n'+\
        '\n'.join(trial['odd_q'])+'\n--shift--\n'+str(trial['ordering_shift'])
    return dict(D=D_string, V=V_string, p=p, precision=precision,
        gamma=gamma, bounds=bounds, weak=weak, derivative_bounds=moments,
        kappa=kappa, spectral=spectral, rankaudit=rankaudit,
        source_sha256=sha256(canonical.encode('ascii')).hexdigest())


def _side(base, M, side):
    a1, a2, CH, s = (base['bounds'][name][side]
                      for name in ('a1', 'a2', 'CH', 's'))
    e = a1+a2/M+CH/M**2
    f = a2+CH/M
    eta = 2*s*e/M
    return e, f, eta


def _minimum_anchor(base):
    trace = []
    def good(M, stage):
        eta = _side(base, M, 1)[2]
        result = eta <= Fraction(1, 2)
        trace.append(dict(stage=stage, M=M,
            eta_lower_exact=str(_side(base, M, 0)[2]),
            eta_upper_exact=str(eta), passed=result))
        return result
    low = high = 2
    while not good(high, 'doubling'):
        high *= 2
    while low < high:
        middle = (low+high)//2
        if good(middle, 'bisection'):
            high = middle
        else:
            low = middle+1
    assert good(low, 'final')
    assert low == 2 or not good(low-1, 'predecessor')
    if low > 2 and _side(base, low-1, 0)[2] <= Fraction(1, 2):
        raise ArithmeticError('predecessor minimality not resolved; increase precision')
    return low, trace


def _from_base(base, M):
    M = _anchor(M)
    low, high = _side(base, M, 0), _side(base, M, 1)
    if not 0 < low[2] <= high[2] < 1:
        raise ValueError('the chosen anchor does not certify eta_M<1')
    def budget(side, parts):
        a1, CH, s = (base['bounds'][name][side] for name in ('a1', 'CH', 's'))
        e, f, eta = parts
        return CH+s*f*e/(1-eta)+2*a1*s*s*e*e/(1-eta)+a1*s*f
    lower, upper = budget(0, low), budget(1, high)
    assert 0 < lower <= upper
    A = _Intervals(base['precision'])
    spectral = base['spectral']
    result = dict(D=base['D'], V=base['V'], features=base['p'],
        precision=base['precision'], M=M,
        gamma_lower_exact=str(base['gamma']),
        gamma_lower=str(_exact(A, base['gamma'])[0]),
        rho_source_lower=spectral['rho_lower'],
        rho_source_upper=spectral['rho_upper'],
        even_rest_source_upper=spectral['even_rest_upper'],
        source_feature_tail_upper=spectral['feature_tail_upper'],
        source_ordering_shift=spectral['ordering_shift'],
        source_sha256=base['source_sha256'],
        valid_for_every_m_ge_M=True,
        endpoints_enclose_majorant_expressions=True,
        K3_lower_is_actual_error_lower_bound=False,
        half_condition=high[2] <= Fraction(1, 2),
        coefficient_identity='uniqueness of the two actual habitat spectral expansion coefficients',
        uses_full_spectral_gap=False, uses_even_spectral_gap=True,
        signed_second_coefficient_computed=False, finite_ratio_localization=False,
        method='trace-class weak measure; common Hilbert Gram space; exact rational Neumann budget')
    for name, (lo, hi) in base['bounds'].items():
        interval = A.fraction(lo, hi)
        result[name+'_lower'] = str(interval[0])
        result[name+'_upper'] = str(interval[1])
        result[name+'_lower_exact'] = str(lo)
        result[name+'_upper_exact'] = str(hi)
    for name, lo, hi in [('eM', low[0], high[0]), ('fM', low[1], high[1]),
                         ('eta', low[2], high[2]), ('K3', lower, upper)]:
        interval = A.fraction(lo, hi)
        result[name+'_lower_exact'], result[name+'_upper_exact'] = str(lo), str(hi)
        result[name+'_lower'], result[name+'_upper'] = str(interval[0]), str(interval[1])
    remainder = upper/M**3
    result['anchor_error_upper_exact'] = str(remainder)
    result['anchor_error_upper'] = str(_exact(A, remainder)[1])
    for j, (weak, derivative) in enumerate(zip(base['weak'], base['derivative_bounds'])):
        result['weak_C'+str(j)+'_exact'] = str(weak)
        result['derivative_M'+str(j)+'_upper'] = str(derivative[1])
    return result


def hilbert_remainder_certificate(D_string, p, trial, M, precision=60, V_string='2'):
    """Rebuild the continuum source and return one uniform certificate and audit.

    The source trial has the same structure as gaussian_rank_certificate.
    Its inertia signs, continuous tail and spectral bounds are replayed; a
    stored approximate eigengap is never accepted as proof input.
    """
    base = _build_base(D_string, p, trial, precision, V_string)
    result = _from_base(base, M)
    audit = dict(summary=result, spectral=base['rankaudit'], trial=trial)
    return result, audit


def hilbert_remainder_family(D_string, p, trial, anchors=(70, 80, 160, 320),
                             precision=60, V_string='2'):
    """Share one source replay among several anchors and the exact half search."""
    for M in anchors:
        _anchor(M)
    base = _build_base(D_string, p, trial, precision, V_string)
    minimum, search = _minimum_anchor(base)
    requested = sorted(set([minimum, 2*minimum]+list(anchors)))
    records = [_from_base(base, M) for M in requested]
    audit = dict(D=D_string, V=V_string, features=p, precision=precision,
        minimum_half_anchor=minimum, search=search,
        spectral=base['rankaudit'], trial=trial)
    return records, audit


def hilbert_remainder_at_count(record, m):
    """Evaluate the already certified uniform bound at an admissible count."""
    m = _anchor(m)
    if m < int(record['M']):
        raise ValueError('count lies below the certified anchor')
    upper = Fraction(record['K3_upper_exact'])/m**3
    A = _Intervals(int(record['precision']))
    return dict(D=record['D'], V=record['V'], M=int(record['M']), m=m,
        remainder_upper_exact=str(upper), remainder_upper=str(_exact(A, upper)[1]),
        valid_second_order_remainder=True, finite_ratio_localization=False)


def hilbert_remainder_self_check(D_string, p, trial, precision=60, V_string='2'):
    first, audit = hilbert_remainder_family(D_string, p, trial, (), precision, V_string)
    minimum = audit['minimum_half_anchor']
    coarse = first[0]
    finer, _ = hilbert_remainder_certificate(D_string, p, trial, minimum,
                                             precision+20, V_string)
    # The chosen rational gamma becomes sharper when the source is refined;
    # consequently s and the positive majorant upper endpoints can shrink.
    for name in ('J0', 'a1', 'a2', 'CH', 'eM', 'fM', 'eta', 'K3'):
        assert Fraction(finer[name+'_upper_exact']) <= Fraction(coarse[name+'_upper_exact']), name
    assert Fraction(finer['gamma_lower_exact']) >= Fraction(coarse['gamma_lower_exact'])
    with localcontext() as foreign:
        foreign.prec = 2
        foreign.rounding = ROUND_CEILING
        alien, _ = hilbert_remainder_certificate(D_string, p, trial, minimum,
                                                  precision, V_string)
        assert alien == coarse
    base = _build_base(D_string, p, trial, precision, V_string)
    successive = [_from_base(base, minimum*factor) for factor in (1, 2, 4, 8)]
    for one, two in zip(successive, successive[1:]):
        assert Fraction(two['eta_upper_exact']) < Fraction(one['eta_upper_exact'])
        assert Fraction(two['K3_upper_exact']) < Fraction(one['K3_upper_exact'])
    assert _side(base, minimum, 1)[2] <= Fraction(1, 2)
    assert minimum == 2 or _side(base, minimum-1, 0)[2] > Fraction(1, 2)
    for bad in (1, 0, -1, True, 2.5, '70'):
        try:
            _from_base(base, bad)
        except (ValueError, TypeError):
            pass
        else:
            raise AssertionError('an invalid anchor was accepted')
    try:
        _from_base(base, 2)
    except ValueError:
        pass
    else:
        # In a geometry where anchor2 happens to be admissible, there is no
        # failed Neumann screen to require here.
        assert _side(base, 2, 1)[2] < 1
    try:
        hilbert_remainder_at_count(coarse, minimum-1)
    except ValueError:
        pass
    else:
        raise AssertionError('an out-of-range count was accepted')
    return dict(D=D_string, V=V_string, features=p, precision=precision,
        minimum_half_anchor=minimum, exact_predecessor_check=True,
        source_replayed=True, upper_precision_nesting=True,
        ambient_context_invariance=True, monotone_anchor_refinement=True,
        invalid_inputs_rejected=True, uniform_for_all_counts=True,
        finite_ratio_localization=False)
