"""All-integer maxima of a coupled stationary-exclusion sufficient margin.

Only exact fractions and owned directed Decimal contexts enter these tests.
Finite input records must be reconstructed continuous spectral/water bounds;
this module checks their coverage and propagates their endpoints. It does not
claim a nonlinear persistence optimum or a dynamical extinction threshold.
"""
from decimal import Decimal,localcontext,ROUND_CEILING
from fractions import Fraction
from validated_gaussian import _Intervals
from validated_peaks import rayleigh_tail_certificate

if not __debug__:
    raise RuntimeError('Run validated certificates without -O.')


def _rational(value,name):
    if isinstance(value,bool) or isinstance(value,float):
        raise TypeError(name+' must be exact, not a binary float or boolean')
    try:
        answer=Fraction(value)
    except (TypeError,ValueError,ZeroDivisionError,OverflowError) as error:
        raise ValueError(name+' must be a finite exact rational') from error
    return answer


def _index(value,name):
    number=_rational(value,name)
    if number.denominator!=1 or number<2:
        raise ValueError(name+' must be an integer at least two')
    return int(number)


def _parameters(dv,Q,B,dw):
    result=tuple(_rational(v,n) for v,n in zip((dv,Q,B,dw),('dv','Q','B','dw')))
    if not (result[0]>0 and result[1]>0 and result[2]>=0 and result[3]>0):
        raise ValueError('requires dv>0, Q>0, B>=0, dw>0')
    return result


def _bounds(A,value):
    return A.fraction(value,value)


def _row_endpoints(row):
    beta_upper=_rational(row['beta_upper'],'beta_upper')
    q_lower=_rational(row['q_lower'],'q_lower')
    q_upper=_rational(row['q_upper'],'q_upper')
    if not (0<=q_lower<=q_upper<=1):
        raise ValueError('the comparison water factor must lie in [0,1]')
    if beta_upper>1:
        raise ValueError('the certified plant-loss upper endpoint must be at most one')
    return beta_upper,q_lower,q_upper


def coupled_margin_endpoints(beta_lower,beta_upper,q_lower,q_upper,
                             dv='.6',Q='7.2',B='.45',precision=60):
    """Direct endpoint propagation for Gamma=dv*beta+B-Q*q.

    beta_lower may be None for a one-sided competitor certificate. All other
    endpoints are required. Fraction arithmetic forms the expressions exactly
    before one directed conversion, so the ambient Decimal context is unused.
    """
    d,water,b,_=_parameters(dv,Q,B,'1')
    upper,ql,qu=_row_endpoints(dict(beta_upper=beta_upper,q_lower=q_lower,q_upper=q_upper))
    A=_Intervals(precision)
    margin_upper=_bounds(A,d*upper+b-water*ql)[1]
    lower_string=None
    if beta_lower is not None:
        lower=_rational(beta_lower,'beta_lower')
        if lower>upper:raise ValueError('the plant spectral interval is reversed')
        lower_string=str(_bounds(A,d*lower+b-water*qu)[0])
    return dict(dv=str(dv),Q=str(Q),B=str(B),precision=precision,
                margin_lower=lower_string,margin_upper=str(margin_upper),
                method='exact rational margin propagation followed by directed conversion')


def coupled_tail_certificate(envelope,M,dv='.6',B='.45',Q='7.2',dw='1'):
    """A bound on every coupled margin with integer m>=M.

    q_m>=0 and Q>0 allow dropping the nonpositive water term. The remaining
    decreasing beta majorant is the independently replayed Rayleigh tail.
    No assertion that the actual coupled margins are monotone is made.
    """
    d,water,b,diffusivity=_parameters(dv,Q,B,dw)
    spectral=rayleigh_tail_certificate(envelope,M)
    A=_Intervals(int(spectral['precision']))
    beta_tail=_rational(spectral['all_m_ge_anchor_beta_upper'],'beta tail')
    upper=_bounds(A,d*beta_tail+b)[1]
    return dict(**spectral,dv=str(dv),Q=str(Q),B=str(B),dw=str(dw),
        all_m_ge_anchor_margin_upper=str(upper),
        nonnegative_water_term_dropped=True,
        coupled_method='decreasing Rayleigh plant-loss upper bound; q_m>=0')


def _prepare(finite_rows,tail,winner_m):
    if not finite_rows:raise ValueError('finite continuum bounds are required')
    D,V=_rational(tail['D'],'D'),_rational(tail['V'],'V')
    M=_index(tail['anchor_m'],'anchor_m');winner_m=_index(winner_m,'winner_m')
    d,Q,B,dw=_parameters(tail['dv'],tail['Q'],tail['B'],tail['dw'])
    precision=max([int(tail['precision'])]+[int(r.get('precision',tail['precision'])) for r in finite_rows])
    bym={}
    for row in finite_rows:
        m=_index(row['m'],'m')
        if m in bym:raise ValueError('duplicate finite patch count')
        if (_rational(row['D'],'D'),_rational(row['V'],'V'))!=(D,V):
            raise ValueError('finite rows mix habitat geometries')
        for key,value in zip(('dv','Q','B','dw'),(d,Q,B,dw)):
            if key in row and _rational(row[key],key)!=value:
                raise ValueError('finite rows mix physical parameters: '+key)
        _row_endpoints(row);bym[m]=row
    if set(bym)!=set(range(2,M)):
        raise ValueError('finite rows must cover every integer 2<=m<anchor exactly once')
    if winner_m not in bym:raise ValueError('the candidate winner is outside finite coverage')
    winner=bym[winner_m]
    beta_w=_rational(winner['beta_lower'],'winner beta_lower')
    beta_w_upper,qw_lower,qw_upper=_row_endpoints(winner)
    if beta_w>beta_w_upper:raise ValueError('the winner spectral interval is reversed')
    if qw_upper<=0:raise ValueError('a positive winner water factor is required for the ratio sector')
    return dict(D=D,V=V,M=M,winner_m=winner_m,d=d,Q=Q,B=B,dw=dw,
                precision=precision,bym=bym,winner=winner,beta_w=beta_w,
                beta_w_upper=beta_w_upper,qw_lower=qw_lower,qw_upper=qw_upper)


def global_coupled_certificate(finite_rows,tail,winner_m):
    """Certify a unique all-integer maximiser and a robust Q/dv sector.

    Required finite fields: D,V,m,beta_upper,q_lower,q_upper; winner also
    beta_lower. Nominal dv,Q,B,dw are taken from the tail record; if provided
    on finite rows they must agree. Margin fields are rebuilt from these
    primitive endpoints, rather than trusting a precomputed claimed gap.
    """
    p=_prepare(finite_rows,tail,winner_m)
    winner_m=p['winner_m']
    D,V,M,d,Q,B,dw=(p[k] for k in ('D','V','M','d','Q','B','dw'))
    beta_w,qw_upper=p['beta_w'],p['qw_upper']
    nominal_ratio=Q/d
    winner_lower=d*beta_w+B-Q*qw_upper
    winner_upper=d*p['beta_w_upper']+B-Q*p['qw_lower']
    beta_tail=_rational(tail['all_m_ge_anchor_beta_upper'],'beta tail')
    tail_upper=d*beta_tail+B
    competitors=[];lower_thresholds=[(Fraction(0),'positive ratio')];upper_thresholds=[]
    for m,row in p['bym'].items():
        if m==winner_m:continue
        beta_upper,q_lower,q_upper=_row_endpoints(row)
        competitors.append((d*beta_upper+B-Q*q_lower,m))
        if m<winner_m:
            denominator=q_lower-qw_upper
            if denominator<=0:
                raise ArithmeticError('water intervals do not strictly separate the smaller count')
            lower_thresholds.append(((beta_upper-beta_w)/denominator,m))
        else:
            denominator=qw_upper-q_lower
            if denominator<=0:
                raise ArithmeticError('water intervals do not strictly separate the larger count')
            upper_thresholds.append(((beta_w-beta_upper)/denominator,m))
    if not competitors:raise ValueError('at least one finite competitor is required')
    finite_upper,runner=max(competitors)
    finite_gap=winner_lower-finite_upper;tail_gap=winner_lower-tail_upper
    if min(finite_gap,tail_gap)<=0:
        raise ArithmeticError('the strict all-integer coupled maximum is not separated')
    tail_ratio=(beta_w-beta_tail)/qw_upper
    upper_thresholds.append((tail_ratio,'tail'))
    ratio_lower,lower_active=max(lower_thresholds,key=lambda pair:pair[0])
    ratio_upper,upper_active=min(upper_thresholds,key=lambda pair:pair[0])
    if not ratio_lower<nominal_ratio<ratio_upper:
        raise ArithmeticError('the nominal reaction/dispersal ratio is outside its certified sector')
    A=_Intervals(p['precision']);gap=min(finite_gap,tail_gap)
    # Lower admissible thresholds are rounded UP; upper thresholds DOWN.
    ratio_lower_safe=_bounds(A,ratio_lower)[1]
    ratio_upper_safe=_bounds(A,ratio_upper)[0]
    if not Fraction(ratio_lower_safe)<nominal_ratio<Fraction(ratio_upper_safe):
        raise ArithmeticError('directed decimal ratio-sector endpoints do not separate the nominal ratio')
    eta=gap/(2*d)
    return dict(D=str(D),V=str(V),precision=p['precision'],dv=str(tail['dv']),Q=str(tail['Q']),
        B=str(tail['B']),dw=str(tail['dw']),winner_m=winner_m,
        finite_first_m=2,finite_last_m=M-1,finite_cases=len(p['bym']),tail_first_m=M,
        winner_beta_lower=str(p['winner']['beta_lower']),winner_beta_upper=str(p['winner']['beta_upper']),
        winner_q_lower=str(p['winner']['q_lower']),winner_q_upper=str(p['winner']['q_upper']),
        winner_margin_lower=str(_bounds(A,winner_lower)[0]),winner_margin_upper=str(_bounds(A,winner_upper)[1]),
        largest_competing_upper_m=runner,competing_finite_margin_upper=str(_bounds(A,finite_upper)[1]),
        infinite_tail_margin_upper=str(_bounds(A,tail_upper)[1]),
        finite_separation_lower=str(_bounds(A,finite_gap)[0]),
        tail_separation_lower=str(_bounds(A,tail_gap)[0]),overall_separation_lower=str(_bounds(A,gap)[0]),
        nominal_ratio_exact=str(nominal_ratio),ratio_lower_exact=str(ratio_lower),ratio_upper_exact=str(ratio_upper),
        ratio_lower_required=str(ratio_lower_safe),ratio_upper_allowed=str(ratio_upper_safe),
        ratio_lower_limiting_m=lower_active,ratio_upper_limiting_m=upper_active,
        tail_ratio_upper_exact=str(tail_ratio),
        kernel_eta_radius_strict=str(_bounds(A,eta)[0]),
        unique_global_maximum=True,ratio_sector_strict=True,ordering_independent_of_B=True,
        method='finite one-sided continuous spectral bounds; Rayleigh infinite tail; exact affine ratio comparisons')


def coupled_optimum_self_check(finite_rows,tail,winner_m):
    """Check endpoint propagation with fixed certified primitive inputs."""
    first=global_coupled_certificate(finite_rows,tail,winner_m)
    fine_tail=dict(tail,precision=int(first['precision'])+20)
    finer=global_coupled_certificate(finite_rows,fine_tail,winner_m)
    for lower,upper in [('winner_margin_lower','winner_margin_upper')]:
        assert Decimal(first[lower])<=Decimal(finer[lower])<=Decimal(finer[upper])<=Decimal(first[upper])
    for key in ('finite_separation_lower','tail_separation_lower','overall_separation_lower','kernel_eta_radius_strict'):
        assert Decimal(finer[key])>=Decimal(first[key])
    assert Decimal(finer['ratio_lower_required'])<=Decimal(first['ratio_lower_required'])
    assert Decimal(finer['ratio_upper_allowed'])>=Decimal(first['ratio_upper_allowed'])
    with localcontext() as alien:
        alien.prec=2;alien.rounding=ROUND_CEILING
        assert global_coupled_certificate(finite_rows,tail,winner_m)==first
    return dict(D=first['D'],winner_m=winner_m,precision_nesting=True,ambient_context_invariance=True,
                exact_rational_ratio_sector=True,complete_finite_coverage=True)


def asymptotic_scale_certificate(boundary_record,dw='1',reference_ratio='12'):
    """Propagate certified c_D endpoints through the large-ratio limit laws.

    For r=Q/dv, the proved asymptotics are m_r/r -> 2b/c_D and
    r*(max_m(beta_m-r*q_m)-beta_infty) -> c_D**2/(4b), with
    b=V**2/(8*dw). Only these LIMIT coefficients are enclosed here.
    Multiplication by reference_ratio supplies a leading-order estimate;
    it is not a finite-r localisation interval for an integer optimiser.
    In particular, no unproved second geometric coefficient is used.
    """
    D=_rational(boundary_record['D'],'D')
    V=_rational(boundary_record['V'],'V')
    c_lower=_rational(boundary_record['c_lower'],'c_lower')
    c_upper=_rational(boundary_record['c_upper'],'c_upper')
    diffusivity=_rational(dw,'dw')
    ratio=_rational(reference_ratio,'reference_ratio')
    if not (D>V>0 and 0<c_lower<=c_upper and diffusivity>0 and ratio>0):
        raise ValueError('requires D>V>0, 0<c_lower<=c_upper, dw>0 and reference_ratio>0')
    precision=int(boundary_record['precision'])
    A=_Intervals(precision)
    b=V*V/(8*diffusivity)
    scale_lower=2*b/c_upper
    scale_upper=2*b/c_lower
    gain_lower=c_lower*c_lower/(4*b)
    gain_upper=c_upper*c_upper/(4*b)
    return dict(D=str(boundary_record['D']),V=str(boundary_record['V']),
        N=int(boundary_record['N']),precision=precision,dw=str(dw),b_exact=str(b),
        c_lower=str(boundary_record['c_lower']),c_upper=str(boundary_record['c_upper']),
        m_over_ratio_lower=str(_bounds(A,scale_lower)[0]),
        m_over_ratio_upper=str(_bounds(A,scale_upper)[1]),
        scaled_gain_limit_lower=str(_bounds(A,gain_lower)[0]),
        scaled_gain_limit_upper=str(_bounds(A,gain_upper)[1]),
        reference_ratio=str(reference_ratio),
        leading_m_lower=str(_bounds(A,ratio*scale_lower)[0]),
        leading_m_upper=str(_bounds(A,ratio*scale_upper)[1]),
        finite_ratio_localization=False,
        method='exact rational propagation of certified c_D through proved asymptotic limit coefficients')
