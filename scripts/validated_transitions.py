"""All-count local phase diagrams for the scalar coupled exclusion margin.

Primitive continuous beta/q endpoints are exact decimal data. All comparisons
are exact Fraction operations; only displayed endpoints use directed Decimal.
This module does not locate a nonlinear persistence bifurcation.
"""
from decimal import Decimal,localcontext,ROUND_CEILING
from fractions import Fraction
from validated_gaussian import _Intervals

if not __debug__:
    raise RuntimeError('Run validated certificates without -O.')


def _f(value,name):
    if isinstance(value,(float,bool)):
        raise TypeError(name+' must be exact, not a float or boolean')
    try:return Fraction(value)
    except (ValueError,TypeError,ZeroDivisionError,OverflowError) as error:
        raise ValueError(name+' must be a finite rational') from error


def _m(value):
    answer=_f(value,'m')
    if answer.denominator!=1 or answer<2:
        raise ValueError('patch count must be an integer at least two')
    return int(answer)


def _row(row,two_sided=False):
    D,V,dw=(_f(row[key],key) for key in ('D','V','dw'))
    m=_m(row['m'])
    betaU,ql,qu=(_f(row[key],key) for key in ('beta_upper','q_lower','q_upper'))
    if not (D>V>0 and dw>0 and 0<=betaU<=1 and 0<ql<=qu<1):
        raise ValueError('invalid habitat, spectral or water endpoints')
    betaL=None
    if two_sided:
        betaL=_f(row['beta_lower'],'beta_lower')
        if not 0<=betaL<=betaU:raise ValueError('reversed spectral interval')
    return dict(D=D,V=V,dw=dw,m=m,betaL=betaL,betaU=betaU,ql=ql,qu=qu)


def _dec(A,value,upper=False):
    return str(A.fraction(value,value)[int(upper)])


def transition_pair_certificate(left,right,precision=60):
    """Enclose the true unique crossing of two consecutive affine objectives."""
    a,b=_row(left,True),_row(right,True)
    if (a['D'],a['V'],a['dw'])!=(b['D'],b['V'],b['dw']) or b['m']!=a['m']+1:
        raise ValueError('crossing requires consecutive counts in one geometry')
    nl=a['betaL']-b['betaU'];nu=a['betaU']-b['betaL']
    dl=a['ql']-b['qu'];du=a['qu']-b['ql']
    if not (0<nl<=nu and 0<dl<=du):
        raise ArithmeticError('positive crossing numerator/denominator not separated')
    lower,upper=nl/du,nu/dl
    A=_Intervals(precision)
    return dict(D=str(a['D']),V=str(a['V']),dw=str(a['dw']),precision=precision,
        left_m=a['m'],right_m=b['m'],
        beta_difference_lower=_dec(A,nl),beta_difference_upper=_dec(A,nu,True),
        q_difference_lower=_dec(A,dl),q_difference_upper=_dec(A,du,True),
        crossing_lower_exact=str(lower),crossing_upper_exact=str(upper),
        crossing_lower=_dec(A,lower),crossing_upper=_dec(A,upper,True),
        crossing_width_upper=_dec(A,upper-lower,True),
        pair_crossing_unique=True,
        method='positive exact affine numerator/denominator; directed quotient endpoints')


def global_transition_certificate(finite_rows,tail,candidates,window,precision=60):
    """Certify an exact local 3-phase diagram over all integer competitors.

Returns (flat summary, complete comparison audit). Candidate counts must be
three consecutive integers. Every finite 2<=m<M needs a beta upper bound and
water enclosure; the three candidates need two-sided beta enclosures.
"""
    if len(candidates)!=3 or len(window)!=2:
        raise ValueError('three candidate counts and two window endpoints required')
    counts=tuple(_m(m) for m in candidates)
    if counts!=(counts[0],counts[0]+1,counts[0]+2):
        raise ValueError('phase candidates must be consecutive and ordered')
    L,U=(_f(v,'ratio window') for v in window)
    if not 0<L<U:raise ValueError('positive ordered ratio window required')
    D,V,dw=(_f(tail[key],key) for key in ('D','V','dw'))
    M=_m(tail['anchor_m']);T=_f(tail['all_m_ge_anchor_beta_upper'],'tail beta upper')
    if not (D>V>0 and dw>0 and 0<=T<=1):raise ValueError('invalid tail inputs')
    bym={};source={}
    for row in finite_rows:
        r=_row(row,_m(row['m']) in counts)
        m=r['m']
        if m in bym:raise ValueError('duplicate finite patch count')
        if (r['D'],r['V'],r['dw'])!=(D,V,dw):raise ValueError('mixed geometry or diffusivity')
        bym[m]=r;source[m]=row
    if set(bym)!=set(range(2,M)):
        raise ValueError('finite rows must cover exactly every 2<=m<M')
    if not 2<=counts[0]<counts[-1]<M:
        raise ValueError('candidate range is outside finite coverage')
    pairs=[transition_pair_certificate(source[a],source[b],precision)
           for a,b in zip(counts,counts[1:])]
    if not (L<Fraction(pairs[0]['crossing_lower_exact'])
        <=Fraction(pairs[0]['crossing_upper_exact'])
        <Fraction(pairs[1]['crossing_lower_exact'])
        <=Fraction(pairs[1]['crossing_upper_exact'])<U):
        raise ArithmeticError('the two true switching brackets are not ordered inside the window')
    first,last=bym[counts[0]],bym[counts[-1]]
    small=[];large=[];A=_Intervals(precision)
    for m,r in bym.items():
        if m<counts[0]:
            if r['ql']<=first['qu']:raise ArithmeticError('smaller-count water ordering not separated')
            gap=first['betaL']-r['betaU']+L*(r['ql']-first['qu'])
            if gap<=0:raise ArithmeticError('left endpoint does not separate count '+str(m))
            small.append(dict(m=m,ratio_exact=str(L),gap_exact=str(gap),gap_lower=_dec(A,gap)))
        if m>counts[-1]:
            if last['ql']<=r['qu']:raise ArithmeticError('larger-count water ordering not separated')
            gap=last['betaL']-r['betaU']-U*(last['qu']-r['ql'])
            if gap<=0:raise ArithmeticError('right endpoint does not separate count '+str(m))
            large.append(dict(m=m,ratio_exact=str(U),gap_exact=str(gap),gap_lower=_dec(A,gap)))
    if not small or not large:raise ValueError('this implementation requires finite competitors on both sides')
    tailgap=last['betaL']-U*last['qu']-T
    if tailgap<=0:raise ArithmeticError('right endpoint does not separate the infinite tail')
    mingapS,activeS=min((Fraction(r['gap_exact']),r['m']) for r in small)
    mingapL,activeL=min((Fraction(r['gap_exact']),r['m']) for r in large)
    summary=dict(D=str(D),V=str(V),dw=str(dw),precision=precision,
        first_candidate=counts[0],middle_candidate=counts[1],last_candidate=counts[2],
        ratio_window_lower=str(window[0]),ratio_window_upper=str(window[1]),
        finite_first_m=2,finite_last_m=M-1,finite_cases=len(bym),tail_first_m=M,
        smaller_cases=len(small),larger_cases=len(large),
        minimum_smaller_gap_lower=_dec(A,mingapS),active_smaller_m=activeS,
        minimum_larger_gap_lower=_dec(A,mingapL),active_larger_m=activeL,
        infinite_tail_gap_lower=_dec(A,tailgap),
        first_switch_lower=pairs[0]['crossing_lower'],first_switch_upper=pairs[0]['crossing_upper'],
        second_switch_lower=pairs[1]['crossing_lower'],second_switch_upper=pairs[1]['crossing_upper'],
        all_integer_phase_diagram=True,exactly_two_switches_in_window=True,
        two_maximisers_at_each_switch=True,unique_elsewhere_in_window=True,
        ordering_independent_of_B_and_positive_dv=True,
        method='monotone affine endpoint dominance; two ordered exact crossing brackets; Rayleigh infinite tail')
    audit=dict(summary=summary,pairs=pairs,smaller_competitors=small,larger_competitors=large,
        infinite_tail=dict(anchor_m=M,beta_upper_exact=str(T),ratio_exact=str(U),
                           gap_exact=str(tailgap),gap_lower=_dec(A,tailgap)))
    return summary,audit


def transition_self_check(finite_rows,tail,candidates,window,precision=60):
    first,audit=global_transition_certificate(finite_rows,tail,candidates,window,precision)
    finer,fine_audit=global_transition_certificate(finite_rows,tail,candidates,window,precision+20)
    for low,high in [('first_switch_lower','first_switch_upper'),('second_switch_lower','second_switch_upper')]:
        assert Decimal(first[low])<=Decimal(finer[low])<=Decimal(finer[high])<=Decimal(first[high])
    for key in ('minimum_smaller_gap_lower','minimum_larger_gap_lower','infinite_tail_gap_lower'):
        assert Decimal(first[key])<=Decimal(finer[key])
    with localcontext() as alien:
        alien.prec=2;alien.rounding=ROUND_CEILING
        assert global_transition_certificate(finite_rows,tail,candidates,window,precision)==(first,audit)
    return dict(D=first['D'],precision_nesting=True,ambient_context_invariance=True,
        all_finite_counts_covered=True,analytic_infinite_tail=True,
        exact_pair_crossings_and_global_phase=True)
