"""Finite-rank continuous Gaussian spectral certificates (standard library).

The feature expansion is positive semidefinite term by term. Positive-series
moment enclosures produce interval Gram matrices; Collatz bounds enclose the
two parity Perron values, and a validated LDL inertia count identifies the
full second eigenvalue. No floating eigenvalue is an input to an inclusion
bound. Exact saved positive trial strings and a rational ordering shift are
proof inputs whose correctness is checked, not assumed.
"""
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from hashlib import sha256
from math import factorial

from validated_gaussian import _Intervals, _pi_fraction_bounds


def _exact(A, value):
    value=Fraction(value)
    return A.fraction(value,value)


def _positive_multiply(A,a,b):
    assert a[0]>=0 and b[0]>=0
    return A.down.multiply(a[0],b[0]),A.up.multiply(a[1],b[1])


def _positive_square(A,a):
    if a[0]<=0<=a[1]:
        return Decimal(0),max(A.up.multiply(a[0],a[0]),A.up.multiply(a[1],a[1]))
    return (min(A.down.multiply(a[0],a[0]),A.down.multiply(a[1],a[1])),
            max(A.up.multiply(a[0],a[0]),A.up.multiply(a[1],a[1])))


def _half_intervals(D,V,m):
    """Positive-half intervals, all geometry retained as exact fractions."""
    if m is None:
        return [(Fraction(0),D/2)],D
    if not isinstance(m,int) or isinstance(m,bool) or m<2 or not D>V>0:
        raise ValueError('fragment geometry requires integer m>=2 and D>V>0')
    ell=V/m;step=ell+(D-V)/(m-1)
    result=[]
    for i in range(m):
        centre=(Fraction(i)-Fraction(m-1,2))*step
        lower,upper=centre-ell/2,centre+ell/2
        if upper>0:
            result.append((max(Fraction(0),lower),upper))
    assert sum((b-a for a,b in result),Fraction(0))==V/2
    return result,V


def _positive_moment(A,n,a):
    r"""Enclose int_0^a exp(-x*x)*x**(2*n) dx by a positive series.

    M_n=a**(2n+1)*exp(-a*a)*sum(t_j), t_0=1/(2n+1),
    t_(j+1)/t_j=2*a*a/(2n+2j+3). Decreasing ratios below one give a
    geometric upper tail. The requested series target affects tightness,
    never inclusion.
    """
    if a==0:return (Decimal(0),Decimal(0)),0
    if a<0 or n<0:raise ValueError('nonnegative moment geometry required')
    term=_exact(A,Fraction(1,2*n+1));total=term
    target=Decimal('1e-'+str(A.near.prec-12))
    a2=a*a
    for j in range(10000):
        ratio=_exact(A,2*a2/Fraction(2*n+2*j+3))
        following=_positive_multiply(A,term,ratio)
        if ratio[1]<1:
            remainder=A.up.divide(following[1],A.down.subtract(Decimal(1),ratio[1]))
            if remainder<=A.down.multiply(total[0],target):
                series=(total[0],A.up.add(total[1],remainder))
                power=_exact(A,a**(2*n+1))
                exponential=A.exp(_exact(A,-a2))
                return _positive_multiply(A,_positive_multiply(A,power,exponential),series),j+1
        total=A.add(total,following);term=following
    raise ArithmeticError('positive moment series did not resolve its tail')


def gaussian_gram_enclosure(D_string,p,precision=80,m=None,V_string='2'):
    """Build rigorous parity Gram intervals and a PSD feature-tail bound.

    This returns Decimal interval objects for use by the trial preparer and
    by the standard-library replay. Public saved audit records are emitted
    by gaussian_rank_certificate.
    """
    if not isinstance(D_string,str) or not isinstance(V_string,str):
        raise TypeError('geometry must be exact decimal strings')
    Dd,Vd=Decimal(D_string),Decimal(V_string)
    if not Dd.is_finite() or not Vd.is_finite() or not Dd>0 or not Vd>0:
        raise ValueError('positive finite geometry required')
    if not isinstance(p,int) or isinstance(p,bool) or p<4:
        raise ValueError('at least four Gaussian features required')
    D,V=Fraction(Dd),Fraction(Vd);a=D/2
    if p<a*a:raise ValueError('the monotone trace tail requires p>=a^2')
    half,measure=_half_intervals(D,V,m)
    A=_Intervals(precision)
    pi=A.fraction(*_pi_fraction_bounds(precision))
    J0=A.divide(A.point(1),A.sqrt(A.multiply(A.point(2),pi)))
    endpoints=sorted({x for interval in half for x in interval})
    cache={};terms=0;maximum_terms=0
    for endpoint in endpoints:
        for n in range(p):
            moment,count=_positive_moment(A,n,endpoint)
            cache[endpoint,n]=moment;terms+=count;maximum_terms=max(maximum_terms,count)
    moments=[]
    for n in range(p):
        value=A.point(0)
        for lower,upper in half:
            difference=A.subtract(cache[upper,n],cache[lower,n])
            # The integral is nonnegative, independently of subtraction width.
            difference=(max(Decimal(0),difference[0]),difference[1])
            value=A.add(value,difference)
        value=_positive_multiply(A,A.point(2),value)
        if value[0]<=0:raise ArithmeticError('positive Gram moment not resolved; increase precision')
        moments.append(value)
    roots=[A.sqrt(A.point(factorial(k))) for k in range(p)]
    blocks=[]
    for parity in (0,1):
        indices=list(range(parity,p,2));matrix=[]
        for i in indices:
            row=[]
            for j in indices:
                numerator=_positive_multiply(A,J0,moments[(i+j)//2])
                denominator=_positive_multiply(A,roots[i],roots[j])
                row.append(A.divide(numerator,denominator))
            matrix.append(row)
        assert all(matrix[i][j]==matrix[j][i] for i in range(len(matrix)) for j in range(len(matrix)))
        blocks.append(matrix)
    ratio=1-a*a/Fraction(p+1)
    if ratio<=0:raise ArithmeticError('positive geometric trace denominator required')
    numerator=_positive_multiply(A,_exact(A,measure),J0)
    numerator=_positive_multiply(A,numerator,A.exp(_exact(A,-a*a)))
    numerator=_positive_multiply(A,numerator,_exact(A,a**(2*p)))
    denominator=_exact(A,Fraction(factorial(p))*ratio)
    tail=A.divide(numerator,denominator)[1]
    return dict(D=D_string,V=V_string,m=m,p=p,precision=precision,measure=measure,
        half=half,blocks=blocks,moments=moments,tail=tail,J0=J0,
        moment_series_terms=terms,maximum_moment_terms=maximum_terms)


def _collatz(A,matrix,trial):
    q=[Decimal(str(x)) for x in trial]
    if len(q)!=len(matrix) or any(not x.is_finite() or x<=0 for x in q):
        raise ValueError('one strictly positive finite decimal per parity feature is required')
    ratios=[]
    for row,qi in zip(matrix,q):
        lower=upper=Decimal(0)
        for value,qj in zip(row,q):
            lower=A.down.add(lower,A.down.multiply(value[0],qj))
            upper=A.up.add(upper,A.up.multiply(value[1],qj))
        ratios.append((A.down.divide(lower,qi),A.up.divide(upper,qi)))
    return (min(r[0] for r in ratios),max(r[1] for r in ratios)),ratios


def _inertia(A,matrix,shift):
    """Validated unpivoted LDL of G-shift*I; all pivots must avoid zero."""
    s=Decimal(str(shift))
    if not s.is_finite() or s<=0:raise ValueError('a positive rational ordering shift is required')
    n=len(matrix);lower=[[A.point(0) for _ in range(n)] for _ in range(n)];diagonal=[]
    signs=[]
    for k in range(n):
        pivot=A.subtract(matrix[k][k],A.point(s))
        for j in range(k):
            term=A.multiply(_positive_square(A,lower[k][j]),diagonal[j])
            pivot=A.subtract(pivot,term)
        if pivot[0]<=0<=pivot[1]:
            raise ArithmeticError('LDL pivot contains zero; change ordering shift or increase precision')
        signs.append(1 if pivot[0]>0 else -1);diagonal.append(pivot)
        for i in range(k+1,n):
            value=matrix[i][k]
            for j in range(k):
                value=A.subtract(value,A.multiply(A.multiply(lower[i][j],lower[k][j]),diagonal[j]))
            lower[i][k]=A.divide(value,pivot)
    return sum(sign>0 for sign in signs),diagonal


def gaussian_rank_certificate(D_string,p,trial,precision=80,m=None,V_string='2'):
    """Return (flat summary, full audit) for the continuous first two modes.

    trial has positive even_q/odd_q decimal strings and an ordering_shift.
    Only verified interval LDL signs, CW inequalities and the analytic trace
    tail enter the conclusion. NumPy is neither imported nor required here.
    """
    gram=gaussian_gram_enclosure(D_string,p,precision,m,V_string)
    A=_Intervals(precision);even,odd=gram['blocks']
    even_cw,even_ratios=_collatz(A,even,trial['even_q'])
    odd_cw,odd_ratios=_collatz(A,odd,trial['odd_q'])
    positive,pivots=_inertia(A,even,trial['ordering_shift'])
    if positive!=1:raise ArithmeticError('the even block must have exactly one eigenvalue above the shift')
    tail=gram['tail'];shift=Decimal(str(trial['ordering_shift']))
    even_rest=A.up.add(shift,tail)
    odd_upper=A.up.add(odd_cw[1],tail)
    principal_upper=A.up.add(even_cw[1],tail)
    if not even_rest<odd_cw[0] or not odd_upper<even_cw[0]:
        raise ArithmeticError('continuous parity ordering not separated after the PSD tail')
    gap_lower=A.down.subtract(even_cw[0],odd_upper)
    gap_upper=A.up.subtract(principal_upper,odd_cw[0])
    beta_lower=A.down.subtract(Decimal(1),principal_upper)
    beta_upper=A.up.subtract(Decimal(1),even_cw[0])
    beta_width=A.up.subtract(beta_upper,beta_lower)
    volume=Fraction(gram['measure']);theta=Fraction(V_string)/Fraction(D_string)
    scaled_gap=A.multiply(_exact(A,theta),(gap_lower,gap_upper))
    qhash=sha256(('\n'.join(trial['even_q'])+'\n--odd--\n'+'\n'.join(trial['odd_q'])).encode('ascii')).hexdigest()
    summary=dict(D=D_string,V=V_string,geometry='envelope' if m is None else 'fragmented',
        m='' if m is None else m,features=p,even_features=len(even),odd_features=len(odd),precision=precision,
        measure_numerator=volume.numerator,measure_denominator=volume.denominator,
        feature_tail_upper=str(tail),rho_lower=str(even_cw[0]),rho_upper=str(principal_upper),
        beta_lower=str(beta_lower),beta_upper=str(beta_upper),beta_width_upper=str(beta_width),
        second_lower=str(odd_cw[0]),second_upper=str(odd_upper),
        gap_lower=str(gap_lower),gap_upper=str(gap_upper),
        gap_width_upper=str(A.up.subtract(gap_upper,gap_lower)),
        scaled_gap_lower=str(scaled_gap[0]),scaled_gap_upper=str(scaled_gap[1]),
        finite_even_cw_lower=str(even_cw[0]),finite_even_cw_upper=str(even_cw[1]),
        finite_odd_cw_lower=str(odd_cw[0]),finite_odd_cw_upper=str(odd_cw[1]),
        ordering_shift=str(shift),even_rest_upper=str(even_rest),even_inertia_positive=positive,
        moment_series_terms=gram['moment_series_terms'],maximum_moment_terms=gram['maximum_moment_terms'],
        q_sha256=qhash,full_second_is_odd=True,
        method='PSD Gaussian feature tail; positive-series Gram moments; parity CW; interval LDL inertia')
    serial=lambda matrix:[[[str(x),str(y)] for x,y in row] for row in matrix]
    audit=dict(summary=summary,trial=trial,
        positive_half_intervals=[[str(x),str(y)] for x,y in gram['half']],
        even_gram=serial(even),odd_gram=serial(odd),
        even_cw_ratios=[[str(x),str(y)] for x,y in even_ratios],
        odd_cw_ratios=[[str(x),str(y)] for x,y in odd_ratios],
        even_ldl_pivots=[[str(x),str(y)] for x,y in pivots],
        even_moments=[[str(x),str(y)] for x,y in gram['moments']])
    return summary,audit


def gaussian_rank_self_check(D_string,p,trial,precision=80,m=None,V_string='2'):
    first,_=gaussian_rank_certificate(D_string,p,trial,precision,m,V_string)
    finer,_=gaussian_rank_certificate(D_string,p,trial,precision+20,m,V_string)
    for lower,upper in (('rho_lower','rho_upper'),('second_lower','second_upper'),('gap_lower','gap_upper')):
        assert Decimal(first[lower])<=Decimal(finer[lower])<=Decimal(finer[upper])<=Decimal(first[upper])
    assert Decimal(finer['feature_tail_upper'])<=Decimal(first['feature_tail_upper'])
    with localcontext() as alien:
        alien.prec=2;alien.rounding=ROUND_CEILING
        assert gaussian_rank_certificate(D_string,p,trial,precision,m,V_string)[0]==first
    return dict(D=D_string,features=p,precision_nesting=True,alien_context_invariance=True,
                full_parity_ordering_with_tail=True)
