"""Principal-only continuous Gaussian certificates on symmetric fragments.

The full Gaussian operator has a positive simple even principal eigenfunction.
It is therefore sufficient to enclose the Perron value of the EVEN feature
Gram block, then include the omitted positive operator. No odd-mode ordering,
inertia computation, quantified eigengap or floating eigenvalue is used.

For excluding a competitor in an upper-margin test, only rho_lower (equivalently
beta_upper) is required. That endpoint has no feature-tail subtraction: PSD
truncation is a one-sided lower approximation to the continuous operator.
"""
from decimal import Decimal,localcontext,ROUND_CEILING
from fractions import Fraction
from hashlib import sha256
from math import factorial,lcm

from validated_gaussian import _Intervals,_pi_fraction_bounds
from validated_gaussian_rank import _half_intervals,_collatz


def _aggregate_gram_enclosure(D_string,p,precision=60,m=None,V_string='2'):
    """Fast exact-power moment inclusion for many small rational patches.

    For t>=0 the odd/even adjacent Maclaurin polynomials of exp(-t) are
    lower/upper bounds, by the signed integral Taylor remainder. Integrate
    both polynomials over the COMPLETE fragmented domain in exact Fraction
    arithmetic, before converting their endpoints to Decimal. Cancellation
    is consequently exact, rather than a floating numerical approximation.
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
    A=_Intervals(precision);exact=lambda x:A.fraction(x,x)
    a2=a*a;exponential=A.exp(exact(-a2))
    target=Fraction(1,10**(precision-12))
    term=Fraction(1);k=None
    for index in range(1,10001):
        term*=a2/index
        if index%2==0 and term<=Fraction(exponential[0])*target:
            k=index;break
    if k is None:raise ArithmeticError('exact polynomial moment enclosure did not resolve its tail')
    # k is even. Degree k-1 is a global lower polynomial, degree k upper.
    maximum=p-1+k
    denominator=lcm(*(x.denominator for ab in half for x in ab))
    powers=[0]*(maximum+1)
    for left,right in half:
        li=left.numerator*(denominator//left.denominator)
        ri=right.numerator*(denominator//right.denominator)
        lp,rp=li,ri;ls,rs=li*li,ri*ri
        for degree in range(maximum+1):
            powers[degree]+=rp-lp
            lp*=ls;rp*=rs
    raw=[];scale=denominator;square=denominator*denominator
    for degree,value in enumerate(powers):
        raw.append(Fraction(2*value,(2*degree+1)*scale))
        scale*=square
    assert raw[0]==measure
    inverse_factorials=[Fraction(1,factorial(j)) for j in range(k+1)]
    moments=[]
    for n in range(p):
        lower=sum(((-1)**j*raw[n+j]*inverse_factorials[j] for j in range(k)),Fraction(0))
        upper=lower+raw[n+k]*inverse_factorials[k]
        assert 0<lower<=upper
        # This independent exact inequality documents the global Taylor-tail
        # target; it is not a stopping tolerance substituted for inclusion.
        assert upper-lower<=raw[n]*term
        moments.append(A.fraction(lower,upper))
    pi=A.fraction(*_pi_fraction_bounds(precision))
    J0=A.divide(A.point(1),A.sqrt(A.multiply(A.point(2),pi)))
    roots=[A.sqrt(A.point(factorial(index))) for index in range(p)]
    indices=list(range(0,p,2));matrix=[]
    for i in indices:
        row=[]
        for j in indices:
            numerator=A.multiply(J0,moments[(i+j)//2])
            row.append(A.divide(numerator,A.multiply(roots[i],roots[j])))
        matrix.append(row)
    assert all(matrix[i][j]==matrix[j][i] for i in range(len(matrix)) for j in range(len(matrix)))
    ratio=1-a2/Fraction(p+1)
    numerator=A.multiply(A.multiply(exact(measure),J0),exponential)
    numerator=A.multiply(numerator,exact(a**(2*p)))
    tail=A.divide(numerator,exact(Fraction(factorial(p))*ratio))[1]
    return dict(D=D_string,V=V_string,m=m,p=p,precision=precision,measure=measure,
        half=half,blocks=[matrix],moments=moments,tail=tail,
        moment_series_terms=p*(k+1),maximum_moment_terms=k+1,
        moment_method='exact rational aggregate powers; signed Taylor inclusion',
        taylor_upper_degree=k)


def _certificate_from_gram(gram,q):
    """Private shared calculation; gram must come from the validated builder.

    The public replay always reconstructs its Gram enclosure from exact
    geometry. The trial-preparation helper calls this private operation on
    the SAME rigorous builder result, avoiding duplicate moment work.
    """
    precision=gram['precision'];A=_Intervals(precision)
    even=gram['blocks'][0]
    canonical=[str(Decimal(str(value))) for value in q]
    cw,ratios=_collatz(A,even,canonical)
    tail=gram['tail']
    rhoL=cw[0]
    # Whole-line normalization also gives ||K||<=1 on every finite habitat.
    rhoU=min(Decimal(1),A.up.add(cw[1],tail))
    if not Decimal(0)<rhoL<=rhoU<=Decimal(1):
        raise ArithmeticError('the continuous principal enclosure is inconsistent')
    betaL=A.down.subtract(Decimal(1),rhoU)
    betaU=A.up.subtract(Decimal(1),rhoL)
    qhash=sha256('\n'.join(canonical).encode('ascii')).hexdigest()
    summary=dict(D=gram['D'],V=gram['V'],m=gram['m'],features=gram['p'],
        even_features=len(even),precision=precision,
        rho_lower=str(rhoL),rho_upper=str(rhoU),
        beta_lower=str(betaL),beta_upper=str(betaU),
        beta_width_upper=str(A.up.subtract(betaU,betaL)),
        finite_even_cw_lower=str(cw[0]),finite_even_cw_upper=str(cw[1]),
        finite_even_cw_width_upper=str(A.up.subtract(cw[1],cw[0])),
        feature_tail_upper=str(tail),
        moment_series_terms=gram['moment_series_terms'],
        maximum_moment_terms=gram['maximum_moment_terms'],
        moment_method=gram.get('moment_method','positive endpoint series'),
        taylor_upper_degree=gram.get('taylor_upper_degree',''),
        q_sha256=qhash,principal_is_even=True,uses_quantified_gap=False,
        uses_odd_ordering=False,competitor_upper_uses_tail=False,
        method='principal even Gram CW; PSD feature ordering; continuous trace-tail upper')
    audit=dict(summary=summary,trial=canonical,
        positive_half_intervals=[[str(a),str(b)] for a,b in gram['half']],
        even_moments=[[str(a),str(b)] for a,b in gram['moments']],
        even_cw_ratios=[[str(a),str(b)] for a,b in ratios])
    return summary,audit


def fragment_principal_certificate(D_string,m,p,q,precision=60,V_string='2'):
    """Return (flat summary, audit) for the exact rational equal-patch family.

    q contains one arbitrary finite positive decimal per EVEN feature. It need
    not be an eigenvector. Its exact decimal values are used in directed CW
    sums. Geometry, Gaussian moments, normalization, factorial roots, feature
    tail and every CW endpoint are rebuilt by the standard-library replay.
    """
    if not isinstance(m,int) or isinstance(m,bool) or m<2:
        raise ValueError('a fragmented family requires integer m>=2')
    gram=_aggregate_gram_enclosure(D_string,p,precision,m,V_string)
    return _certificate_from_gram(gram,q)


def fragment_principal_self_check(D_string,m,p,q,precision=60,V_string='2'):
    """Check nested continuum endpoints and ambient-context independence."""
    first,_=fragment_principal_certificate(D_string,m,p,q,precision,V_string)
    finer,_=fragment_principal_certificate(D_string,m,p,q,precision+20,V_string)
    for lower,upper in [('rho_lower','rho_upper'),('beta_lower','beta_upper')]:
        assert Decimal(first[lower])<=Decimal(finer[lower])<=Decimal(finer[upper])<=Decimal(first[upper])
    assert Decimal(finer['feature_tail_upper'])<=Decimal(first['feature_tail_upper'])
    with localcontext() as alien:
        alien.prec=2;alien.rounding=ROUND_CEILING
        assert fragment_principal_certificate(D_string,m,p,q,precision,V_string)[0]==first
    return dict(D=D_string,V=V_string,m=m,features=p,
        precision_nesting=True,ambient_context_invariance=True,
        principal_even_only=True,uses_quantified_gap=False)


def principal_margin_certificate(record,water,dv_string='.6',B_string='.45',
                                 Amax_string='1.8',R_string='4'):
    """Enclose dv*beta+B-Amax*R*q using a separately certified water cap.

    This is an interval for a sufficient exclusion-test margin, not for a
    nonlinear stationary state or for a persistence threshold.
    """
    A=_Intervals(int(record['precision']))
    fractions=[Fraction(value) for value in (dv_string,B_string,Amax_string,R_string)]
    dv,B,Amax,R=fractions
    if dv<=0 or B<=0 or Amax<0 or R<0:
        raise ValueError('requires dv,B>0 and Amax,R>=0')
    exact=lambda value:A.fraction(value,value)
    beta=(Decimal(record['beta_lower']),Decimal(record['beta_upper']))
    q=(Decimal(water['q_lower']),Decimal(water['q_upper']))
    margin=A.subtract(A.add(A.multiply(exact(dv),beta),exact(B)),A.multiply(exact(Amax*R),q))
    # Independent exact Fraction checks on the combined saved endpoints.
    lower=dv*Fraction(beta[0])+B-Amax*R*Fraction(q[1])
    upper=dv*Fraction(beta[1])+B-Amax*R*Fraction(q[0])
    assert Fraction(margin[0])<=lower<=upper<=Fraction(margin[1])
    return dict(dv=dv_string,B=B_string,Amax=Amax_string,R=R_string,
        water_q_lower=water['q_lower'],water_q_upper=water['q_upper'],
        margin_lower=str(margin[0]),margin_upper=str(margin[1]),
        margin_width_upper=str(A.up.subtract(margin[1],margin[0])),
        exact_fraction_margin_check=True,
        margin_decision='excluded' if margin[0]>0 else 'inconclusive')
