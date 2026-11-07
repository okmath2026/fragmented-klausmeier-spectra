"""Exact rational checks of the second weak habitat-measure expansion.

For fixed-endpoint equal intervals the analytic C^3 remainder is
  |mu_m(f)-mu0(f)-mu1(f)/m-mu2(f)/m^2|
       <= m^-3 * sum(C_j * ||f^(j)||_infinity, j=0..3).
This module checks that statement on monomials using exact Fraction
arithmetic. It does not certify the second *spectral* coefficient or its
operator remainder. No floating eigenvalue, fitted coefficient, quadrature
or Decimal transcendental primitive enters these records.
"""
from fractions import Fraction
from math import comb,factorial


if not __debug__:
    raise RuntimeError('Run validated certificates without -O.')


def _geometry(D_string,V_string,m,n):
    if not isinstance(D_string,str) or not isinstance(V_string,str):
        raise TypeError('D and V must be exact decimal or rational strings')
    try:
        D,V=Fraction(D_string),Fraction(V_string)
    except (ValueError,ZeroDivisionError) as error:
        raise ValueError('D and V must be finite exact rational strings') from error
    if not D>V>0:
        raise ValueError('requires D>V>0')
    if not isinstance(m,int) or isinstance(m,bool) or m<2:
        raise ValueError('requires an integer patch count m>=2')
    if not isinstance(n,int) or isinstance(n,bool) or n<0:
        raise ValueError('requires an integer monomial degree n>=0')
    return D,V


def _endpoint_moment(D,V,m,n):
    """Full-interval endpoint integration, with first endpoint exactly -D/2."""
    width=V/m
    step=(D-width)/(m-1)
    result=Fraction(0)
    for i in range(m):
        left=-D/2+i*step
        right=left+width
        result+=(right**(n+1)-left**(n+1))/(n+1)
    return result


def _centred_binomial_moment(D,V,m,n):
    """Independent integration after expanding x=(centre+local coordinate).

    Only even local powers survive on a symmetric patch. This is separate
    from powers of the full-domain endpoints in _endpoint_moment.
    """
    width=V/m
    separation=width+(D-V)/(m-1)
    result=Fraction(0)
    for i in range(m):
        centre=(Fraction(i)-Fraction(m-1,2))*separation
        for j in range(0,n+1,2):
            result+=Fraction(comb(n,j),2**j*(j+1))*centre**(n-j)*width**(j+1)
    return result


def measure_moment_certificate(D_string,V_string,m,n):
    """Return a flat exact record for f(x)=x^n on the specified habitat.

    D_string and V_string are exact decimal/rational strings. The interval
    endpoints, coefficients, derivative suprema, remainder and its analytic
    bound are all Fractions. Finite monomial checks support the analytic
    weak-expansion proof; they do not replace its proof for general C^3 f.
    """
    D,V=_geometry(D_string,V_string,m,n)
    theta=V/D;delta=D-V;a=-D/2;b=D/2
    integral=(b**(n+1)-a**(n+1))/(n+1)
    fa,fb=a**n,b**n
    dfa=Fraction(0) if n==0 else n*a**(n-1)
    dfb=Fraction(0) if n==0 else n*b**(n-1)
    mu0=theta*integral
    mu1=theta*delta*((fa+fb)/2-integral/D)
    kappa=V*D*(1-theta)*(1-2*theta)/12
    mu2=theta*mu1+kappa*(dfb-dfa)
    mu_m=_endpoint_moment(D,V,m,n)
    independent=_centred_binomial_moment(D,V,m,n)
    assert mu_m==independent
    assert (n%2==0 or mu_m==mu0==mu1==mu2==0)
    if n==0:assert mu_m==mu0==V and mu1==mu2==0
    constants=[
        4*theta**3*(1-theta)*D,
        Fraction(2,3)*theta*(1-theta)*V**2+V*delta/3,
        V**3*(theta/12+Fraction(1,6))+V**2*D/6,
        Fraction(2,3)*V*D**3+V**3*D/48+V**4/192,
    ]
    norms=[Fraction(0) if j>n else Fraction(factorial(n),factorial(n-j))*b**(n-j)
           for j in range(4)]
    assert all(value>=0 for value in constants+norms)
    approximation=mu0+mu1/m+mu2/m**2
    residual=mu_m-approximation
    scaled_bound=sum((C*norm for C,norm in zip(constants,norms)),Fraction(0))
    bound=scaled_bound/m**3
    assert abs(residual)<=bound
    row=dict(D=D_string,V=V_string,m=m,degree=n,D_exact=str(D),V_exact=str(V),
        theta_exact=str(theta),kappa_exact=str(kappa),mu_m_exact=str(mu_m),
        mu0_exact=str(mu0),mu1_exact=str(mu1),mu2_exact=str(mu2),
        approximation_exact=str(approximation),residual_exact=str(residual),
        absolute_residual_exact=str(abs(residual)),remainder_bound_exact=str(bound),
        scaled_absolute_residual_exact=str(abs(residual)*m**3),
        scaled_bound_exact=str(scaled_bound),
        scaled_second_difference_exact=str(m**2*(mu_m-mu0-mu1/m)),
        bound_verified=True,independent_binomial_match=True,
        arithmetic='exact rational',
        method='full endpoint powers; independent centred binomial moments; C3 weak-measure remainder')
    row.update({'C'+str(j)+'_exact':str(value) for j,value in enumerate(constants)})
    row.update({'norm'+str(j)+'_exact':str(value) for j,value in enumerate(norms)})
    return row


def fragment_measure_self_check():
    """Odd/even, fractional geometry, ambient context and exact water algebra."""
    from decimal import localcontext,ROUND_CEILING
    geometries=[('4','2'),('6','2'),('3.7','2.1'),('5','1')]
    cases=0
    for D,V in geometries:
        for m in (2,3,7,16):
            for degree in range(10):
                measure_moment_certificate(D,V,m,degree);cases+=1
    baseline=measure_moment_certificate('3.7','2.1',7,8)
    with localcontext() as context:
        context.prec=2;context.rounding=ROUND_CEILING
        assert measure_moment_certificate('3.7','2.1',7,8)==baseline
    water_checks=0
    for b in (Fraction(1,2),Fraction(1,10),Fraction(2),Fraction(7,3)):
        for m in (2,3,7,32):
            difference=b/m**2-b/(m**2+b)
            exact=b*b/(m**2*(m**2+b))
            assert difference==exact and 0<exact<=b*b/m**4
            water_checks+=1
    for invalid in [('4','4',2,0),('2','4',2,0),('4','2',1,0),
                    ('4','2',True,0),('4','2',2,-1),('4','2',2,False)]:
        try:measure_moment_certificate(*invalid)
        except ValueError:pass
        else:raise AssertionError(('invalid input accepted',invalid))
    try:measure_moment_certificate(4.0,'2',2,0)
    except TypeError:pass
    else:raise AssertionError('inexact geometry input accepted')
    return dict(exact_additional_cases=cases,odd_and_even_counts=True,
        odd_and_even_monomials=True,fractional_geometry=True,
        independent_endpoint_and_binomial_integration=True,
        ambient_context_invariance=True,water_rational_identity_cases=water_checks,
        invalid_inputs_rejected=True,
        spectral_coefficient_certified=False,
        scope='exact polynomial tests of the proved weak C3 measure expansion')
