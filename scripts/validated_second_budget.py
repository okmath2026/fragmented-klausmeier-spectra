"""Directed budgets for a genuine second fragmentation coefficient.

The input records are independently certified bounds for the unscaled
Gaussian envelope Perron value rho and the FULL theta-scaled gap gamma.
This module propagates them through the proved C^3 perturbation bounds.
It does not compute the signed second coefficient or localise a finite-r
optimizer: its coarse |d_D| and cubic-remainder budgets are upper bounds.
Only Python's standard library and owned Decimal contexts are used.
"""
from decimal import Decimal,localcontext,ROUND_CEILING
from fractions import Fraction
from math import comb,factorial

from validated_gaussian import _Intervals,_pi_fraction_bounds

if not __debug__:
    raise RuntimeError('Run validated certificates without -O.')


def _rational(value,name):
    if isinstance(value,(bool,float)):
        raise TypeError(name+' must be exact, not a float or boolean')
    try:
        return Fraction(value)
    except (TypeError,ValueError,ZeroDivisionError,OverflowError) as error:
        raise ValueError(name+' must be finite and exact') from error


def _input_records(envelope,gap):
    D,V=(_rational(envelope[k],k) for k in ('D','V'))
    if not D>V>0:
        raise ValueError('requires D>V>0')
    for key,value in (('D',D),('V',V)):
        if _rational(gap[key],key)!=value:
            raise ValueError('envelope and gap geometries do not match')
    n=_rational(envelope['N'],'N')
    if n.denominator!=1 or n<4 or n.numerator%2:
        raise ValueError('input N must be an even integer at least four')
    if _rational(gap['N'],'N')!=n:
        raise ValueError('input envelope and gap resolutions do not match')
    rL,rU=(_rational(envelope[k],k) for k in ('rho_lower','rho_upper'))
    gL,gU=(_rational(gap[k],k) for k in ('gamma_lower','gamma_upper'))
    if not 0<rL<=rU<=1:
        raise ValueError('requires 0<rho_lower<=rho_upper<=1')
    if not 0<gL<=gU or gL>(V/D)*rU:
        raise ValueError('requires a positive consistent theta-scaled Gaussian gap')
    return D,V,int(n),rL,rU,gL,gU


def _ceil_fraction(value):
    return -((-value.numerator)//value.denominator)


def second_budget_certificate(envelope_record,gap_record,precision=60):
    """Enclose every positive expression in the cubic perturbation budget.

    All reported *_lower/upper pairs enclose the corresponding displayed
    BOUND EXPRESSION, rather than asserting lower bounds on optimal error
    constants. In particular d_abs_upper bounds the true |d_D|; d_abs_lower
    is only an endpoint for its conservative norm-majorant expression.
    Input source bounds remain fixed when arithmetic precision is increased.
    """
    D,V,n,rL,rU,gL,gU=_input_records(envelope_record,gap_record)
    A=_Intervals(precision)
    exact=lambda x:A.fraction(Fraction(x),Fraction(x))
    add=A.add;mul=A.multiply;div=A.divide
    def total(values):
        answer=exact(0)
        for value in values:answer=add(answer,value)
        return answer
    def maximum(values):return max(x[0] for x in values),max(x[1] for x in values)
    theta=V/D;delta=D-V;alpha=1-theta
    kappa=V*D*alpha*(1-2*theta)/12
    C=[4*theta**3*alpha*D,
       Fraction(2,3)*theta*alpha*V**2+V*delta/3,
       V**3*(theta/12+Fraction(1,6))+V**2*D/6,
       Fraction(2,3)*V*D**3+V**3*D/48+V**4/192]
    pi=A.fraction(*_pi_fraction_bounds(precision))
    J0=div(exact(1),A.sqrt(mul(exact(2),pi)))
    # Fourier Gaussian moments: even (k-1)!!/sqrt(2pi), odd 2^j*j!/pi.
    B=[]
    for k in range(7):
        if k%2:
            j=(k-1)//2
            B.append(div(exact(2**j*factorial(j)),pi))
        else:
            double=1
            for value in range(1,k,2):double*=value
            B.append(mul(exact(double),J0))
    Bstar=maximum(B[:4]);rho=A.fraction(rL,rU);gamma=A.fraction(gL,gU)
    L=A.sqrt(exact(D))
    U=div(mul(L,Bstar),rho)
    q=add(exact(1),mul(L,U))
    s=div(add(exact(1),div(mul(exact(theta*D),Bstar),gamma)),mul(exact(theta),rho))
    a1=mul(exact(2*theta*delta),Bstar)
    a2=add(mul(exact(theta),a1),
           mul(exact(2*abs(kappa)),maximum([add(B[j],B[j+1]) for j in range(4)])))
    CX=maximum([total([mul(exact(C[j]),
                    total([mul(exact(comb(j,i)),B[k+i]) for i in range(j+1)]))
                    for j in range(4)]) for k in range(4)])
    e0=total([a1,mul(exact(Fraction(1,2)),a2),mul(exact(Fraction(1,4)),CX)])
    f0=add(a2,mul(exact(Fraction(1,2)),CX))
    z0=mul(exact(2*V*delta),B[1]);LU=mul(L,U)
    combination=add(z0,mul(q,e0))
    term1=mul(exact(2),mul(s,mul(q,mul(f0,e0))))
    term2=mul(exact(2),mul(a1,mul(mul(s,s),mul(q,mul(e0,combination)))))
    term3=mul(a1,mul(s,mul(q,f0)))
    K3=mul(LU,total([CX,term1,term2,term3]))
    d_abs=mul(LU,add(a2,mul(a1,mul(s,mul(q,a1)))))
    threshold=mul(exact(2),mul(s,combination))
    anchor=max(2,_ceil_fraction(Fraction(threshold[1])))
    # Verify the actual all-upper sufficient inequality by exact fractions.
    assert Fraction(anchor)>=2*Fraction(s[1])*(Fraction(z0[1])+Fraction(q[1])*Fraction(e0[1]))
    result=dict(D=str(envelope_record['D']),V=str(envelope_record['V']),N=n,
        precision=precision,rho_lower=str(envelope_record['rho_lower']),
        rho_upper=str(envelope_record['rho_upper']),
        gamma_lower=str(gap_record['gamma_lower']),gamma_upper=str(gap_record['gamma_upper']),
        gamma_is_theta_scaled=True,theta_exact=str(theta),delta_exact=str(delta),
        kappa_exact=str(kappa),anchor_M=anchor,
        signed_second_coefficient_computed=False,finite_ratio_localization=False,
        method='Fourier Gaussian derivative bounds; exact weak-measure constants; directed C3 resolvent and Schur cubic budget')
    for name,value in [('C'+str(j),exact(c)) for j,c in enumerate(C)]:
        result[name+'_exact']=str(C[int(name[1:])])
        result[name+'_lower']=str(value[0]);result[name+'_upper']=str(value[1])
    expressions=[('B'+str(j),value) for j,value in enumerate(B)]+[
        ('Bstar',Bstar),('L',L),('U',U),('q',q),('s',s),('a1',a1),('a2',a2),
        ('CX',CX),('e0',e0),('f0',f0),('z0',z0),('LU',LU),('K3',K3),
        ('d_abs',d_abs),('resolution_threshold',threshold)]
    for name,(lower,upper) in expressions:
        assert 0<=lower<=upper and lower.is_finite() and upper.is_finite(),name
        result[name+'_lower']=str(lower);result[name+'_upper']=str(upper)
    return result


def second_budget_self_check(envelope_record,gap_record,precision=60):
    """Test arithmetic nesting, alien context, exact anchor and bad inputs."""
    first=second_budget_certificate(envelope_record,gap_record,precision)
    finer=second_budget_certificate(envelope_record,gap_record,precision+20)
    for key in first:
        if key.endswith('_lower') and key[:-6]+'_upper' in first:
            upper=key[:-6]+'_upper'
            assert Fraction(first[key])<=Fraction(finer[key])<=Fraction(finer[upper])<=Fraction(first[upper]),key
    assert finer['anchor_M']<=first['anchor_M']
    with localcontext() as foreign:
        foreign.prec=2;foreign.rounding=ROUND_CEILING
        alien=second_budget_certificate(envelope_record,gap_record,precision)
    assert first==alien
    for side,key,value in [('envelope','D','0'),('envelope','V',first['D']),
                           ('envelope','rho_lower','0'),('gap','gamma_lower','0'),
                           ('gap','D','99'),('gap','N','3')]:
        e=dict(envelope_record);g=dict(gap_record)
        (e if side=='envelope' else g)[key]=value
        try:second_budget_certificate(e,g,precision)
        except (TypeError,ValueError):pass
        else:raise AssertionError('invalid input accepted: '+side+'/'+key)
    for bad in (0,20,60.0,True):
        try:second_budget_certificate(envelope_record,gap_record,bad)
        except (TypeError,ValueError):pass
        else:raise AssertionError('invalid precision accepted')
    return dict(D=first['D'],V=first['V'],N=first['N'],precision=precision,
        precision_nesting=True,ambient_context_invariance=True,
        exact_anchor_inequality=True,invalid_inputs_rejected=True,
        signed_second_coefficient_computed=False,finite_ratio_localization=False)
