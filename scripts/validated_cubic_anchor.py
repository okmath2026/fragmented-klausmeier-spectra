"""Anchor-dependent cubic norm budgets with exact rational propagation.

The source positive expressions are already enclosed by validated_second_budget.
The new budget retains h<=1/M inside e_M and f_M instead of replacing h by1/2.
This module certifies a smaller norm-majorant expression; it does not estimate
the unknown optimal cubic error constant or a signed second coefficient.
"""
from decimal import Decimal,localcontext,ROUND_CEILING
from fractions import Fraction
from validated_gaussian import _Intervals

if not __debug__:
    raise RuntimeError('Run validated certificates without -O.')


_EXPRESSIONS=('a1','a2','CX','z0','s','q','LU','K3')


def _exact(value,name):
    if isinstance(value,(bool,float)):
        raise TypeError(name+' must be exact, not float or boolean')
    try:return Fraction(value)
    except (TypeError,ValueError,OverflowError,ZeroDivisionError) as error:
        raise ValueError(name+' must be finite and exact') from error


def _inputs(record):
    D,V=_exact(record['D'],'D'),_exact(record['V'],'V')
    if not D>V>0:raise ValueError('requires D>V>0')
    n=_exact(record['N'],'N')
    oldM=_exact(record['anchor_M'],'anchor_M')
    if n.denominator!=1 or n<4 or n.numerator%2:
        raise ValueError('source N must be an even integer at least four')
    if oldM.denominator!=1 or oldM<2:raise ValueError('source anchor must be integral and at least two')
    values={}
    for name in _EXPRESSIONS:
        lower=_exact(record[name+'_lower'],name+'_lower')
        upper=_exact(record[name+'_upper'],name+'_upper')
        if not 0<lower<=upper:raise ValueError('positive ordered source endpoints required: '+name)
        values[name]=(lower,upper)
    return D,V,int(n),int(oldM),values


def _M(value):
    result=_exact(value,'M')
    if result.denominator!=1 or result<2:raise ValueError('M must be integral and at least two')
    return int(result)


def _side(values,M,side):
    a1,a2,CX,z0,s,q,LU=(values[key][side] for key in ('a1','a2','CX','z0','s','q','LU'))
    e=a1+a2/M+CX/M**2
    f=a2+CX/M
    combination=z0+q*e
    eta=s*combination/M
    return e,f,combination,eta


def _anchor_search(record):
    *_,values=_inputs(record)
    trace=[]
    def good(M,stage):
        eta=_side(values,M,1)[3]
        result=eta<=Fraction(1,2)
        trace.append(dict(stage=stage,M=M,eta_upper_exact=str(eta),half_condition=result))
        return result
    low=high=2
    while not good(high,'doubling'):high*=2
    while low<high:
        middle=(low+high)//2
        if good(middle,'bisection'):high=middle
        else:low=middle+1
    assert good(low,'final') and (low==2 or not good(low-1,'predecessor'))
    return low,trace


def minimum_cubic_anchor(record):
    """Smallest integer M>=2 satisfying the exact all-upper eta_M<=1/2."""
    return _anchor_search(record)[0]


def cubic_anchor_search_audit(record):
    M,trace=_anchor_search(record)
    return dict(D=str(record['D']),V=str(record['V']),N=int(record['N']),
        minimum_M=M,search='exact rational doubling and bisection',trace=trace)


def cubic_anchor_certificate(record,M,precision=60):
    """Enclose the exact anchor-dependent positive norm-majorant formula.

    Every *_lower/upper pair encloses a bound EXPRESSION; its lower endpoint
    is not a lower bound for the actual spectral remainder. Validity holds
    for every integer m>=M when eta_upper<1. The dataset uses eta<=1/2.
    """
    D,V,N,oldM,values=_inputs(record);M=_M(M)
    A=_Intervals(precision)
    low=_side(values,M,0);high=_side(values,M,1)
    if not 0<low[3]<=high[3]<1:
        raise ValueError('anchor does not satisfy eta_M<1')
    def budget(side,parts):
        a1,CX,s,q,LU=(values[key][side] for key in ('a1','CX','s','q','LU'))
        e,f,combination,eta=parts
        return LU*(CX+s*q*f*e/(1-eta)
            +a1*s*s*q*e*combination/(1-eta)+a1*s*q*f)
    KL,KU=budget(0,low),budget(1,high)
    assert 0<KL<=KU
    result=dict(D=str(record['D']),V=str(record['V']),N=N,precision=precision,M=M,
        source_anchor_M=oldM,source_K3_lower=str(record['K3_lower']),
        source_K3_upper=str(record['K3_upper']),
        valid_for_every_m_ge_M=True,signed_second_coefficient_computed=False,
        finite_ratio_localization=False,
        method='exact rational anchor-dependent Neumann budget; final directed Decimal conversion')
    for name,L,U in [('eM',low[0],high[0]),('fM',low[1],high[1]),
                     ('eta',low[3],high[3]),('K3',KL,KU)]:
        endpoints=A.fraction(L,U)
        result[name+'_lower_exact']=str(L);result[name+'_upper_exact']=str(U)
        result[name+'_lower']=str(endpoints[0]);result[name+'_upper']=str(endpoints[1])
    gainL=values['K3'][0]/KU;gainU=values['K3'][1]/KL
    result['K3_reduction_lower']=str(A.fraction(gainL,gainL)[0])
    result['K3_reduction_upper']=str(A.fraction(gainU,gainU)[1])
    anchorGain=Fraction(oldM,M)
    result['anchor_reduction_lower']=str(A.fraction(anchorGain,anchorGain)[0])
    result['anchor_reduction_upper']=str(A.fraction(anchorGain,anchorGain)[1])
    result['eta_at_m_is_decreasing']=True
    result['eta_half_condition']=high[3]<=Fraction(1,2)
    return result


def cubic_anchor_self_check(record,precision=60):
    M=minimum_cubic_anchor(record)
    coarse=cubic_anchor_certificate(record,M,precision)
    fine=cubic_anchor_certificate(record,M,precision+20)
    for name in ('eM','fM','eta','K3','K3_reduction','anchor_reduction'):
        lo,up=name+'_lower',name+'_upper'
        assert Fraction(coarse[lo])<=Fraction(fine[lo])<=Fraction(fine[up])<=Fraction(coarse[up])
    with localcontext() as foreign:
        foreign.prec=2;foreign.rounding=ROUND_CEILING
        assert cubic_anchor_certificate(record,M,precision)==coarse
        assert minimum_cubic_anchor(record)==M
    if M>2:
        *_,values=_inputs(record)
        assert _side(values,M-1,1)[3]>Fraction(1,2)
    successive=[cubic_anchor_certificate(record,M*factor,precision) for factor in (1,2,4,8)]
    for first,second in zip(successive,successive[1:]):
        assert Fraction(second['K3_upper_exact'])<Fraction(first['K3_upper_exact'])
        assert Fraction(second['eta_upper_exact'])<Fraction(first['eta_upper_exact'])
    for invalid in (1,0,-1,True,3.5,'1.5'):
        try:cubic_anchor_certificate(record,invalid,precision)
        except (TypeError,ValueError):pass
        else:raise AssertionError('invalid M accepted')
    for field,bad in [('a1_lower','0'),('q_upper','-1'),('V',record['D'])]:
        changed=dict(record);changed[field]=bad
        try:minimum_cubic_anchor(changed)
        except (TypeError,ValueError):pass
        else:raise AssertionError('invalid source accepted')
    return dict(D=str(record['D']),V=str(record['V']),N=int(record['N']),precision=precision,
        minimum_M=M,minimum_half_anchor_exact=True,precision_nesting=True,
        ambient_context_invariance=True,monotone_anchor_refinement=True,
        invalid_inputs_rejected=True,signed_second_coefficient_computed=False,
        finite_ratio_localization=False)
