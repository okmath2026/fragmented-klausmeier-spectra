"""Standard-library interval certificates for Gaussian boundary corrections.

Every floating-point input is an exact decimal trial value, not an asserted
eigenvector.  Decimal/Fraction interval primitives are shared with the verified
midpoint module.  Production code imports neither NumPy nor a BLAS eigensolver.
"""
from decimal import Context, Decimal, ROUND_CEILING, ROUND_FLOOR, localcontext
from fractions import Fraction
from hashlib import sha256

from validated_gaussian import _Intervals, _pi_fraction_bounds, gaussian_certificate


def _sqrt_up(arithmetic, value):
    if value < 0:
        raise ArithmeticError("negative certified square-root argument")
    return Decimal(0) if value == 0 else arithmetic.sqrt(arithmetic.point(value))[1]


def _exact(arithmetic, value):
    value=Fraction(value)
    return arithmetic.fraction(value,value)


def boundary_certificate(D_string,n,q,precision=60):
    """Enclose rho, the full spectral gap, and c_D=-C_D for V=2.

    The exact positive decimal trial vector must be even.  Both the quadratic
    observable and endpoint trace independently enclose c_D; their intersection
    is reported.  All norm, arithmetic, and cell-averaging errors are included.
    """
    if not isinstance(D_string,str):
        raise TypeError("D_string must be an exact decimal string")
    D=Decimal(D_string)
    if not D.is_finite() or D<=2:
        raise ValueError("the boundary certificate requires finite D>2 and V=2")
    if len(D.as_tuple().digits)>precision-10:
        raise ValueError("geometry has too many digits for the requested working precision")
    if not isinstance(n,int) or isinstance(n,bool) or n<4 or n%2:
        raise ValueError("n must be an even integer at least four")
    q=[Decimal(str(value)) for value in q]
    if len(q)!=n or any(not x.is_finite() or x<=0 for x in q):
        raise ValueError("q must contain n finite, strictly positive decimals")
    if q!=q[::-1]:
        raise ValueError("the exact decimal trial vector must be even")
    A=_Intervals(precision);lo,up=A.down,A.up
    point=A.point;one,two=point(1),point(2)
    DF=Fraction(D);VF=Fraction(2);deltaF=DF-VF;thetaF=VF/DF
    hF=DF/n;factorF=thetaF*deltaF/DF
    dI,hI,deltaI,thetaI,factorI=(_exact(A,x) for x in (DF,hF,deltaF,thetaF,factorF))
    pi=A.fraction(*_pi_fraction_bounds(precision));sqrtpi=A.sqrt(pi)
    J0=A.divide(one,A.sqrt(A.multiply(two,pi)))
    exp_half=A.exp(point("-.5"));B1=A.multiply(J0,exp_half)
    # Exact division by two needs at most one additional decimal digit.
    half=Context(prec=max(precision,len(D.as_tuple().digits)+5)).divide(D,Decimal(2))
    base=gaussian_certificate(str(half),n,[str(x) for x in q],precision)
    if base["quadratic_beta_lower"] is None:
        raise ArithmeticError("the principal spectral enclosure is unavailable")
    rhoL=lo.subtract(Decimal(1),Decimal(base["quadratic_beta_upper"]))
    rhoU=up.subtract(Decimal(1),Decimal(base["quadratic_beta_lower"]))
    if rhoL<=0:
        raise ArithmeticError("a positive principal-eigenvalue lower bound is required")
    eps=Decimal(base["projection_error_upper"])
    h2=A.multiply(hI,hI);pi2=A.multiply(pi,pi)
    # Young bounds improve the Schur derivative bounds without changing the
    # already independently valid principal-eigenvalue enclosure.
    derivative1=min(A.multiply(dI,B1)[1],A.multiply(two,J0)[1])
    derivative2=min(A.multiply(dI,J0)[1],
                    A.multiply(point(4),A.multiply(J0,exp_half))[1])
    a=A.divide(A.multiply(point(derivative2),h2),pi2)[1]
    b=A.divide(A.multiply(point(derivative1),hI),pi)[1]
    block_gap=lo.subtract(rhoL,a)
    if block_gap<=0:
        raise ArithmeticError("the complement resolvent denominator is not positive")
    t=up.divide(b,block_gap);chi=up.divide(up.multiply(b,b),block_gap)
    mu=A.near.divide(A.near.add(Decimal(base["rho_lower"]),Decimal(base["rho_upper"])),Decimal(2))
    norm2_exact=sum((Fraction(x)**2 for x in q),Fraction(0))
    norm2=A.fraction(norm2_exact,norm2_exact);norm=A.sqrt(norm2)

    entries=[];Wentries=[]
    for k in range(n):
        zF=hF*k
        J=A.multiply(J0,A.exp(_exact(A,-zF*zF/2)))
        entry=A.multiply(hI,J)
        entries.append(entry);Wentries.append(A.multiply(entry,_exact(A,zF*zF)))
    residual2=Decimal(0);WnumL=WnumU=Decimal(0)
    for i in range(n):
        rowL=rowU=wrowL=wrowU=Decimal(0)
        for j,value in enumerate(q):
            e=entries[abs(i-j)];w=Wentries[abs(i-j)]
            rowL=lo.add(rowL,lo.multiply(e[0],value))
            rowU=up.add(rowU,up.multiply(e[1],value))
            wrowL=lo.add(wrowL,lo.multiply(w[0],value))
            wrowU=up.add(wrowU,up.multiply(w[1],value))
        rlo=lo.subtract(rowL,up.multiply(mu,q[i]))
        rup=up.subtract(rowU,lo.multiply(mu,q[i]))
        rmax=max(rlo.copy_abs(),rup.copy_abs())
        residual2=up.add(residual2,up.multiply(rmax,rmax))
        WnumL=lo.add(WnumL,lo.multiply(q[i],wrowL))
        WnumU=up.add(WnumU,up.multiply(q[i],wrowU))
    residual=_sqrt_up(A,up.divide(residual2,norm2[0]))
    Wmid=A.divide((WnumL,WnumU),norm2)
    edgeL=edgeU=Decimal(0)
    for i,value in enumerate(q):
        zF=DF-(Fraction(i)+Fraction(1,2))*hF
        J=A.multiply(J0,A.exp(_exact(A,-zF*zF/2)))
        edgeL=lo.add(edgeL,lo.multiply(value,J[0]))
        edgeU=up.add(edgeU,up.multiply(value,J[1]))
    edge_mid=A.divide(A.multiply(A.sqrt(hI),(edgeL,edgeU)),norm)

    # Hilbert--Schmidt bounds, separated into even and odd subspaces.
    JD=A.multiply(J0,A.exp(_exact(A,-DF*DF/2)))
    exp_D2=A.exp(_exact(A,-DF*DF))
    HSI=A.subtract(A.divide(dI,A.multiply(two,sqrtpi)),
                    A.divide(A.subtract(one,exp_D2),A.multiply(two,pi)))
    HSplus=HSI[1]
    Te=up.add(up.divide(HSplus,Decimal(2)),Decimal(".25"))
    LFI=DF/2;LI=_exact(A,LFI)
    tail=A.divide(A.exp(_exact(A,-LFI*LFI)),A.multiply(LI,sqrtpi))[1]
    erfL=max(Decimal(0),lo.subtract(Decimal(1),tail))
    odd_sub=lo.divide(lo.multiply(erfL,erfL),Decimal(4))
    To=up.subtract(up.divide(HSplus,Decimal(2)),odd_sub)
    even_rest=_sqrt_up(A,up.subtract(Te,lo.multiply(rhoL,rhoL)))
    odd_rest=_sqrt_up(A,To)
    sigma=max(even_rest,odd_rest)
    gammaL=A.multiply(thetaI,point(lo.subtract(rhoL,sigma)))[0]
    # Odd trial x, with an explicit lower bound on its Rayleigh quotient.
    odd_factor=A.subtract(A.divide(point(6),dI),A.divide(point(8),A.multiply(A.multiply(dI,dI),dI)))
    odd_trial=A.subtract(A.subtract(one,A.multiply(J0,odd_factor)),
                         A.divide(A.multiply(point(8),JD),A.multiply(A.multiply(dI,dI),dI)))[0]
    odd_trial=max(Decimal(0),odd_trial)
    gammaU=A.multiply(thetaI,point(up.subtract(rhoU,odd_trial)))[1]
    if gammaL<=0 or gammaU<gammaL:
        raise ArithmeticError("a positive full spectral-gap interval was not established")
    direction_gap=lo.subtract(lo.subtract(mu,even_rest),chi)
    if direction_gap<=0:
        raise ArithmeticError("the even effective-operator separation is not positive")
    direction_numerator=up.add(up.add(residual,eps),chi)
    direction=A.divide(A.multiply(A.sqrt(two),point(direction_numerator)),point(direction_gap))[1]
    direction=min(direction,A.sqrt(two)[1])

    BT2=A.multiply(two,J0)
    epsT=A.divide(A.multiply(A.multiply(dI,BT2),h2),point(12))[1]
    aT=A.divide(A.multiply(A.multiply(dI,BT2),h2),pi2)[1]
    total_variation=A.multiply(point(8),A.multiply(J0,A.exp(point(-1))))
    bT=A.divide(A.multiply(total_variation,hI),pi)[1]
    t2=up.multiply(t,t)
    Efunctional=up.add(up.add(epsT,up.multiply(Decimal(2),direction)),
                       up.add(up.multiply(up.add(Decimal(1),aT),t2),
                              up.multiply(Decimal(2),up.multiply(bT,t))))
    cfunctional=A.multiply(factorI,A.add(Wmid,(Efunctional.copy_negate(),Efunctional)))
    # Half-line L2 bounds for the endpoint kernel and its derivative.
    A0=A.sqrt(A.divide(one,A.multiply(point(4),sqrtpi)))
    A1=A.sqrt(A.divide(one,A.multiply(point(8),sqrtpi)))
    Eedge=A.add(A.multiply(A0,point(up.add(direction,up.divide(t2,Decimal(2))))),
                A.add(A.multiply(A.divide(hI,pi),A.multiply(A1,point(t))),
                      A.divide(A.multiply(A.sqrt(dI),A.multiply(J0,h2)),point(24))))[1]
    Nlo=max(Decimal(0),lo.subtract(edge_mid[0],Eedge));Nup=up.add(edge_mid[1],Eedge)
    traceLo=lo.subtract(rhoL,up.divide(up.multiply(D,up.multiply(Nup,Nup)),rhoL))
    traceUp=up.subtract(rhoU,lo.divide(lo.multiply(D,lo.multiply(Nlo,Nlo)),rhoU))
    ctrace=A.multiply(factorI,(traceLo,traceUp))
    cL=max(cfunctional[0],ctrace[0]);cU=min(cfunctional[1],ctrace[1])
    if cL<=0 or cU<cL:
        raise ArithmeticError("the coefficient certificates do not give a positive intersection")
    F=A.add(A.divide(A.multiply(A.multiply(dI,dI),A.multiply(B1,B1)),point(rhoL)),
            A.multiply(dI,J0))[1]
    G=A.add(A.add(A.divide(A.multiply(point(8*4),A.multiply(A.multiply(deltaI,deltaI),A.multiply(B1,B1))),point(gammaL)),
                    A.multiply(point(4),A.multiply(J0,A.multiply(deltaI,deltaI)))),
            A.add(point(up.multiply(Decimal(2),cU)),
                  A.divide(A.multiply(thetaI,A.multiply(deltaI,A.multiply(dI,point(F)))),two)))[1]
    m_required=A.divide(A.multiply(point(16),A.multiply(deltaI,B1)),point(gammaL))[1]
    m_min=max(2,int(m_required.to_integral_value(rounding=ROUND_CEILING)))
    sign_ratio=up.divide(G,cL)
    m_sign=max(m_min,int(sign_ratio.to_integral_value(rounding=ROUND_FLOOR))+1)
    output=dict(D=D_string,V="2",N=n,precision=precision,
        rho_lower=str(rhoL),rho_upper=str(rhoU),mu_trial=str(mu),
        residual_upper=str(residual),projection_error_upper=str(eps),
        complement_a_upper=str(a),coupling_b_upper=str(b),
        complement_ratio_upper=str(t),schur_correction_upper=str(chi),
        hs_squared_upper=str(HSplus),hs_even_squared_upper=str(Te),hs_odd_squared_upper=str(To),
        even_rest_upper=str(even_rest),odd_rest_upper=str(odd_rest),rest_upper=str(sigma),
        odd_trial_lower=str(odd_trial),gamma_lower=str(gammaL),gamma_upper=str(gammaU),
        gamma_width_upper=str(up.subtract(gammaU,gammaL)),
        direction_gap_lower=str(direction_gap),direction_error_upper=str(direction),
        observable_mid_lower=str(Wmid[0]),observable_mid_upper=str(Wmid[1]),
        observable_error_upper=str(Efunctional),observable_averaging_error_upper=str(epsT),
        observable_complement_a_upper=str(aT),observable_coupling_b_upper=str(bT),
        edge_mid_lower=str(edge_mid[0]),edge_mid_upper=str(edge_mid[1]),
        edge_error_upper=str(Eedge),edge_integral_lower=str(Nlo),edge_integral_upper=str(Nup),
        c_functional_lower=str(cfunctional[0]),c_functional_upper=str(cfunctional[1]),
        c_trace_lower=str(ctrace[0]),c_trace_upper=str(ctrace[1]),
        c_lower=str(cL),c_upper=str(cU),C_lower=str(cU.copy_negate()),C_upper=str(cL.copy_negate()),
        c_functional_width_upper=str(up.subtract(cfunctional[1],cfunctional[0])),
        c_trace_width_upper=str(up.subtract(ctrace[1],ctrace[0])),
        c_width_upper=str(up.subtract(cU,cL)),
        functional_coefficient_error_upper=str(up.multiply(factorI[1],Efunctional)),
        B1_upper=str(B1[1]),B2_upper=str(J0[1]),F_upper=str(F),G_upper=str(G),
        m_resolution_min=m_min,m_sign_all=m_sign,
        q_sha256=sha256("\n".join(map(str,q)).encode("ascii")).hexdigest(),
        method="even/odd HS gap; effective projection; observable and endpoint intersection",
        derivative_bounds="min(Schur,Young); endpoint half-line L2 bounds",
    )
    return output


def boundary_remainder(record,m):
    """Propagate the certified constants through the unsimplified theorem bound."""
    if not isinstance(m,int) or isinstance(m,bool) or m<2:
        raise ValueError("m must be an integer at least two")
    A=_Intervals(int(record["precision"]));up,lo=A.up,A.down;point=A.point
    D=Decimal(record["D"]);V=Decimal(2);delta=up.subtract(D,V)
    theta=_exact(A,Fraction(V)/Fraction(D));rhoL=Decimal(record["rho_lower"]);rhoU=Decimal(record["rho_upper"])
    gamma=Decimal(record["gamma_lower"]);B1=Decimal(record["B1_upper"]);B2=Decimal(record["B2_upper"])
    cL=Decimal(record["c_lower"]);cU=Decimal(record["c_upper"]);F=Decimal(record["F_upper"])
    def bound(k):
        e=up.divide(up.multiply(up.multiply(Decimal(2)*V,delta),B1),Decimal(k))
        if e>lo.divide(gamma,Decimal(4)):
            raise ArithmeticError("m does not meet the certified small-perturbation condition")
        denominator=lo.subtract(gamma,up.multiply(Decimal(2),e))
        first=up.divide(up.multiply(e,e),denominator)
        second=up.divide(up.multiply(up.multiply(Decimal(2)*V,B2),up.multiply(delta,delta)),Decimal(k*k))
        third=up.divide(cU,Decimal(k*(k-1)))
        fourth=A.divide(A.multiply(theta,A.multiply(point(delta),A.multiply(point(D),point(F)))),point(4*k*(k-1)))[1]
        return up.add(up.add(first,second),up.add(third,fourth))
    R=bound(m);R2=bound(2*m)
    betaInf=A.subtract(point(1),A.multiply(theta,(rhoL,rhoU)))
    betaM=A.add(A.add(betaInf,A.divide((cL,cU),point(m))),(R.copy_negate(),R))
    coefficient_mid=A.near.divide(A.near.add(cL,cU),Decimal(2))
    coefficient_error=max(up.subtract(coefficient_mid,cL),up.subtract(cU,coefficient_mid))
    return dict(D=record["D"],N=int(record["N"]),m=m,
        c_midpoint=str(coefficient_mid),c_midpoint_error_upper=str(coefficient_error),
        remainder_upper=str(R),richardson_error_upper=str(up.add(R,up.multiply(Decimal(2),R2))),
        exact_coefficient_corrected_error_upper=str(R),
        midpoint_coefficient_corrected_error_upper=str(up.add(R,up.divide(coefficient_error,Decimal(m)))),
        beta_infty_lower=str(betaInf[0]),beta_infty_upper=str(betaInf[1]),
        beta_m_lower=str(betaM[0]),beta_m_upper=str(betaM[1]))


def boundary_self_check(D_string,n,q):
    """Replay a supplied exact vector at two precisions and an alien context."""
    first=boundary_certificate(D_string,n,q,60)
    finer=boundary_certificate(D_string,n,q,80)
    for lower,upper in (("rho_lower","rho_upper"),("gamma_lower","gamma_upper"),
                        ("c_lower","c_upper")):
        assert Decimal(first[lower])<=Decimal(finer[lower])
        assert Decimal(finer[upper])<=Decimal(first[upper])
    with localcontext() as unrelated:
        unrelated.prec=2;unrelated.rounding=ROUND_CEILING
        assert boundary_certificate(D_string,n,q,60)==first
    return dict(D=D_string,N=n,precision_nesting_60_80=True,
                ambient_context_invariance=True)
