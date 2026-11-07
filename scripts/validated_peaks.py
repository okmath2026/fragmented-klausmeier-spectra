"""Direct continuous Gaussian certificates on fixed-span fragmented intervals.

Only Python's standard library is used. Positive exact decimal trial vectors
are proof inputs, not asserted eigenvectors. The matrix arithmetic, exact
rational geometry, midpoint averaging and complement-space errors are enclosed
independently of any FFT or floating eigensolve.
"""
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from hashlib import sha256
from validated_gaussian import _Intervals, _pi_fraction_bounds, gaussian_certificate


def _exact(A, value):
    value = Fraction(value)
    return A.fraction(value,value)


def fragment_certificate(D_string,m,r,q,precision=60,V_string="2"):
    """Enclose beta_m on m equal intervals, using r midpoint cells per patch.

    Geometry has exact total length V and fixed endpoint span D. The vector
    must be exactly even, allowing reflection to halve the CW row work. Every
    discarded row has the same exact ratio as its reflected retained row.
    """
    if not isinstance(D_string,str) or not isinstance(V_string,str):
        raise TypeError("D_string and V_string must be exact decimal strings")
    D,V = Decimal(D_string),Decimal(V_string)
    if not D.is_finite() or not V.is_finite() or not D>V>0:
        raise ValueError("requires finite exact geometry D>V>0")
    if any(not isinstance(k,int) or isinstance(k,bool) or k<2 for k in (m,r)):
        raise ValueError("m and r must be integers at least two")
    n=m*r
    q=[Decimal(str(value)) for value in q]
    if len(q)!=n or any(not x.is_finite() or x<=0 for x in q):
        raise ValueError("q must contain m*r finite strictly positive decimals")
    if q!=q[::-1]:
        raise ValueError("the saved exact trial vector must be reflection symmetric")
    A=_Intervals(precision);lo,up,point=A.down,A.up,A.point
    DF,VF=Fraction(D),Fraction(V)
    ellF=VF/m;gapF=(DF-VF)/(m-1);stepF=ellF+gapF;hF=ellF/r
    vI,hI=_exact(A,VF),_exact(A,hF)
    pi=A.fraction(*_pi_fraction_bounds(precision))
    J0=A.divide(point(1),A.sqrt(A.multiply(point(2),pi)))
    B1=A.multiply(J0,A.exp(point("-.5")))
    blocks=[]
    for k in range(m):
        row=[]
        for d in range(1-r,r):
            z=k*stepF+d*hF
            value=A.multiply(A.multiply(hI,J0),A.exp(_exact(A,-z*z/2)))
            row.append(value)
        blocks.append(row)
    cwL,cwU=None,None
    for i in range((n+1)//2):
        patch_i,a=divmod(i,r)
        rowL=rowU=Decimal(0)
        for j,value in enumerate(q):
            patch_j,b=divmod(j,r)
            if patch_i>=patch_j:k,d=patch_i-patch_j,a-b
            else:k,d=patch_j-patch_i,b-a
            entry=blocks[k][d+r-1]
            rowL=lo.add(rowL,lo.multiply(entry[0],value))
            rowU=up.add(rowU,up.multiply(entry[1],value))
        ratioL,ratioU=lo.divide(rowL,q[i]),up.divide(rowU,q[i])
        cwL=ratioL if cwL is None else min(cwL,ratioL)
        cwU=ratioU if cwU is None else max(cwU,ratioU)
    h2=A.multiply(hI,hI);pi2=A.multiply(pi,pi)
    eps=A.divide(A.multiply(A.multiply(vI,J0),h2),point(12))[1]
    a=A.divide(A.multiply(A.multiply(vI,J0),h2),pi2)[1]
    b=A.divide(A.multiply(A.multiply(vI,B1),hI),pi)[1]
    projectedL=lo.subtract(cwL,eps);projectedU=up.add(cwU,eps)
    denominator=lo.subtract(projectedL,a)
    if denominator<=0:
        raise ArithmeticError("the direct fragment complement gap is not positive")
    correction=up.divide(up.multiply(b,b),denominator)
    rhoL=projectedL;rhoU=min(Decimal(1),up.add(projectedU,correction))
    betaL,betaU=lo.subtract(Decimal(1),rhoU),up.subtract(Decimal(1),rhoL)
    if not Decimal(0)<rhoL<=rhoU<=Decimal(1):
        raise ArithmeticError("invalid continuous fragment enclosure")
    return dict(D=D_string,V=V_string,m=m,cells_per_patch=r,N=n,precision=precision,
        patch_length_numerator=ellF.numerator,patch_length_denominator=ellF.denominator,
        gap_numerator=gapF.numerator,gap_denominator=gapF.denominator,
        cell_width_numerator=hF.numerator,cell_width_denominator=hF.denominator,
        cw_lower=str(cwL),cw_upper=str(cwU),cw_width_upper=str(up.subtract(cwU,cwL)),
        projection_error_upper=str(eps),complement_a_upper=str(a),coupling_b_upper=str(b),
        complement_gap_lower=str(denominator),schur_correction_upper=str(correction),
        rho_lower=str(rhoL),rho_upper=str(rhoU),beta_lower=str(betaL),beta_upper=str(betaU),
        beta_width_upper=str(up.subtract(betaU,betaL)),
        q_sha256=sha256("\n".join(map(str,q)).encode("ascii")).hexdigest(),
        method="direct union midpoint CW; cell-average quadratic complement enclosure")


def rayleigh_envelope_certificate(D_string,n,q,precision=60,V_string="2"):
    """Certify a one-sided Rayleigh envelope without a spectral-gap input.

    The only numerical spectral input is the standalone continuous Gaussian
    principal-eigenvalue enclosure on an interval of exact length D.  In
    particular, this computation never calls boundary_certificate and uses
    neither an eigenfunction-direction enclosure nor a boundary-coefficient
    enclosure.  All geometry is converted to exact rational numbers first.

    For theta=V/D, Delta=D-V, and kappa=theta*Delta/D, the result certifies

        beta_m <= 1-rho_lower*(theta-kappa/(m-1)) + S/(m*(m-1)),  m>=2.

    Here S is a certified upper bound on the sum of the direct Taylor and
    cell-cancellation Rayleigh errors.  The coefficient substitution uses
    only the analytic inequality c_D<=kappa*rho; it does not divide by a
    spectral gap.  gaussian_certificate documents the canonical decimal
    trial vector used by its CW loop; its hash is retained in this record.
    """
    if not isinstance(D_string,str) or not isinstance(V_string,str):
        raise TypeError("D_string and V_string must be exact decimal strings")
    D,V=Decimal(D_string),Decimal(V_string)
    if not D.is_finite() or not V.is_finite() or not D>V>0:
        raise ValueError("requires finite exact geometry D>V>0")
    if not isinstance(n,int) or isinstance(n,bool) or n<1:
        raise ValueError("n must be a positive integer")
    q=[Decimal(str(value)) for value in q]
    if len(q)!=n or any(not x.is_finite() or x<=0 for x in q):
        raise ValueError("q must contain n finite strictly positive decimals")
    A=_Intervals(precision);lo,up,point=A.down,A.up,A.point
    # Division by two is terminating; choose enough digits to retain the
    # input decimal geometry exactly, independently of the ambient context.
    with localcontext(A.near) as exact:
        exact.prec=max(precision,len(D.as_tuple().digits)+1)
        half_D=exact.divide(D,Decimal(2))
    principal=gaussian_certificate(str(half_D),n,q,precision)
    if principal["quadratic_beta_lower"] is None:
        raise ArithmeticError("the principal quadratic complement enclosure is unavailable")
    rhoL=lo.subtract(Decimal(1),Decimal(principal["quadratic_beta_upper"]))
    rhoU=up.subtract(Decimal(1),Decimal(principal["quadratic_beta_lower"]))
    if not Decimal(0)<rhoL<=rhoU:
        raise ArithmeticError("a strictly positive principal lower bound is required")
    DF,VF=Fraction(D),Fraction(V)
    deltaF=DF-VF;thetaF=VF/DF;kappaF=thetaF*deltaF/DF
    dI,deltaI,thetaI=_exact(A,DF),_exact(A,deltaF),_exact(A,thetaF)
    kappaI=_exact(A,kappaF)
    pi=A.fraction(*_pi_fraction_bounds(precision))
    J0=A.divide(point(1),A.sqrt(A.multiply(point(2),pi)))
    B1=A.multiply(J0,A.exp(point("-.5")))
    # Exact Gaussian identities: ||J||_2=(2*sqrt(pi))^(-1/2),
    # ||J'||_1=2*J(0), and ||J''||_1=4*B1.
    J_l2=A.sqrt(A.divide(point(1),A.multiply(point(2),A.sqrt(pi))))
    Jprime_l1=A.multiply(point(2),J0)
    Jsecond_l1=A.multiply(point(4),B1)
    first=min(A.multiply(dI,B1)[1],Jprime_l1[1])
    second=min(A.multiply(dI,J0)[1],Jsecond_l1[1])
    F=up.add(up.divide(up.multiply(first,first),rhoL),second)
    numerator=A.multiply(thetaI,J0)
    for factor in (A.sqrt(dI),J_l2,dI,A.multiply(deltaI,deltaI)):
        numerator=A.multiply(numerator,factor)
    taylor=A.divide(numerator,A.multiply(point(6),point(rhoL)))[1]
    cancellation=A.divide(A.multiply(A.multiply(A.multiply(thetaI,deltaI),dI),point(F)),point(4))[1]
    S=up.add(taylor,cancellation)
    betaInf=A.subtract(point(1),A.multiply(thetaI,(rhoL,rhoU)))
    return dict(D=D_string,V=V_string,N=n,precision=precision,
        rho_lower=str(rhoL),rho_upper=str(rhoU),
        principal_cw_lower=principal["rho_lower"],principal_cw_upper=principal["rho_upper"],
        principal_projection_error_upper=principal["projection_error_upper"],
        principal_complement_a_upper=principal["complement_a_upper"],
        principal_coupling_b_upper=principal["coupling_b_upper"],
        principal_complement_gap_lower=principal["complement_gap_lower"],
        theta_numerator=thetaF.numerator,theta_denominator=thetaF.denominator,
        delta_numerator=deltaF.numerator,delta_denominator=deltaF.denominator,
        kappa_numerator=kappaF.numerator,kappa_denominator=kappaF.denominator,
        theta_lower=str(thetaI[0]),theta_upper=str(thetaI[1]),
        kappa_lower=str(kappaI[0]),kappa_upper=str(kappaI[1]),
        B1_upper=str(B1[1]),B2_upper=str(J0[1]),
        gaussian_l2_upper=str(J_l2[1]),
        gaussian_derivative_l1_upper=str(Jprime_l1[1]),
        gaussian_second_derivative_l1_upper=str(Jsecond_l1[1]),
        derivative_operator_upper=str(first),second_derivative_operator_upper=str(second),
        F_young_upper=str(F),rayleigh_taylor_constant_upper=str(taylor),
        rayleigh_cell_constant_upper=str(cancellation),rayleigh_remainder_constant_upper=str(S),
        beta_infty_lower=str(betaInf[0]),beta_infty_upper=str(betaInf[1]),
        first_valid_m=2,q_sha256=principal["q_sha256"],
        source_q_sha256=sha256("\n".join(map(str,q)).encode("ascii")).hexdigest(),
        arithmetic=principal["arithmetic"],
        method="principal Gaussian CW enclosure; Young derivative bounds; one-sided Rayleigh envelope")


def rayleigh_tail_certificate(envelope,M):
    """Bound every m>=M by a decreasing, gap-free Rayleigh majorant.

    The exact factor theta-kappa/(M-1) is positive for every M>=2.
    Consequently replacing rho by its lower enclosure has the required
    direction.  Both positive error terms decrease with m, so evaluation at
    M excludes the complete infinite tail without a Schur anchor.
    """
    if not isinstance(M,int) or isinstance(M,bool) or M<2:
        raise ValueError("the anchor M must be an integer at least two")
    DF,VF=Fraction(envelope["D"]),Fraction(envelope["V"])
    if not DF>VF>0:
        raise ValueError("requires exact geometry D>V>0")
    A=_Intervals(int(envelope["precision"]));lo,up=A.down,A.up
    rhoL,rhoU=Decimal(envelope["rho_lower"]),Decimal(envelope["rho_upper"])
    S=Decimal(envelope["rayleigh_remainder_constant_upper"])
    if (any(not x.is_finite() for x in (rhoL,rhoU,S))
            or not Decimal(0)<rhoL<=rhoU or S<=0):
        raise ValueError("requires positive finite principal and Rayleigh bounds")
    thetaF=VF/DF;kappaF=thetaF*(DF-VF)/DF
    # Check saved rational geometry as well as recomputing it.  The input
    # envelope is a certificate record, not an unchecked source of geometry.
    for prefix,value in (("theta",thetaF),("delta",DF-VF),("kappa",kappaF)):
        if Fraction(int(envelope[prefix+"_numerator"]),int(envelope[prefix+"_denominator"]))!=value:
            raise ValueError("inconsistent exact Rayleigh-envelope geometry")
    factorF=thetaF-kappaF/(M-1)
    if factorF<=0:
        raise ArithmeticError("the coefficient of the principal lower bound must be positive")
    factor=_exact(A,factorF)
    constant_tail=up.subtract(Decimal(1),lo.multiply(rhoL,factor[0]))
    remainder_tail=up.divide(S,Decimal(M*(M-1)))
    tail=up.add(constant_tail,remainder_tail)
    return dict(D=envelope["D"],V=envelope["V"],N=int(envelope["N"]),
        precision=int(envelope["precision"]),anchor_m=M,
        rho_lower=str(rhoL),rho_upper=str(rhoU),B1_upper=envelope["B1_upper"],
        kappa_numerator=kappaF.numerator,kappa_denominator=kappaF.denominator,
        principal_factor_numerator=factorF.numerator,principal_factor_denominator=factorF.denominator,
        principal_factor_lower=str(factor[0]),principal_factor_upper=str(factor[1]),
        rayleigh_remainder_constant_upper=str(S),
        anchor_constant_part_upper=str(constant_tail),anchor_remainder_upper=str(remainder_tail),
        beta_infty_lower=envelope["beta_infty_lower"],beta_infty_upper=envelope["beta_infty_upper"],
        all_m_ge_anchor_beta_upper=str(tail),q_sha256=envelope["q_sha256"],
        uses_quantified_spectral_gap=False,uses_boundary_coefficient_interval=False,
        method="decreasing one-sided Rayleigh envelope for every integer m>=anchor")


def tail_certificate(record,uniform):
    """Certify an upper bound for every beta_m with m>=the saved anchor.

    Inputs must be independently reconstructed boundary_certificate and
    sharp_eventual_sign outputs. No float value enters this propagation.
    """
    if (record["D"],int(record["N"]))!=(uniform["D"],int(uniform["N"])):
        raise ValueError("the envelope and uniform remainder records must match")
    A=_Intervals(int(record["precision"]));up,lo=A.up,A.down
    DF,VF=Fraction(record["D"]),Fraction(record["V"])
    M=int(uniform["anchor_m"])
    K=Decimal(uniform["uniform_remainder_constant_upper"])
    cU=Decimal(record["c_upper"])
    if M<2 or not K.is_finite() or K<=0 or not cU.is_finite() or cU<0:
        raise ValueError("requires an anchor M>=2 and finite positive remainder data")
    cL_decimal=Decimal(record["c_lower"])
    if not cL_decimal.is_finite() or cL_decimal<=0:raise ValueError("requires positive c_lower")
    cL=Fraction(cL_decimal);KF=Fraction(K);dyadic=M
    while KF/(2*dyadic-1)+2*KF/(dyadic-1)>=cL:dyadic+=1
    dyadic_margin=cL-KF/(2*dyadic-1)-2*KF/(dyadic-1)
    theta=_exact(A,VF/DF)
    betaInf=A.subtract(A.point(1),A.multiply(theta,(Decimal(record["rho_lower"]),Decimal(record["rho_upper"]))))
    tail=up.add(up.add(betaInf[1],up.divide(cU,Decimal(M))),up.divide(K,Decimal(M*(M-1))))
    return dict(D=record["D"],V=record["V"],N=int(record["N"]),precision=int(record["precision"]),
        anchor_m=M,uniform_remainder_constant_upper=str(K),c_upper=str(cU),
        B1_upper=record["B1_upper"],
        beta_infty_lower=str(betaInf[0]),beta_infty_upper=str(betaInf[1]),
        all_m_ge_anchor_beta_upper=str(tail),
        dyadic_first_sufficient_m=dyadic,
        dyadic_strict_margin_lower=str(A.fraction(dyadic_margin,dyadic_margin)[0]),
        method="uniform analytic infinite-tail bound from a certified Schur anchor")


def global_peak_certificate(finite_rows,tail,winner_m):
    """Prove strict uniqueness of the global maximum over every integer m>=2."""
    if not finite_rows:raise ValueError("finite direct certificates are required")
    D=tail["D"];M=int(tail["anchor_m"])
    if any(row["D"]!=D or Fraction(row["V"])!=Fraction(tail["V"]) for row in finite_rows):
        raise ValueError("mixed geometries or total habitat lengths")
    bym={int(row["m"]):row for row in finite_rows}
    if len(bym)!=len(finite_rows) or set(bym)!=set(range(2,M)):
        raise ValueError("finite cases must cover every integer 2<=m<anchor exactly once")
    if winner_m not in bym:raise ValueError("the proposed maximiser is not a finite certified case")
    A=_Intervals(int(tail["precision"]));lo,up=A.down,A.up
    winner=bym[winner_m];lower=Decimal(winner["beta_lower"])
    competitors=[row for m,row in bym.items() if m!=winner_m]
    runner=max(competitors,key=lambda row:Decimal(row["beta_upper"]))
    runner_upper=Decimal(runner["beta_upper"])
    tail_upper=Decimal(tail["all_m_ge_anchor_beta_upper"])
    finite_margin=lo.subtract(lower,runner_upper);tail_margin=lo.subtract(lower,tail_upper)
    if min(finite_margin,tail_margin)<=0:
        raise ArithmeticError("strict global peak separation is not established")
    separation=min(finite_margin,tail_margin)
    robust_radius=lo.divide(separation,Decimal(2))
    volume_upper=_exact(A,Fraction(tail["V"]))[1]
    derivative=Decimal(tail["B1_upper"])
    if not derivative.is_finite() or derivative<=0:raise ValueError("requires positive B1_upper")
    span_denominator=up.multiply(Decimal(2),up.multiply(volume_upper,derivative))
    span_radius=lo.divide(separation,span_denominator)
    return dict(D=D,V=tail["V"],winner_m=winner_m,finite_first_m=2,finite_last_m=M-1,
        finite_cases=len(finite_rows),tail_first_m=M,
        winner_beta_lower=winner["beta_lower"],winner_beta_upper=winner["beta_upper"],
        competing_finite_upper=str(runner_upper),largest_competing_upper_m=int(runner["m"]),
        infinite_tail_upper=str(tail_upper),finite_separation_lower=str(finite_margin),
        tail_separation_lower=str(tail_margin),overall_separation_lower=str(separation),
        kernel_eta_certified_radius_strict=str(robust_radius),unique_global_maximum=True,
        span_radius_certified_strict=str(span_radius),
        method="finite direct continuum intervals plus analytic infinite-tail exclusion")


def peak_self_check(D,m,r,q):
    first=fragment_certificate(D,m,r,q,60)
    finer=fragment_certificate(D,m,r,q,80)
    assert Decimal(first["beta_lower"])<=Decimal(finer["beta_lower"])
    assert Decimal(finer["beta_upper"])<=Decimal(first["beta_upper"])
    with localcontext() as unrelated:
        unrelated.prec=2;unrelated.rounding=ROUND_CEILING
        assert fragment_certificate(D,m,r,q,60)==first
    return dict(D=D,m=m,cells_per_patch=r,precision_nesting_60_80=True,
                ambient_context_invariance=True)


def rayleigh_self_check(D,n,q,M=2,V_string="2"):
    """Replay the production envelope at two precisions and alien context."""
    first=rayleigh_envelope_certificate(D,n,q,60,V_string)
    finer=rayleigh_envelope_certificate(D,n,q,80,V_string)
    for lower,upper in (("rho_lower","rho_upper"),
                        ("beta_infty_lower","beta_infty_upper")):
        assert Decimal(first[lower])<=Decimal(finer[lower])
        assert Decimal(finer[upper])<=Decimal(first[upper])
    for key in ("B1_upper","B2_upper","gaussian_l2_upper",
                "F_young_upper","rayleigh_taylor_constant_upper",
                "rayleigh_cell_constant_upper","rayleigh_remainder_constant_upper"):
        assert Decimal(finer[key])<=Decimal(first[key])
    tail=rayleigh_tail_certificate(first,M)
    fine_tail=rayleigh_tail_certificate(finer,M)
    assert Decimal(fine_tail["all_m_ge_anchor_beta_upper"])<=Decimal(tail["all_m_ge_anchor_beta_upper"])
    # Check the rational factor at m=2 as well as the requested anchor.  This
    # includes D close to V and fractional geometry, with no float geometry.
    theta=Fraction(V_string)/Fraction(D)
    smallest_factor=rayleigh_tail_certificate(first,2)
    assert Fraction(smallest_factor["principal_factor_numerator"],
                    smallest_factor["principal_factor_denominator"])==theta*theta
    with localcontext() as unrelated:
        unrelated.prec=2;unrelated.rounding=ROUND_CEILING
        assert rayleigh_envelope_certificate(D,n,q,60,V_string)==first
        assert rayleigh_tail_certificate(first,M)==tail
    return dict(D=D,V=V_string,N=n,anchor_m=M,
        principal_precision_nesting_60_80=True,rayleigh_upper_precision_nesting_60_80=True,
        exact_positive_factor_check=True,ambient_context_invariance=True,
        uses_quantified_spectral_gap=False,uses_boundary_coefficient_interval=False)
