"""Directed certificates for the linear Dirichlet water comparison problem.

The water cap is a comparison bound for the nonlinear stationary water
equation, not an asserted water/vegetation steady state. Standard library only.
"""
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction
from validated_gaussian import _Intervals


def water_certificate(length_string,dw_string,distance_string="0",precision=60,
                      Amax_string="1.8",R_string="4",B_string=".45"):
    """Enclose max z on plant points at distance >=distance from water centre.

    z solves -dw*z''+z=1 on an interval of length L with zero endpoint values.
    distance=0 gives its whole-interval maximum. For ambient water on I_D and
    even m plants, distance is the exact half central gap; the bound is then
    the supremum over the plant habitat, not necessarily over all of I_D.
    """
    values=[length_string,dw_string,distance_string,Amax_string,R_string,B_string]
    if any(not isinstance(v,str) for v in values):
        raise TypeError("all physical inputs must be exact decimal strings")
    L,dw,a,Amax,R,B=[Decimal(v) for v in values]
    if any(not x.is_finite() for x in (L,dw,a,Amax,R,B)):
        raise ValueError("inputs must be finite")
    LF,df,af=map(Fraction,(L,dw,a))
    if not (LF>0 and df>0 and 0<=af<=LF/2 and Amax>=0 and R>0 and B>0):
        raise ValueError("requires L,dw,R,B>0, Amax>=0 and 0<=distance<=L/2")
    A=_Intervals(precision);lo,up,point=A.down,A.up,A.point
    exact=lambda x:A.fraction(x,x)
    root=A.sqrt(point(dw));x=A.divide(exact(LF/2),root)
    y=A.divide(exact(af),root)
    # cosh(y)/cosh(x) = [exp(-(x-y))+exp(-(x+y))]/[1+exp(-2x)].
    # Every exponent is nonpositive, avoiding overflow on small water scales.
    t=A.exp(A.negate(x))
    if af==0:
        # Stable form 1-sech(x)=(1-exp(-x))^2/(1+exp(-2x)).
        diff=A.subtract(point(1),t)
        if diff[0]<0:raise ArithmeticError("positive water cap not resolved")
        q=A.divide(A.multiply(diff,diff),A.add(point(1),A.multiply(t,t)))
    else:
        numerator=A.add(A.exp(A.negate(A.subtract(x,y))),A.exp(A.negate(A.add(x,y))))
        q=A.subtract(point(1),A.divide(numerator,A.add(point(1),A.multiply(t,t))))
    qL=max(Decimal(0),q[0]);qU=min(Decimal(1),q[1])
    # Whole-interval cap <=min(1,L^2/(8dw)), also valid on its plant subset.
    quadratic=exact(LF*LF/(8*df))[1]
    qU=min(qU,quadratic)
    if not 0<=qL<=qU<=1:raise ArithmeticError("invalid water cap interval")
    water_max=A.multiply(point(Amax),(qL,qU))
    reaction=A.subtract(A.multiply(point(R),water_max),point(B))
    margin=lo.subtract(B,up.multiply(R,water_max[1]))
    return dict(water_length=length_string,dw=dw_string,nearest_plant_distance=distance_string,
        precision=precision,Amax=Amax_string,R=R_string,B=B_string,
        q_lower=str(qL),q_upper=str(qU),quadratic_cap_upper=str(min(Decimal(1),quadratic)),
        comparison_cap_lower=str(water_max[0]),comparison_cap_upper=str(water_max[1]),
        reaction_M_lower=str(reaction[0]),reaction_M_upper=str(reaction[1]),
        water_only_margin_lower=str(margin),water_only_decision="excluded" if margin>0 else "inconclusive",
        method="Dirichlet water comparison; exact cosh ratio; directed exponentials")


def rational_water_certificate(length,dw,distance=Fraction(0),precision=60):
    """Water cap for exact rational geometry without finite-decimal restriction.

    This function passes enclosing rational endpoint data into the arithmetic
    directly, so patch widths 2/m and central gaps need not have terminating
    decimal representations. The result records the exact fractions.
    """
    L,d,a=map(Fraction,(length,dw,distance))
    if not (L>0 and d>0 and 0<=a<=L/2):raise ValueError("invalid rational water geometry")
    A=_Intervals(precision);lo,up,point=A.down,A.up,A.point
    exact=lambda x:A.fraction(x,x)
    root=A.sqrt(exact(d));x=A.divide(exact(L/2),root);y=A.divide(exact(a),root)
    t=A.exp(A.negate(x))
    if a==0:
        diff=A.subtract(point(1),t)
        if diff[0]<0:raise ArithmeticError("positive water cap not resolved")
        q=A.divide(A.multiply(diff,diff),A.add(point(1),A.multiply(t,t)))
    else:
        ratio=A.divide(A.add(A.exp(A.negate(A.subtract(x,y))),A.exp(A.negate(A.add(x,y)))),
                       A.add(point(1),A.multiply(t,t)))
        q=A.subtract(point(1),ratio)
    qL=max(Decimal(0),q[0]);qU=min(Decimal(1),q[1],exact(L*L/(8*d))[1])
    if not 0<=qL<=qU<=1:raise ArithmeticError("invalid water cap")
    reaction=A.subtract(A.multiply(point("7.2"),(qL,qU)),point(".45"))
    margin=lo.subtract(Decimal(".45"),up.multiply(Decimal("7.2"),qU))
    return dict(length_numerator=L.numerator,length_denominator=L.denominator,
        dw_numerator=d.numerator,dw_denominator=d.denominator,
        distance_numerator=a.numerator,distance_denominator=a.denominator,precision=precision,
        Amax="1.8",R="4",B=".45",q_lower=str(qL),q_upper=str(qU),
        quadratic_cap_upper=str(min(Decimal(1),exact(L*L/(8*d))[1])),
        reaction_M_lower=str(reaction[0]),reaction_M_upper=str(reaction[1]),
        water_only_margin_lower=str(margin),water_only_decision="excluded" if margin>0 else "inconclusive",
        method="exact rational Dirichlet water comparison; directed exponential cap")


def coupled_margin(water,beta_lower,dv_string=".1"):
    A=_Intervals(int(water["precision"]));lo,up=A.down,A.up
    dv=Decimal(dv_string)
    if not dv.is_finite() or dv<0:raise ValueError("dv must be finite nonnegative")
    margin=lo.subtract(lo.multiply(dv,Decimal(beta_lower)),Decimal(water["reaction_M_upper"]))
    return dict(dv=dv_string,plant_beta_lower=str(beta_lower),coupled_margin_lower=str(margin),
                coupled_decision="excluded" if margin>0 else "inconclusive")


def synergy_certificate(water,beta_lower,beta_upper,dv_string=".6"):
    """Compare three sufficient tests using the same certified spectral interval.

    Negative upper margins prove that the separate sufficient tests do not
    pass; they do not assert existence of a stationary vegetation state.
    """
    A=_Intervals(int(water["precision"]));lo,up=A.down,A.up
    lower,upper,dv=map(Decimal,(str(beta_lower),str(beta_upper),dv_string))
    if not (lower.is_finite() and upper.is_finite() and lower<=upper):
        raise ValueError("ordered finite spectral endpoints are required")
    combined=coupled_margin(water,str(lower),dv_string)
    crude_reaction=A.subtract(A.multiply(A.point(water["Amax"]),A.point(water["R"])),A.point(water["B"]))
    crude_lower=lo.subtract(lo.multiply(dv,lower),crude_reaction[1])
    crude_upper=up.subtract(up.multiply(dv,upper),crude_reaction[0])
    joint_upper=up.subtract(up.multiply(dv,upper),Decimal(water["reaction_M_lower"]))
    return dict(**combined,plant_beta_upper=str(upper),
        coupled_margin_upper=str(joint_upper),
        water_only_margin_upper=str(up.minus(Decimal(water["reaction_M_lower"]))),
        crude_margin_lower=str(crude_lower),crude_margin_upper=str(crude_upper),
        crude_decision="excluded" if crude_lower>0 else "inconclusive")


def first_water_only_m(dw_string,maximum=10000):
    dw=Fraction(dw_string)
    for m in range(2,maximum+1):
        current=rational_water_certificate(Fraction(2,m),dw)
        if current["water_only_decision"]=="excluded":
            previous=rational_water_certificate(Fraction(2,m-1),dw)
            if m>2 and Decimal(previous["q_lower"])<=Decimal(".0625"):
                raise ArithmeticError("minimal water-only threshold not separated")
            return dict(V="2",dw=dw_string,first_m=m,
                q_first_lower=current["q_lower"],q_first_upper=current["q_upper"],
                previous_m=m-1,q_previous_lower=previous["q_lower"],q_previous_upper=previous["q_upper"],
                water_only_margin_lower=current["water_only_margin_lower"],
                search_first_m=2,predecessor_separation_required=m>2,
                precision=60,method="monotone exact water cap; adjacent certified threshold")
    raise ArithmeticError("no water-only threshold in search range")


def water_self_check():
    first=rational_water_certificate(Fraction(2,3),Fraction(".1"))
    finer=rational_water_certificate(Fraction(2,3),Fraction(".1"),precision=80)
    assert Decimal(first["q_lower"])<=Decimal(finer["q_lower"])<=Decimal(finer["q_upper"])<=Decimal(first["q_upper"])
    ambient=rational_water_certificate(Fraction(6),Fraction(1),Fraction(2,3))
    with localcontext() as alien:
        alien.prec=2;alien.rounding=ROUND_CEILING
        assert rational_water_certificate(Fraction(2,3),Fraction(".1"))==first
        assert rational_water_certificate(Fraction(6),Fraction(1),Fraction(2,3))==ambient
    return dict(precision_nesting_60_80=True,ambient_context_invariance=True,
                exact_rational_patch_and_ambient_geometry=True)
