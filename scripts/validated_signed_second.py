"""Signed second fragmentation coefficient by a continuous energy enclosure.

Exact saved decimal feature vectors are checked with directed arithmetic.
Finite Gram matrices, the omitted positive Gaussian operator, eigenfunction
direction, endpoint derivatives and a deflated resolvent energy are all
included. No approximate eigensolve or approximate linear solve is trusted.
"""
from decimal import Decimal,localcontext,ROUND_CEILING
from fractions import Fraction
from math import factorial
from hashlib import sha256
from validated_gaussian import _Intervals
from validated_gaussian_rank import gaussian_gram_enclosure,gaussian_rank_certificate

if not __debug__:raise RuntimeError('Run validated certificates without -O.')


def _exact(A,x):
    x=Fraction(x);return A.fraction(x,x)


def _square(A,x):
    if x[0]<=0<=x[1]:
        return Decimal(0),max(A.up.multiply(a,a) for a in x)
    return min(A.down.multiply(a,a) for a in x),max(A.up.multiply(a,a) for a in x)


def _sum(A,values):
    answer=A.point(0)
    for value in values:answer=A.add(answer,value)
    return answer


def _dot(A,x,y):return _sum(A,[A.multiply(a,b) for a,b in zip(x,y)])


def _mv(A,matrix,vector):return [_dot(A,row,vector) for row in matrix]


def _norm_upper(A,squared):
    if squared[1]<0:raise ArithmeticError('a PSD norm square has negative upper endpoint')
    if squared[1]==0:return Decimal(0)
    return A.sqrt((squared[1],squared[1]))[1]


def _feature_endpoint_data(A,gram):
    a=Fraction(gram['D'])/2;p=gram['p'];z=a*a
    if not p>z:raise ValueError('endpoint derivative tail requires features>D^2/4')
    normal=A.sqrt(gram['J0']);exponential=A.exp(_exact(A,-z/2))
    values=[];derivatives=[]
    for k in range(0,p,2):
        f=A.divide(A.multiply(A.multiply(normal,exponential),_exact(A,a**k)),
                   A.sqrt(_exact(A,factorial(k))))
        values.append(f);derivatives.append(A.multiply(_exact(A,Fraction(k)/a-a),f))
    q=z/Fraction(p+1)
    base=A.divide(A.multiply(A.multiply(gram['J0'],A.exp(_exact(A,-z))),_exact(A,a**(2*p))),
                  _exact(A,factorial(p)))
    diagonal=A.divide(base,_exact(A,1-q))[1]
    # For k=p+j, f'_k(a)=(p/a-a+j/a) f_k(a); subsequent feature
    # squares are at most base*q^j. Sum j^0,j^1,j^2 exactly.
    alpha=Fraction(p)/a-a
    series=(alpha*alpha/(1-q)+2*alpha/a*q/(1-q)**2
            +q*(1+q)/(a*a*(1-q)**3))
    derivative_diagonal=A.multiply(base,_exact(A,series))[1]
    return values,derivatives,diagonal,derivative_diagonal


def signed_second_certificate(D_string,p,trial,precision=60,V_string='2'):
    """Return (flat strict record, full replay audit) for envelope d_D.

    trial={'spectral': positive Gaussian rank trial, 'a': exact positive
    Rayleigh centre string, 'resolvent_coefficients': signed exact strings}.
    The resolvent trial is arbitrary and need not solve any equation.
    """
    if not isinstance(precision,int) or isinstance(precision,bool):raise ValueError('integer precision required')
    if not isinstance(trial['a'],str) or any(not isinstance(x,str) for x in trial['resolvent_coefficients']):
        raise TypeError('the centre and resolvent trial must be exact decimal strings')
    A=_Intervals(precision)
    spectral,rankaudit=gaussian_rank_certificate(D_string,p,trial['spectral'],precision,None,V_string)
    gram=gaussian_gram_enclosure(D_string,p,precision,None,V_string)
    G=gram['blocks'][0];n=len(G);D,V=Fraction(D_string),Fraction(V_string)
    if not D>V>0:raise ValueError('requires D>V>0')
    theta=V/D;delta=D-V;kappa=V*D*(1-theta)*(1-2*theta)/12
    qraw=[Decimal(x) for x in trial['spectral']['even_q']]
    zraw=[Decimal(x) for x in trial['resolvent_coefficients']]
    a=Decimal(trial['a'])
    if len(zraw)!=n or any(not x.is_finite() for x in zraw) or not a.is_finite() or a<=0:
        raise ValueError('one finite signed exact resolvent coefficient per even feature required')
    q=[A.point(x) for x in qraw];z=[A.point(x) for x in zraw]
    Gq=_mv(A,G,q);Gz=_mv(A,G,z)
    normq=A.sqrt(_dot(A,q,Gq))
    phi=[A.divide(x,normq) for x in q]
    rho=(Decimal(spectral['rho_lower']),Decimal(spectral['rho_upper']))
    secondU=Decimal(spectral['second_upper'])
    separation=A.down.subtract(a,secondU)
    gap=min(Decimal(1),A.down.subtract(rho[0],secondU))
    if separation<=0 or gap<=0:raise ArithmeticError('principal and complement must be strictly separated')
    eps=Decimal(spectral['feature_tail_upper'])
    ce=[A.divide(A.subtract(yi,A.multiply(A.point(a),qi)),normq) for yi,qi in zip(Gq,q)]
    finite_res=_norm_upper(A,_dot(A,ce,_mv(A,G,ce)))
    continuous_res=A.up.add(finite_res,eps)
    direction=A.up.divide(A.up.multiply(A.sqrt(A.point(2))[1],continuous_res),separation)
    if direction>=Decimal('.5'):raise ArithmeticError('eigenfunction direction not resolved')
    fb,dfb,diag,ddiag=_feature_endpoint_data(A,gram)
    point_tail=_norm_upper(A,A.multiply(A.point(eps),A.point(diag)))
    derivative_tail=_norm_upper(A,A.multiply(A.point(eps),A.point(ddiag)))
    sqrtD=A.sqrt(_exact(A,D));J0=gram['J0']
    Jnorm=A.multiply(sqrtD,J0)[1]
    JprimeNorm=A.multiply(A.multiply(sqrtD,J0),A.exp(A.point('-.5')))[1]
    Kphi=[A.divide(x,normq) for x in Gq]
    kb=_dot(A,fb,Kphi);dkb=_dot(A,dfb,Kphi)
    edge_error=A.up.add(point_tail,A.up.multiply(Jnorm,direction))
    derivative_error=A.up.add(derivative_tail,A.up.multiply(JprimeNorm,direction))
    ub=A.divide((A.down.subtract(kb[0],edge_error),A.up.add(kb[1],edge_error)),rho)
    dub=A.divide((A.down.subtract(dkb[0],derivative_error),A.up.add(dkb[1],derivative_error)),rho)
    if ub[0]<=0:raise ArithmeticError('positive endpoint not resolved')
    h=[A.multiply(A.point(2),x) for x in fb]
    normt2=_dot(A,z,Gz);normt=_norm_upper(A,normt2)
    beta0=_dot(A,phi,Gz)
    hp_t=_dot(A,h,Gz)
    Kp_t_t=_sum(A,[_square(A,x) for x in Gz])
    Bp=A.subtract(A.add(A.multiply(A.point(2),hp_t),Kp_t_t),
                  A.add(A.multiply(A.point(a),normt2),_square(A,beta0)))
    residual_coeff=[A.subtract(A.add(A.subtract(hi,A.multiply(A.point(a),zi)),gzi),A.multiply(pi,beta0))
                    for hi,zi,gzi,pi in zip(h,z,Gz,phi)]
    residual_finite=_norm_upper(A,_dot(A,residual_coeff,_mv(A,G,residual_coeff)))
    # abs(Decimal) can use the ambient context: use exact copy_abs.
    delta_rho=max(A.down.subtract(rho[0],a).copy_abs(),A.up.subtract(rho[1],a).copy_abs())
    delta_h=A.up.multiply(Decimal(2),point_tail)
    projection_error=A.up.multiply(Decimal(2),direction)
    err=A.up.add(A.up.multiply(A.up.multiply(Decimal(2),delta_h),normt),
        A.up.multiply(A.up.add(delta_rho,projection_error),normt2[1]))
    tail_energy=A.up.multiply(eps,normt2[1])
    residual=A.up.add(residual_finite,A.up.add(delta_h,
        A.up.multiply(A.up.add(A.up.add(delta_rho,eps),projection_error),normt)))
    quadratic=A.up.divide(A.up.multiply(residual,residual),gap)
    S=(max(Decimal(0),A.down.subtract(Bp[0],err)),
       A.up.add(A.up.add(A.up.add(Bp[1],err),tail_energy),quadratic))
    H=A.multiply(A.point(2),A.multiply(rho,ub));H2=_square(A,H)
    energy=(max(Decimal(0),A.down.subtract(S[0],H2[1])),A.up.subtract(S[1],H2[0]))
    if energy[1]<energy[0]:raise ArithmeticError('reduced energy interval is inconsistent')
    ub2=_square(A,ub)
    C=A.multiply(_exact(A,theta*delta),A.multiply(rho,A.subtract(ub2,_exact(A,1/D))))
    direct=A.add(A.multiply(_exact(A,theta),C),A.multiply(_exact(A,4*kappa),A.multiply(rho,A.multiply(ub,dub))))
    bracket=A.add(A.subtract(A.add(J0,A.multiply(J0,A.exp(_exact(A,-D*D/2)))),
                              A.multiply(A.point(2),A.multiply(rho,ub2))),
                  A.divide(energy,A.point(2)))
    coupling=A.multiply(_exact(A,theta*delta*delta/2),A.multiply(ub2,bracket))
    signed=A.negate(A.add(direct,coupling))
    c=A.negate(C)
    qhash=sha256(('\n'.join(trial['spectral']['even_q'])+'\n--resolvent--\n'+
        '\n'.join(trial['resolvent_coefficients'])+'\n--centre--\n'+trial['a']).encode('ascii')).hexdigest()
    summary=dict(D=D_string,V=V_string,features=p,precision=precision,
        rho_lower=str(rho[0]),rho_upper=str(rho[1]),
        second_upper=str(secondU),unscaled_gap_lower=str(gap),
        feature_tail_upper=str(eps),finite_eigenfunction_residual_upper=str(finite_res),
        continuous_eigenfunction_residual_upper=str(continuous_res),direction_error_upper=str(direction),
        endpoint_tail_upper=str(point_tail),endpoint_derivative_tail_upper=str(derivative_tail),
        resolvent_trial_norm_upper=str(normt),finite_resolvent_residual_upper=str(residual_finite),
        continuous_deflated_residual_upper=str(residual),energy_quadratic_error_upper=str(quadratic),
        endpoint_error_upper=str(edge_error),endpoint_derivative_error_upper=str(derivative_error),
        d_lower=str(signed[0]),d_upper=str(signed[1]),d_width_upper=str(A.up.subtract(signed[1],signed[0])),
        d_sign_certified_positive=signed[0]>0,signed_second_coefficient_certified=True,
        finite_ratio_localization=False,q_sha256=qhash,
        method='Gaussian feature residual and direction; endpoint derivative tails; SPD deflated resolvent energy')
    for name,interval in [('u_edge',ub),('u_edge_derivative',dub),('energy',energy),('c',c),
                          ('direct_lambda2',direct),('coupling_lambda2',coupling),('deflated_energy',S)]:
        summary[name+'_lower']=str(interval[0]);summary[name+'_upper']=str(interval[1])
    serial=lambda vec:[[str(a),str(b)] for a,b in vec]
    audit=dict(summary=summary,trial=trial,spectral=rankaudit,
        eigenfunction_norm=serial([normq]),endpoint_features=serial(fb),endpoint_feature_derivatives=serial(dfb),
        normalized_eigenfunction_coefficients=serial(phi),finite_eigenfunction_residual_coefficients=serial(ce),
        finite_deflated_residual_coefficients=serial(residual_coeff),
        scalars=dict(finite_trial_energy=list(map(str,Bp)),energy_linear_error_upper=str(err),
                     positive_kernel_energy_tail_upper=str(tail_energy),rho_centre_error_upper=str(delta_rho),
                     boundary_h_tail_upper=str(delta_h),projection_operator_error_upper=str(projection_error)))
    return summary,audit


def signed_second_self_check(D_string,p,trial,precision=60,V_string='2'):
    first,_=signed_second_certificate(D_string,p,trial,precision,V_string)
    finer,_=signed_second_certificate(D_string,p,trial,precision+20,V_string)
    for prefix in ('d','c','u_edge','u_edge_derivative','energy','deflated_energy'):
        assert Decimal(first[prefix+'_lower'])<=Decimal(finer[prefix+'_lower'])<=Decimal(finer[prefix+'_upper'])<=Decimal(first[prefix+'_upper']),prefix
    with localcontext() as alien:
        alien.prec=2;alien.rounding=ROUND_CEILING
        assert signed_second_certificate(D_string,p,trial,precision,V_string)[0]==first
    for bad in (dict(trial,a='0'),dict(trial,resolvent_coefficients=trial['resolvent_coefficients'][:-1]),
                dict(trial,resolvent_coefficients=['Infinity']*len(trial['resolvent_coefficients']))):
        try:signed_second_certificate(D_string,p,bad,precision,V_string)
        except (ValueError,ArithmeticError):pass
        else:raise AssertionError('invalid signed-second trial accepted')
    return dict(D=D_string,V=V_string,features=p,precision=precision,
        precision_nesting=True,ambient_context_invariance=True,
        invalid_trials_rejected=True,
        signed_second_coefficient_certified=True,positive_sign_certified=first['d_sign_certified_positive'],
        finite_ratio_localization=False)


def derived_second_centers(record,dw='1',reference_ratio='12'):
    """Propagate certified signed coefficients into asymptotic expressions.

    The resulting count-centre intervals enclose a formula only: no finite-r
    localisation follows without a useful, separately certified remainder.
    All arithmetic before the final directed conversions is exact Fraction.
    """
    def rational(value,name):
        if isinstance(value,(bool,float)):raise TypeError(name+' must be exact')
        try:return Fraction(value)
        except (ValueError,TypeError,ZeroDivisionError,OverflowError) as error:
            raise ValueError(name+' must be finite and rational') from error
    D,V=(rational(record[k],k) for k in ('D','V'))
    cL,cU=(rational(record[k],k) for k in ('c_lower','c_upper'))
    dL,dU=(rational(record[k],k) for k in ('d_lower','d_upper'))
    diffusivity=rational(dw,'dw');ratio=rational(reference_ratio,'reference_ratio')
    if not (D>V>0 and 0<cL<=cU and 0<dL<=dU and diffusivity>0 and ratio>0):
        raise ValueError('positive geometry, diffusivity, ratio and signed coefficients required')
    precision=record['precision']
    if isinstance(precision,str):precision=int(precision)
    if not isinstance(precision,int) or isinstance(precision,bool):raise ValueError('integer precision required')
    A=_Intervals(precision);b=V*V/(8*diffusivity)
    if b*ratio<=dU:raise ValueError('the second-order count centre must be positive throughout the inputs')
    intervals=dict(offset=(2*dL/cU,2*dU/cL),
        leading_center=(2*b*ratio/cU,2*b*ratio/cL),
        second_center=(2*(b*ratio-dU)/cU,2*(b*ratio-dL)/cL),
        second_value_coefficient=(cL*cL*dL/(4*b*b),cU*cU*dU/(4*b*b)))
    result=dict(D=str(record['D']),V=str(record['V']),features=int(record['features']),precision=precision,
        dw=str(dw),reference_ratio=str(reference_ratio),b_exact=str(b),
        c_lower=str(record['c_lower']),c_upper=str(record['c_upper']),
        d_lower=str(record['d_lower']),d_upper=str(record['d_upper']),
        finite_ratio_localization=False,signed_second_coefficient_certified=True,
        method='exact rational propagation of certified signed coefficients into asymptotic expressions only')
    for name,(lower,upper) in intervals.items():
        converted=A.fraction(lower,upper)
        result[name+'_lower']=str(converted[0]);result[name+'_upper']=str(converted[1])
    return result


def derived_second_centers_self_check(record,dw='1',reference_ratio='12'):
    first=derived_second_centers(record,dw,reference_ratio)
    refined=dict(record,precision=first['precision']+20)
    finer=derived_second_centers(refined,dw,reference_ratio)
    for prefix in ('offset','leading_center','second_center','second_value_coefficient'):
        assert Fraction(first[prefix+'_lower'])<=Fraction(finer[prefix+'_lower'])
        assert Fraction(finer[prefix+'_upper'])<=Fraction(first[prefix+'_upper'])
    with localcontext() as alien:
        alien.prec=2;alien.rounding=ROUND_CEILING
        assert derived_second_centers(record,dw,reference_ratio)==first
    for bad in (dict(record,c_lower='0'),dict(record,d_lower='0')):
        try:derived_second_centers(bad,dw,reference_ratio)
        except ValueError:pass
        else:raise AssertionError('invalid derived-centre input accepted')
    return dict(D=first['D'],V=first['V'],features=first['features'],precision=first['precision'],
        exact_rational_propagation=True,precision_nesting=True,ambient_context_invariance=True,
        finite_ratio_localization=False,signed_second_coefficient_certified=True)
