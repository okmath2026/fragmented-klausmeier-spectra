"""Near-Decimal preparation only for strict signed second coefficients.

The saved spectral and signed linear-system vectors are arbitrary exact
decimal proof inputs. Inclusion never assumes their preparation accuracy.
"""
from decimal import Context,Decimal,ROUND_HALF_EVEN
from validated_gaussian import _Intervals
from validated_gaussian_rank import gaussian_gram_enclosure,gaussian_rank_certificate
from gaussian_rank_trial import prepare_rank_trial
from validated_signed_second import _feature_endpoint_data,signed_second_certificate


def _solve_near(matrix,rhs,near):
    n=len(rhs);a=[list(row)+[value] for row,value in zip(matrix,rhs)]
    for k in range(n):
        pivot=max(range(k,n),key=lambda i:a[i][k].copy_abs())
        if not a[pivot][k]:raise ArithmeticError('singular near-preparation system')
        a[k],a[pivot]=a[pivot],a[k]
        for i in range(k+1,n):
            factor=near.divide(a[i][k],a[k][k]);a[i][k]=Decimal(0)
            for j in range(k+1,n+1):a[i][j]=near.subtract(a[i][j],near.multiply(factor,a[k][j]))
    result=[Decimal(0)]*n
    for i in range(n-1,-1,-1):
        value=a[i][n]
        for j in range(i+1,n):value=near.subtract(value,near.multiply(a[i][j],result[j]))
        result[i]=near.divide(value,a[i][i])
    return result


def prepare_signed_second_trial(D_string,p,precision=60,V_string='2',polish_digits=40):
    if precision<60:raise ValueError('trial preparation uses precision>=60')
    spectral,diagnostics=prepare_rank_trial(D_string,p,None,V_string,precision,polish_digits)
    record,_=gaussian_rank_certificate(D_string,p,spectral,precision,None,V_string)
    gram=gaussian_gram_enclosure(D_string,p,precision,None,V_string)
    near=Context(prec=precision,rounding=ROUND_HALF_EVEN)
    midpoint=lambda interval:near.divide(near.add(interval[0],interval[1]),Decimal(2))
    G=[[midpoint(value) for value in row] for row in gram['blocks'][0]]
    q=[Decimal(value) for value in spectral['even_q']]
    total=lambda values:sum_near(values,near)
    Gq=[total([near.multiply(x,qi) for x,qi in zip(row,q)]) for row in G]
    norm2=total([near.multiply(x,y) for x,y in zip(q,Gq)])
    a=near.divide(near.add(Decimal(record['finite_even_cw_lower']),Decimal(record['finite_even_cw_upper'])),Decimal(2))
    fb,_,_,_=_feature_endpoint_data(_Intervals(precision),gram)
    rhs=[near.multiply(Decimal(2),midpoint(value)) for value in fb]
    matrix=[]
    for i,row in enumerate(G):
        result=[]
        for j,value in enumerate(row):
            entry=near.subtract(a if i==j else Decimal(0),value)
            entry=near.add(entry,near.divide(near.multiply(q[i],Gq[j]),norm2))
            result.append(entry)
        matrix.append(result)
    coefficients=_solve_near(matrix,rhs,near)
    residual=[]
    for row,yi in zip(matrix,rhs):
        residual.append(near.subtract(total([near.multiply(x,y) for x,y in zip(row,coefficients)]),yi).copy_abs())
    trial=dict(spectral=spectral,a=str(a),resolvent_coefficients=list(map(str,coefficients)))
    diagnostics.update(near_linear_residual=str(max(residual)),linear_trial_preparation='near-Decimal partial-pivot Gaussian elimination; accuracy not assumed')
    return trial,diagnostics


def sum_near(values,near):
    result=Decimal(0)
    for value in values:result=near.add(result,value)
    return result


def signed_second_case(D_string,p,precision=60,V_string='2',polish_digits=40):
    trial,diagnostics=prepare_signed_second_trial(D_string,p,precision,V_string,polish_digits)
    summary,audit=signed_second_certificate(D_string,p,trial,precision,V_string)
    return summary,audit,trial,diagnostics
