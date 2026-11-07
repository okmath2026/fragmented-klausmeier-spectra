"""Optional NumPy trial preparation; inclusion never assumes its accuracy."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np
from decimal import Context,Decimal,ROUND_HALF_EVEN
from validated_gaussian_rank import gaussian_gram_enclosure


def _polish_positive_trial(matrix,q,precision,digits):
    """Near-rounded preparation only; the saved vector is checked by CW."""
    near=Context(prec=precision,rounding=ROUND_HALF_EVEN)
    midpoint=[[near.divide(near.add(x,y),Decimal(2)) for x,y in row] for row in matrix]
    q=[Decimal(x) for x in q];target=Decimal('1e-'+str(digits))
    for iteration in range(1,10001):
        y=[]
        for row in midpoint:
            value=Decimal(0)
            for entry,qi in zip(row,q):
                value=near.add(value,near.multiply(entry,qi))
            y.append(value)
        if any(not value.is_finite() or value<=0 for value in y):
            raise ArithmeticError('positive high-precision trial lost positivity')
        ratios=[near.divide(yi,qi) for yi,qi in zip(y,q)]
        width=near.subtract(max(ratios),min(ratios))
        if width<=target:
            return list(map(str,q)),dict(polish_digits=digits,polish_iterations=iteration,
                polish_cw_width_decimal=str(width))
        scale=max(y);q=[near.divide(value,scale) for value in y]
    raise ArithmeticError('high-precision positive Gram trial did not converge')


def prepare_rank_trial(D_string,p,m=None,V_string='2',precision=80,polish_digits=0):
    if (not isinstance(polish_digits,int) or isinstance(polish_digits,bool)
            or not 0<=polish_digits<=precision-20):
        raise ValueError('polish_digits must be an integer between zero and precision-20')
    gram=gaussian_gram_enclosure(D_string,p,precision,m,V_string)
    blocks=[np.array([[(float(x)+float(y))/2 for x,y in row] for row in block]) for block in gram['blocks']]
    vectors=[];diagnostics=[]
    for matrix,interval_matrix in zip(blocks,gram['blocks']):
        q=np.ones(len(matrix))/np.sqrt(len(matrix))
        for iteration in range(1,10001):
            y=matrix@q
            width=float(np.ptp(y/q))
            if width<8e-15:break
            q=y/np.linalg.norm(y)
        else:raise ArithmeticError('finite positive Gram trial did not converge')
        if not np.all(q>0):raise ArithmeticError('strict positive trial required')
        strings=[str(float(x)) for x in q]
        detail=dict(cw_width_float=width,iterations=iteration,polish_digits=0)
        if polish_digits:
            strings,polishing=_polish_positive_trial(interval_matrix,strings,precision,polish_digits)
            detail.update(polishing)
        vectors.append(strings);diagnostics.append(detail)
    even_eigs=np.linalg.eigvalsh(blocks[0]);odd_eigs=np.linalg.eigvalsh(blocks[1])
    shift=float((even_eigs[-2]+odd_eigs[-1])/2)
    if not shift>0:raise ArithmeticError('positive parity-ordering shift required')
    return dict(even_q=vectors[0],odd_q=vectors[1],ordering_shift=str(shift)),dict(
        even_principal_float=float(even_eigs[-1]),odd_principal_float=float(odd_eigs[-1]),
        even_second_float=float(even_eigs[-2]),even_trial=diagnostics[0],odd_trial=diagnostics[1])
