"""Standard-library preparation of positive even-feature trial vectors.

Near-rounded iteration affects tightness only. Every saved positive decimal
is subsequently checked in directed Collatz inequalities. principal_case
builds rigorous moments once, rather than repeating them during preparation.
"""
from decimal import Context,Decimal,ROUND_HALF_EVEN

from validated_fragment_principal import _certificate_from_gram,_aggregate_gram_enclosure


def _positive_even_trial(matrix,precision,polish_digits):
    if (not isinstance(polish_digits,int) or isinstance(polish_digits,bool)
            or not 0<polish_digits<=precision-15):
        raise ValueError('polish_digits must lie between one and precision-15')
    near=Context(prec=precision,rounding=ROUND_HALF_EVEN)
    midpoint=[[near.divide(near.add(a,b),Decimal(2)) for a,b in row] for row in matrix]
    q=[Decimal(1) for _ in matrix]
    target=Decimal('1e-'+str(polish_digits))
    for iteration in range(1,10001):
        y=[]
        for row in midpoint:
            value=Decimal(0)
            for entry,qi in zip(row,q):
                value=near.add(value,near.multiply(entry,qi))
            y.append(value)
        if any(not v.is_finite() or v<=0 for v in y):
            raise ArithmeticError('trial preparation lost finite strict positivity')
        ratios=[near.divide(yi,qi) for yi,qi in zip(y,q)]
        width=near.subtract(max(ratios),min(ratios))
        if width<=target:
            return list(map(str,q)),dict(trial_iterations=iteration,
                trial_precision=precision,trial_target_digits=polish_digits,
                trial_near_cw_width=str(width),
                trial_preparation='positive near-Decimal even-Gram power iteration; accuracy not assumed')
        scale=max(y)
        q=[near.divide(yi,scale) for yi in y]
    raise ArithmeticError('positive even-Gram trial did not resolve its preparation target')


def prepare_fragment_principal_trial(D_string,m,p,precision=60,
                                     polish_digits=35,V_string='2'):
    """Return (q strings, preparation diagnostics), using no NumPy/SciPy."""
    gram=_aggregate_gram_enclosure(D_string,p,precision,m,V_string)
    return _positive_even_trial(gram['blocks'][0],precision,polish_digits)


def principal_case(D_string,m,p,precision=60,polish_digits=35,V_string='2'):
    """Build moments once; return (certificate,audit,q,trial diagnostics)."""
    gram=_aggregate_gram_enclosure(D_string,p,precision,m,V_string)
    q,diagnostic=_positive_even_trial(gram['blocks'][0],precision,polish_digits)
    record,audit=_certificate_from_gram(gram,q)
    return record,audit,q,diagnostic
