"""Standard-library replay of signed second-coefficient certificates.

Rebuilds continuous moment, spectral, direction, derivative and deflated
energy records from exact saved trials. A separate Fraction implementation
checks the finite trial energy and all derived centre expressions.
"""
import csv,json
from fractions import Fraction as F
from pathlib import Path
from validated_signed_second import (signed_second_certificate,signed_second_self_check,
    derived_second_centers,derived_second_centers_self_check)

if not __debug__:raise RuntimeError('Run proof replay without -O.')
ROOT=Path(__file__).resolve().parents[1]


def _rows(name):
    with (ROOT/'data'/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _json(name):return json.loads((ROOT/'data'/(name+'.json')).read_text(encoding='utf-8'))


def _compare(actual,saved,label):
    for key,value in actual.items():
        assert key in saved,(label,'missing field',key)
        assert str(value)==saved[key],(label,key,value,saved[key])


def _interval(lo,up=None):
    if up is None:
        if isinstance(lo,(list,tuple)):return F(lo[0]),F(lo[1])
        return F(lo),F(lo)
    return F(lo),F(up)


def _add(x,y):return x[0]+y[0],x[1]+y[1]
def _neg(x):return -x[1],-x[0]
def _sub(x,y):return _add(x,_neg(y))
def _mul(x,y):
    values=[a*b for a in x for b in y]
    return min(values),max(values)
def _square(x):
    return (F(0) if x[0]<=0<=x[1] else min(t*t for t in x),max(t*t for t in x))
def _sum(values):
    result=(F(0),F(0))
    for value in values:result=_add(result,value)
    return result
def _dot(x,y):return _sum([_mul(a,b) for a,b in zip(x,y)])
def _mv(matrix,x):return [_dot(row,x) for row in matrix]
def _contained(outer,inner,label):
    assert outer[0]<=inner[0]<=inner[1]<=outer[1],(label,outer,inner)


def _energy_checks(row,audit):
    """Independent exact arithmetic from stored Gram/endpoint primitives."""
    G=[[_interval(value) for value in rr] for rr in audit['spectral']['even_gram']]
    trial=audit['trial'];z=[_interval(x) for x in trial['resolvent_coefficients']]
    phi=[_interval(x) for x in audit['normalized_eigenfunction_coefficients']]
    h=[_mul(_interval(2),_interval(x)) for x in audit['endpoint_features']]
    Gz=_mv(G,z);norm2=_dot(z,Gz);beta0=_dot(phi,Gz)
    finite=_sub(_add(_mul(_interval(2),_dot(h,Gz)),_sum([_square(x) for x in Gz])),
        _add(_mul(_interval(trial['a']),norm2),_square(beta0)))
    scalars=audit['scalars']
    B=_interval(scalars['finite_trial_energy'])
    _contained(B,finite,'finite variational energy from exact Gram operations')
    err=F(scalars['energy_linear_error_upper'])
    tail=F(scalars['positive_kernel_energy_tail_upper'])
    residual=F(row['continuous_deflated_residual_upper'])
    gap=F(row['unscaled_gap_lower']);quadratic=F(row['energy_quadratic_error_upper'])
    assert 0<gap<=1 and quadratic>=residual*residual/gap
    S=_interval(row['deflated_energy_lower'],row['deflated_energy_upper'])
    _contained(S,(max(F(0),B[0]-err),B[1]+err+tail+quadratic),'deflated resolvent energy')
    rho=_interval(row['rho_lower'],row['rho_upper'])
    ub=_interval(row['u_edge_lower'],row['u_edge_upper'])
    H2=_square(_mul(_interval(2),_mul(rho,ub)))
    E=_interval(row['energy_lower'],row['energy_upper'])
    _contained(E,(max(F(0),S[0]-H2[1]),S[1]-H2[0]),'orthogonal resolvent energy')
    assert E[0]>=0 and E[1]>0
    D,V=F(row['D']),F(row['V']);theta=V/D;delta=D-V
    C=_mul(_interval(theta*delta),_mul(rho,_sub(_square(ub),_interval(1/D))))
    _contained(_interval(row['c_lower'],row['c_upper']),_neg(C),'first boundary coefficient')
    direct=_interval(row['direct_lambda2_lower'],row['direct_lambda2_upper'])
    coupling=_interval(row['coupling_lambda2_lower'],row['coupling_lambda2_upper'])
    _contained(_interval(row['d_lower'],row['d_upper']),_neg(_add(direct,coupling)),
        'signed second coefficient propagation')


def _center_checks(row,source):
    cL,cU=(F(source[k]) for k in ('c_lower','c_upper'))
    dL,dU=(F(source[k]) for k in ('d_lower','d_upper'))
    V,dw,r=F(row['V']),F(row['dw']),F(row['reference_ratio'])
    b=V*V/(8*dw);assert b==F(row['b_exact']) and b*r>dU>0
    expected=dict(offset=(2*dL/cU,2*dU/cL),
        leading_center=(2*b*r/cU,2*b*r/cL),
        second_center=(2*(b*r-dU)/cU,2*(b*r-dL)/cL),
        second_value_coefficient=(cL*cL*dL/(4*b*b),cU*cU*dU/(4*b*b)))
    for name,interval in expected.items():
        _contained(_interval(row[name+'_lower'],row[name+'_upper']),interval,name)
    assert row['finite_ratio_localization']=='False'
    assert row['signed_second_coefficient_certified']=='True'


def main():
    saved=_rows('signed_second');vectors=_json('signed_second_vectors')
    audits=_json('signed_second_audits');checks=_json('signed_second_checks')
    expected={('4',p) for p in (20,30,40)}|{('6',p) for p in (30,40,50)}
    assert len(saved)==len(vectors)==len(audits)==6
    key=lambda row:(row['D'],int(row['features']))
    assert {key(row) for row in saved}=={key(row) for row in vectors}==expected
    lookup={key(row):row for row in saved};audit_lookup={key(a['summary']):a for a in audits}
    boundaries=_rows('boundary_certificates');self_checks=[]
    for vector in vectors:
        D,p=key(vector);precision=int(vector['precision']);V=vector['V'];trial=vector['trial']
        assert V=='2' and precision==60
        record,audit=signed_second_certificate(D,p,trial,precision,V)
        row=lookup[(D,p)]
        _compare(record,row,('signed second',D,p))
        assert audit==audit_lookup[(D,p)],('complete signed-second audit',D,p)
        assert F(row['d_lower'])>0 and row['d_sign_certified_positive']=='True'
        assert row['signed_second_coefficient_certified']=='True' and row['finite_ratio_localization']=='False'
        assert F(row['u_edge_lower'])>0 and F(row['u_edge_derivative_upper'])<0
        assert F(row['d_width_upper'])>=F(row['d_upper'])-F(row['d_lower'])>0
        boundary=next(b for b in boundaries if b['D']==D and int(b['N'])==640)
        assert F(boundary['c_lower'])<=F(row['c_lower'])<=F(row['c_upper'])<=F(boundary['c_upper'])
        _energy_checks(row,audit)
        self_checks.append(signed_second_self_check(D,p,trial,precision,V))
    assert checks['self_checks']==self_checks
    assert checks['signed_second_coefficient_certified'] and not checks['finite_ratio_localization']
    for D in ('4','6'):
        rr=sorted((r for r in saved if r['D']==D),key=lambda r:int(r['features']))
        assert all(F(b['d_width_upper'])<F(a['d_width_upper']) for a,b in zip(rr,rr[1:]))
    centers=_rows('signed_second_centers');assert len(centers)==2 and {r['D'] for r in centers}=={'4','6'}
    center_checks=[]
    for row in centers:
        D=row['D'];p=40 if D=='4' else 50
        assert int(row['features'])==p and row['dw']=='1' and row['reference_ratio']=='12'
        source=lookup[(D,p)]
        _compare(derived_second_centers(source,row['dw'],row['reference_ratio']),row,('derived centre',D))
        _center_checks(row,source)
        center_checks.append(derived_second_centers_self_check(source,row['dw'],row['reference_ratio']))
    if 'center_checks' in checks:assert checks['center_checks']==center_checks
    print(json.dumps(dict(signed_second_rows=6,complete_feature_coverage=True,
        full_saved_audits_reproduced=True,independent_fraction_energy_checks=True,
        boundary_first_coefficient_containment=True,positive_d_and_negative_endpoint_derivative=True,
        signed_second_coefficient_certified=True,derived_center_rows=2,
        finite_ratio_localization=False,self_checks=self_checks,center_checks=center_checks,
        dependencies='Python standard library only; exact saved trials'),indent=2))


if __name__=='__main__':main()
