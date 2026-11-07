"""Standard-library replay of all anchor-dependent cubic remainder bounds."""
import csv,json
from fractions import Fraction as F
from pathlib import Path
from validated_cubic_anchor import (minimum_cubic_anchor,cubic_anchor_certificate,
    cubic_anchor_self_check,cubic_anchor_search_audit)
from validated_second_budget import second_budget_certificate

if not __debug__:raise RuntimeError('Run proof replay without -O.')
ROOT=Path(__file__).resolve().parents[1]


def rows(name):
    with (ROOT/'data'/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def main():
    source=rows('second_budgets');saved=rows('cubic_anchor')
    envelopes=rows('peak_envelopes');gaps=rows('sharp_gaps')
    assert len(source)==2 and len(saved)==8
    assert {(r['D'],int(r['anchor_multiplier'])) for r in saved}=={
        (D,factor) for D in ('4','6') for factor in (1,2,4,8)}
    checks=[];searches=[]
    for original in source:
        D=original['D'];N=int(original['N'])
        envelope=next(r for r in envelopes if r['D']==D and int(r['N'])==N)
        gap=next(r for r in gaps if r['D']==D and int(r['N'])==N)
        rebuilt=second_budget_certificate(envelope,gap,int(original['precision']))
        assert all(str(value)==original[key] for key,value in rebuilt.items())
        M=minimum_cubic_anchor(original)
        for row in (r for r in saved if r['D']==D):
            factor=int(row['anchor_multiplier']);assert int(row['M'])==M*factor
            actual=cubic_anchor_certificate(original,M*factor,int(row['precision']))
            assert all(str(value)==row[key] for key,value in actual.items())
            # Independent exact upper-budget algebra, using only primitive saved source fields.
            a,b,C,z,s,q,LU=(F(original[key+'_upper']) for key in ('a1','a2','CX','z0','s','q','LU'))
            m=M*factor;e=a+b/m+C/m**2;f=b+C/m;eta=s*(z+q*e)/m
            K=LU*(C+s*q*f*e/(1-eta)+a*s*s*q*e*(z+q*e)/(1-eta)+a*s*q*f)
            assert eta==F(row['eta_upper_exact'])<=F(1,2)
            assert K==F(row['K3_upper_exact'])<=F(row['K3_upper'])
            assert F(row['K3_reduction_lower'])<=F(original['K3_lower'])/K
            assert row['valid_for_every_m_ge_M']=='True'
            assert row['finite_ratio_localization']==row['signed_second_coefficient_computed']=='False'
        checks.append(cubic_anchor_self_check(original))
        searches.append(cubic_anchor_search_audit(original))
    audit=json.loads((ROOT/'data/cubic_anchor_checks.json').read_text())
    assert audit['checks']==checks and audit['anchor_factors']==[1,2,4,8] and audit['experiment_rows']==8
    assert audit['anchor_searches']==searches
    print(json.dumps(dict(rows=8,complete_anchor_coverage=True,
        full_source_budgets_replayed=True,all_exact_fraction_fields_reproduced=True,
        checks=checks,dependencies='Python standard library only'),indent=2))


if __name__=='__main__':main()
