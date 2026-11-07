"""Standard-library replay of every coupled all-integer optimum certificate.

Run from any directory with Python (without -O). Saved positive feature trial
vectors are the only numerical proof inputs: the continuous Gaussian moments,
Collatz endpoints, exact rational water geometry, finite coverage, infinite
tail and open parameter sectors are rebuilt here. No floating eigensolver or
study-driver import enters this replay.
"""
import csv
import json
from decimal import Decimal,localcontext,ROUND_CEILING
from fractions import Fraction
from pathlib import Path

from validated_coupled_optimum import (coupled_margin_endpoints,
    coupled_tail_certificate,global_coupled_certificate,coupled_optimum_self_check,
    asymptotic_scale_certificate)
from validated_fragment_principal import (fragment_principal_certificate,
    fragment_principal_self_check,principal_margin_certificate,
    _aggregate_gram_enclosure)
from validated_gaussian_rank import gaussian_gram_enclosure
from validated_peaks import rayleigh_envelope_certificate
from validated_water import rational_water_certificate


if not __debug__:
    raise RuntimeError('Run validated certificates without -O.')


ROOT=Path(__file__).resolve().parents[1]


def _rows(name):
    with (ROOT/'data'/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _json(name):
    return json.loads((ROOT/'data'/(name+'.json')).read_text(encoding='utf-8'))


def _compare(actual,saved,label):
    """Compare every reconstructed strict field, including decimal spelling."""
    for key,value in actual.items():
        expected='' if value is None else str(value)
        assert key in saved,(label,'missing field',key)
        assert expected==saved[key],(label,key,expected,saved[key])


def _key(record):
    return (str(record['D']),str(record.get('V','2')),int(record['m']),
            int(record['features']),int(record['precision']))


def _assert_exact_margin(row):
    """Check saved margin bounds by independent exact Fraction propagation."""
    d,Q,B=(Fraction(row[name]) for name in ('dv','Q','B'))
    lower=d*Fraction(row['beta_lower'])+B-Q*Fraction(row['q_upper'])
    upper=d*Fraction(row['beta_upper'])+B-Q*Fraction(row['q_lower'])
    assert Fraction(row['margin_lower'])<=lower<=upper<=Fraction(row['margin_upper'])
    direct=coupled_margin_endpoints(row['beta_lower'],row['beta_upper'],
        row['q_lower'],row['q_upper'],row['dv'],row['Q'],row['B'],int(row['precision']))
    assert Fraction(row['margin_lower'])<=Fraction(direct['margin_lower'])
    assert Fraction(direct['margin_upper'])<=Fraction(row['margin_upper'])


def _cross_moment_checks():
    """Compare two independently bounded continuous moment constructions.

    Aggregate exact signed Taylor polynomials and positive endpoint series
    use different remainder arguments. Their intervals must intersect for
    every moment and every even Gram entry. Geometry and analytic feature
    tails must agree exactly. No eigenvector or eigenvalue is used.
    """
    checks=[]
    for D,V,m,p in [('4','2',5,12),('6','2',7,20),
                    ('3.7','2.1',3,12),('4','2',81,31)]:
        aggregate=_aggregate_gram_enclosure(D,p,60,m,V)
        positive=gaussian_gram_enclosure(D,p,60,m,V)
        assert aggregate['half']==positive['half']
        assert aggregate['measure']==positive['measure']==Fraction(V)
        assert aggregate['tail']==positive['tail']
        assert len(aggregate['moments'])==len(positive['moments'])==p
        for first,second in zip(aggregate['moments'],positive['moments']):
            assert 0<max(first[0],second[0])<=min(first[1],second[1])
        for first_row,second_row in zip(aggregate['blocks'][0],positive['blocks'][0]):
            for first,second in zip(first_row,second_row):
                assert 0<max(first[0],second[0])<=min(first[1],second[1])
        checks.append(dict(D=D,V=V,m=m,features=p,precision=60,
            aggregate_and_positive_series_intersect=True,
            exact_geometry_and_tail_match=True))
    return checks


def _scale_checks():
    """Replay the limiting coefficients from the prior certified c_D data.

    This checks limit constants and their leading-order values at r=12; it
    does not interpret them as finite-r bounds for the integer maximiser.
    The originating boundary intervals have their own full certificate
    replay in verify_certificates.py.
    """
    boundaries=_rows('boundary_certificates');saved=_rows('coupled_scale')
    assert len(saved)==2 and {row['D'] for row in saved}=={'4','6'}
    results=[]
    for row in saved:
        D=row['D']
        boundary=next(r for r in boundaries if r['D']==D and int(r['N'])==640)
        actual=asymptotic_scale_certificate(boundary)
        _compare(actual,row,('asymptotic limit coefficients',D))
        assert not actual['finite_ratio_localization']
        V,dw=Fraction(row['V']),Fraction(row['dw'])
        c_lower,c_upper=Fraction(row['c_lower']),Fraction(row['c_upper'])
        b=V*V/(8*dw);ratio=Fraction(row['reference_ratio'])
        assert b==Fraction(row['b_exact']) and 0<c_lower<=c_upper
        exact_pairs=[('m_over_ratio_lower','m_over_ratio_upper',2*b/c_upper,2*b/c_lower),
            ('scaled_gain_limit_lower','scaled_gain_limit_upper',c_lower*c_lower/(4*b),c_upper*c_upper/(4*b)),
            ('leading_m_lower','leading_m_upper',ratio*2*b/c_upper,ratio*2*b/c_lower)]
        finer=asymptotic_scale_certificate(dict(boundary,precision=80))
        for lower,upper,exact_lower,exact_upper in exact_pairs:
            assert Fraction(row[lower])<=exact_lower<=exact_upper<=Fraction(row[upper])
            assert Decimal(row[lower])<=Decimal(finer[lower])
            assert Decimal(finer[upper])<=Decimal(row[upper])
        with localcontext() as alien:
            alien.prec=2;alien.rounding=ROUND_CEILING
            assert asymptotic_scale_certificate(boundary)==actual
        results.append(dict(D=D,exact_fraction_propagation=True,
            precision_nesting=True,ambient_context_invariance=True,
            finite_ratio_localization=False))
    return results


def main():
    vectors=_json('coupled_peak_vectors')
    audits=_json('coupled_peak_audits')
    checks=_json('coupled_peak_checks')
    assert isinstance(vectors,list) and isinstance(audits,list)
    assert len(vectors)==len(audits)
    replay={};trial_by_key={}
    for vector,saved_audit in zip(vectors,audits):
        key=_key(vector)
        assert key not in replay,('duplicate proof vector',key)
        assert vector['precision']==60
        record,audit=fragment_principal_certificate(vector['D'],vector['m'],
            vector['features'],vector['q'],vector['precision'],vector['V'])
        assert audit==saved_audit,('complete principal audit',key)
        assert record['principal_is_even']
        assert not record['uses_quantified_gap']
        assert not record['uses_odd_ordering']
        assert not record['competitor_upper_uses_tail']
        water=rational_water_certificate(Fraction(vector['V'])/vector['m'],
                                        Fraction(1),precision=vector['precision'])
        margin=principal_margin_certificate(record,water)
        combined=dict(**record,**margin,Q='7.2',dw='1',
                      q_lower=water['q_lower'],q_upper=water['q_upper'])
        _assert_exact_margin(combined)
        replay[key]=combined;trial_by_key[key]=vector

    datasets={name:_rows(name) for name in
              ('coupled_candidates','coupled_refinement','coupled_competitors')}
    referenced=set()
    for name,rows in datasets.items():
        seen=set()
        for row in rows:
            key=_key(row)
            assert key in replay,('missing proof vector',name,key)
            assert key not in seen,('duplicate row',name,key)
            seen.add(key);referenced.add(key)
            _compare(replay[key],row,(name,key))
            _assert_exact_margin(row)
    assert referenced==set(replay),('unused/missing proof vectors',referenced^set(replay))
    coarse=datasets['coupled_candidates']
    expected={('4','2',m,20,60) for m in range(2,240)}
    expected|={('6','2',m,20,60) for m in range(2,200)}
    assert len(coarse)==436 and {_key(row) for row in coarse}==expected
    refinement=datasets['coupled_refinement']
    assert len(refinement)==6
    assert {_key(row) for row in refinement}=={
        ('4','2',80,p,60) for p in (20,24,30)}|{
        ('6','2',70,p,60) for p in (20,30,40)}
    assert all((row['D'],int(row['m'])) not in [('4',80),('6',70)]
               for row in datasets['coupled_competitors'])

    # Rebuild the pre-existing standalone principal envelopes, rather than
    # trusting a copied beta-infinity or infinite-tail number in the new data.
    envelope_vectors=_json('peak_envelope_vectors')
    saved_envelopes=_rows('peak_envelopes')
    envelopes={}
    for D in ('4','6'):
        vector=next(v for v in envelope_vectors if v['D']==D and int(v['N'])==640)
        envelope=rayleigh_envelope_certificate(D,vector['N'],vector['q'],
            vector['precision'],vector.get('V','2'))
        saved=next(r for r in saved_envelopes if r['D']==D and int(r['N'])==640)
        _compare(envelope,saved,('standalone principal Rayleigh envelope',D))
        envelopes[D]=envelope

    tails={};saved_tails=_rows('coupled_tails')
    assert len(saved_tails)==2 and {r['D'] for r in saved_tails}=={'4','6'}
    for row in saved_tails:
        D=row['D'];M={'4':240,'6':200}[D]
        tail=coupled_tail_certificate(envelopes[D],M)
        _compare(tail,row,('coupled infinite tail',D))
        assert tail['nonnegative_water_term_dropped']
        assert not tail['uses_quantified_spectral_gap']
        assert not tail['uses_boundary_coefficient_interval']
        tails[D]=tail

    selected={};global_records=[];global_checks=[];principal_checks=[]
    saved_optima=_rows('coupled_optima')
    assert len(saved_optima)==2 and {r['D'] for r in saved_optima}=={'4','6'}
    for D,winner,p in [('4',80,30),('6',70,40)]:
        choices=checks['selection'][D]
        M=tails[D]['anchor_m']
        assert checks['finite_coverage'][D]==[2,M-1]
        assert len(choices)==M-2
        assert {int(choice['m']) for choice in choices}==set(range(2,M))
        finite=[replay[(D,'2',int(choice['m']),int(choice['features']),60)]
                for choice in choices]
        assert next(row for row in finite if row['m']==winner)['features']==p
        selected[D]=finite
        actual=global_coupled_certificate(finite,tails[D],winner)
        _compare(actual,next(row for row in saved_optima if row['D']==D),
                 ('all-integer coupled optimum',D))
        assert actual['unique_global_maximum'] and actual['ratio_sector_strict']
        assert actual['ordering_independent_of_B']
        target_lower,target_upper=map(Fraction,checks['targets'][D])
        assert Fraction(actual['ratio_lower_required'])<target_lower
        assert target_lower<Fraction(actual['nominal_ratio_exact'])<target_upper
        assert target_upper<Fraction(actual['ratio_upper_allowed'])
        assert target_lower==Fraction({'4':'11.95','6':'11.87'}[D])
        assert target_upper==Fraction({'4':'12.09','6':'12.03'}[D])
        assert Fraction(actual['overall_separation_lower'])>0
        assert Fraction(actual['kernel_eta_radius_strict'])>0
        global_records.append(actual)
        global_checks.append(coupled_optimum_self_check(finite,tails[D],winner))
        vector=trial_by_key[(D,'2',winner,p,60)]
        principal_checks.append(fragment_principal_self_check(D,winner,p,vector['q'],60,'2'))
    assert global_checks==checks['global_checks']
    assert principal_checks==checks['principal_checks']

    scale_checks=_scale_checks()
    cross_checks=_cross_moment_checks()
    print(json.dumps(dict(principal_vectors=len(vectors),
        coarse_cases=len(coarse),winner_refinements=len(refinement),
        adaptive_competitors=len(datasets['coupled_competitors']),
        full_saved_audits_reproduced=True,exact_fraction_margin_checks=True,
        finite_coverage={D:[2,tails[D]['anchor_m']-1,len(selected[D])] for D in ('4','6')},
        infinite_tails_rebuilt=True,strict_global_winners={r['D']:r['winner_m'] for r in global_records},
        closed_ratio_targets_strictly_inside=True,
        precision_and_context_checks=True,cross_moment_checks=cross_checks,
        scale_checks=scale_checks,
        dependencies='Python standard library only; exact saved proof inputs'),indent=2))


if __name__=='__main__':
    main()
