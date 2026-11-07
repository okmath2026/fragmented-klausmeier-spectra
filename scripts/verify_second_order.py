"""Standard-library replay of exact weak measures and cubic norm budgets.

The floating second spectral-coefficient diagnostics are intentionally
excluded from rigorous numeric certification by this verifier.
"""
import argparse,csv,json
from fractions import Fraction
from pathlib import Path
from validated_fragment_measure import measure_moment_certificate,fragment_measure_self_check


if not __debug__:
    raise RuntimeError('Run validated certificates without -O.')


def main(self_check_only=False):
    checks=fragment_measure_self_check()
    if self_check_only:
        print(json.dumps(checks,indent=2));return
    root=Path(__file__).resolve().parents[1]
    with (root/'data/weak_measure.csv').open(newline='',encoding='utf-8') as stream:
        rows=list(csv.DictReader(stream))
    expected={(D,V,m,degree) for D,V in [('4','2'),('6','2'),('3.7','2.1'),('5','1')]
              for m in (2,4,8,16,32,64) for degree in (0,2,4,6,8)}
    actual={(row['D'],row['V'],int(row['m']),int(row['degree'])) for row in rows}
    assert len(rows)==len(actual)==len(expected)==120
    assert actual==expected,(actual-expected,expected-actual)
    for row in rows:
        record=measure_moment_certificate(row['D'],row['V'],int(row['m']),int(row['degree']))
        for key,value in record.items():
            assert key in row and str(value)==row[key],(row['D'],row['m'],row['degree'],key,value,row.get(key))
        # Independent propagation of the saved weak-measure polynomial fields.
        m=Fraction(row['m'])
        exact_approximation=(Fraction(row['mu0_exact'])+Fraction(row['mu1_exact'])/m
                             +Fraction(row['mu2_exact'])/m**2)
        exact_residual=Fraction(row['mu_m_exact'])-exact_approximation
        bound=sum((Fraction(row['C'+str(j)+'_exact'])*Fraction(row['norm'+str(j)+'_exact'])
                   for j in range(4)),Fraction(0))/m**3
        assert exact_approximation==Fraction(row['approximation_exact'])
        assert exact_residual==Fraction(row['residual_exact'])
        assert abs(exact_residual)<=bound==Fraction(row['remainder_bound_exact'])
    from validated_second_budget import second_budget_certificate,second_budget_self_check
    def saved_rows(name):
        with (root/'data'/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    envelopes=saved_rows('peak_envelopes');gaps=saved_rows('sharp_gaps')
    budgets=saved_rows('second_budgets')
    assert len(budgets)==2 and {row['D'] for row in budgets}=={'4','6'}
    budget_checks=[]
    for row in budgets:
        D,N=row['D'],int(row['N'])
        assert row['V']=='2' and N==640
        envelope=next(e for e in envelopes if e['D']==D and int(e['N'])==N)
        gap=next(g for g in gaps if g['D']==D and int(g['N'])==N)
        actual=second_budget_certificate(envelope,gap,int(row['precision']))
        for key,value in actual.items():
            assert key in row and str(value)==row[key],('second budget',D,key,value,row.get(key))
        assert row['gamma_is_theta_scaled']=='True'
        assert row['signed_second_coefficient_computed']==row['finite_ratio_localization']=='False'
        assert Fraction(row['K3_upper'])>0 and Fraction(row['d_abs_upper'])>0
        assert int(row['anchor_M'])>=2
        assert Fraction(row['anchor_M'])>=2*Fraction(row['s_upper'])*(Fraction(row['z0_upper'])
            +Fraction(row['q_upper'])*Fraction(row['e0_upper']))
        measure_row=next(m for m in rows if m['D']==D)
        assert all(row['C'+str(j)+'_exact']==measure_row['C'+str(j)+'_exact'] for j in range(4))
        budget_checks.append(second_budget_self_check(envelope,gap,int(row['precision'])))
    study_checks=json.loads((root/'data/second_order_checks.json').read_text(encoding='utf-8'))
    assert study_checks['second_budget_checks']==budget_checks
    assert study_checks['polynomial_cases']==len(rows)
    assert study_checks['all_polynomial_bounds'] and not study_checks['coefficient_intervals_certified']
    print(json.dumps(dict(weak_measure_rows=len(rows),complete_geometry_degree_count_coverage=True,
        full_saved_fraction_fields_reproduced=True,
        strict_scope='weak measure and cubic norm budget; signed second coefficient not certified',
        cubic_budget_rows=len(budgets),full_saved_cubic_budget_fields_reproduced=True,
        signed_second_coefficient_certified=False,finite_ratio_localization=False,
        cubic_budget_self_checks=budget_checks,self_checks=checks,
        dependencies='Python standard library only'),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--self-check',action='store_true',help='run exact independent tests without saved CSV input')
    main(parser.parse_args().self_check)
