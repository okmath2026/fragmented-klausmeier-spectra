"""Standard-library replay of spatial-water, broad-kernel and rank proofs."""
import csv,json
from fractions import Fraction
from pathlib import Path
from decimal import Decimal
from validated_weighted_water import (weighted_water_certificate,
    weighted_scalar_comparison,weighted_water_self_check)
from validated_broad_kernel import broad_kernel_certificate,broad_kernel_self_check
from validated_gaussian_rank import gaussian_rank_certificate,gaussian_rank_self_check
from validated_peaks import fragment_certificate


def main():
    root=Path(__file__).resolve().parents[1]
    def rows(name):
        with (root/'data'/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    def compare(actual,saved,label):
        for key,value in actual.items():
            expected='' if value is None else str(value)
            assert expected==saved[key],(label,key,expected,saved.get(key))
    summary=json.loads((root/'data/summary.json').read_text())
    vectors=json.loads((root/'data/weighted_water_vectors.json').read_text())
    weighted=rows('weighted_water')
    audits=json.loads((root/'data/weighted_water_audits.json').read_text())
    peak_vectors=json.loads((root/'data/peak_vectors.json').read_text())
    peak=next(v for v in peak_vectors if (v['D'],v['m'],v['cells_per_patch'])==('4',2,4))
    direct=fragment_certificate('4',2,4,peak['q'],60)
    assert len(vectors)==len(weighted)==len(audits['cases'])==5
    for vector,row,saved_audit in zip(vectors,weighted,audits['cases']):
        record,audit=weighted_water_certificate(vector['D'],vector['m'],vector['cells_per_patch'],vector['q'],
            dv_string=vector['dv'],dw_string=vector['dw'],precision=vector['precision'],
            V_string=vector['V'],Amax=vector['Amax'],R=vector['R'],B=vector['B'],return_details=True)
        compare(record,row,'weighted continuum majorant')
        compare(weighted_scalar_comparison(record,direct['beta_lower'],direct['beta_upper']),row,'strict scalar comparison')
        assert audit==saved_audit
        assert Fraction(row['scalar_margin_upper'])<0
        if int(row['cells_per_patch'])>=8:assert Fraction(row['cw_upper'])<1
    check=weighted_water_self_check('4',2,8,vectors[2]['q'])
    assert check==audits['self_check']==summary['weighted_water_checks']
    assert Fraction(weighted[2]['cw_upper'])<Fraction('.871655')
    assert Fraction(weighted[2]['scalar_margin_upper'])<Fraction('-.05192687')
    assert Fraction(weighted[2]['kernel_eta_radius_strict'])>Fraction('.03')
    broad=rows('broad_kernel')
    for row in broad:
        actual=broad_kernel_certificate(row['D'],row['V'],row['sigma_anchor'],row['sqrt_variance_upper'])
        compare(actual,row,'all-count parameter-halfline')
        assert Fraction(actual['strict_margin'])>0
    assert broad_kernel_self_check()==summary['broad_kernel_checks']
    rank=rows('gaussian_rank')
    rank_vectors=json.loads((root/'data/gaussian_rank_vectors.json').read_text())
    rank_audits=json.loads((root/'data/gaussian_rank_audits.json').read_text())
    assert len(rank)==len(rank_vectors)==len(rank_audits)==9
    for vector,row,saved_audit in zip(rank_vectors,rank,rank_audits):
        record,audit=gaussian_rank_certificate(vector['D'],vector['p'],vector['trial'],precision=vector['precision'])
        compare(record,row,'finite-rank continuous spectrum')
        assert audit==saved_audit
        assert record['even_inertia_positive']==1
        assert Decimal(record['even_rest_upper'])<Decimal(record['second_lower'])
        assert Decimal(record['second_upper'])<Decimal(record['rho_lower'])
        assert Decimal(record['gap_lower'])>0
    checks=[gaussian_rank_self_check(v['D'],v['p'],v['trial'],precision=v['precision'])
            for v in rank_vectors if (v['D'],v['p']) in [('6',40),('14',128)]]
    assert checks==json.loads((root/'data/gaussian_rank_checks.json').read_text())==summary['gaussian_rank_checks']
    assert Decimal(next(r for r in rank if r['D']=='6' and r['features']=='40')['feature_tail_upper'])<Decimal('7e-14')
    assert Decimal(next(r for r in rank if r['D']=='14' and r['features']=='128')['gap_lower'])>Decimal('.06097')
    for D,p,bound in [('4','40','9.598e-26'),('6','40','1.372e-13'),('14','128','5.421e-20')]:
        row=next(r for r in rank if r['D']==D and r['features']==p)
        assert Decimal(row['gap_width_upper'])<Decimal(bound)
    larger=next(r for r in rank if r['D']=='14' and r['features']=='128')
    assert Decimal('.06097401908685')<=Decimal(larger['gap_lower'])
    assert Decimal(larger['gap_upper'])<=Decimal('.06097401908687')
    earlier_gaps=rows('sharp_gaps')
    for D in ('4','6'):
        previous=next(r for r in earlier_gaps if r['D']==D and r['N']=='640')
        refined=next(r for r in rank if r['D']==D and r['features']=='40')
        assert Decimal(previous['gamma_lower'])<=Decimal(refined['scaled_gap_lower'])
        assert Decimal(refined['scaled_gap_upper'])<=Decimal(previous['gamma_upper'])
    print(json.dumps(dict(weighted_water_rows=len(weighted),strict_spatial_gain=True,
        broad_parameter_rows=len(broad),all_count_coverage='analytic moment inequality',
        gaussian_rank_rows=len(rank),larger_envelope_D14=True,
        precision_and_context_checks=True,full_saved_audits_reproduced=True,cross_method_gap_inclusion=True,
        dependencies='Python standard library only; exact saved proof inputs'),indent=2))


if __name__=='__main__':main()
