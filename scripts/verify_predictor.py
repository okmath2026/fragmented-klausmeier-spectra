"""Standard-library replay of envelope-predictor residual certificates.

Saved decimal predictors are exact trial inputs, never trusted eigensolves.
The signed-coefficient sources are reconstructed from their own saved trials.
"""
import csv
import json
import sys
from decimal import Decimal, localcontext, ROUND_CEILING
from fractions import Fraction as F
from hashlib import sha256
from pathlib import Path

from validated_gaussian import _Intervals
from validated_predictor import (predictor_certificate, second_remainder_certificate,
                                 finite_trace_certificate)
from validated_signed_second import signed_second_certificate

if not __debug__:
    raise RuntimeError('Run proof replay without -O.')

ROOT = Path(__file__).resolve().parents[1]


def _rows(name):
    with (ROOT/'data'/(name+'.csv')).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def _json(name):
    return json.loads((ROOT/'data'/(name+'.json')).read_text(encoding='utf-8-sig'))


def _key(row):
    return (str(row['D']), str(row['V']), int(row['m']), int(row['features']),
            int(row['predictor_order']), int(row['precision']))


def _compare(actual, saved, label):
    for name, value in actual.items():
        assert name in saved, (label, 'missing strict field', name)
        assert str(value) == saved[name], (label, name, value, saved[name])


def _interval(lower, upper=None):
    if upper is None:
        if isinstance(lower, (list, tuple)):
            return F(lower[0]), F(lower[1])
        return F(lower), F(lower)
    return F(lower), F(upper)


def _add(x, y):
    return x[0]+y[0], x[1]+y[1]


def _sub(x, y):
    return x[0]-y[1], x[1]-y[0]


def _mul(x, y):
    products = [a*b for a in x for b in y]
    return min(products), max(products)


def _square(x):
    return (F(0) if x[0] <= 0 <= x[1] else min(a*a for a in x),
            max(a*a for a in x))


def _contained(outer, inner, label):
    assert outer[0] <= inner[0] <= inner[1] <= outer[1], (label, outer, inner)


def _fraction_checks(row, audit, trial):
    """Independent exact checks on the stored directed residual/budget steps."""
    m = int(row['m'])
    corrected = int(row['predictor_order']) == 1
    q = [F(a)+(F(b)/m if corrected else 0)
         for a, b in zip(trial['q0'], trial['q1'])]
    canonical = list(map(str, q))
    assert audit['exact_trial'] == canonical
    assert len(q) == int(row['even_features'])
    assert sha256('\n'.join(canonical).encode('ascii')).hexdigest() == row['q_sha256']
    norm = sum((a*a for a in q), F(0))
    assert norm == F(row['q_norm_squared_exact']) > 0
    residual = [_interval(x) for x in audit['residual_intervals']]
    assert len(residual) == len(q)
    squares = [_square(x) for x in residual]
    squared = (sum((x[0] for x in squares), F(0))/norm,
               sum((x[1] for x in squares), F(0))/norm)
    assert F(row['residual_squared_upper']) >= squared[1]
    r = _interval(row['rayleigh_lower'], row['rayleigh_upper'])
    trace = _interval(audit['trace_interval'])
    assert trace[1] == F(row['finite_trace_upper'])
    separation = F(row['trace_separation_lower'])
    assert 0 < separation <= 2*r[0]-trace[1]
    correction = F(row['quadratic_correction_upper'])
    assert correction >= F(row['residual_squared_upper'])/separation >= 0
    tau = F(row['feature_tail_upper'])
    assert tau >= 0 and F(row['rho_lower']) == r[0] > 0
    assert F(row['rho_upper']) >= min(F(1), r[1]+correction+tau)
    assert F(row['rho_lower']) <= F(row['rho_upper']) <= 1
    beta = _interval(row['beta_lower'], row['beta_upper'])
    _contained(beta, (1-F(row['rho_upper']), 1-F(row['rho_lower'])), 'beta propagation')
    assert F(row['beta_width_upper']) >= beta[1]-beta[0] >= 0
    for name in ('trial_from_envelope_only', 'is_pointwise_certificate'):
        assert row[name] == 'True'
    for name in ('uses_fragment_eigensolve', 'uses_quantified_gap', 'uses_inertia',
                 'uniform_cubic_tail_certified'):
        assert row[name] == 'False'


def _remainder_fraction_checks(remainder, row, source):
    m = int(row['m'])
    theta = F(row['V'])/F(row['D'])
    rho = _interval(source['rho_lower'], source['rho_upper'])
    c = _interval(source['c_lower'], source['c_upper'])
    d = _interval(source['d_lower'], source['d_upper'])
    beta0 = _sub(_interval(1), _mul(_interval(theta), rho))
    expansion = _add(beta0, _add(_mul(_interval(F(1, m)), c),
                                _mul(_interval(F(1, m*m)), d)))
    beta = _interval(row['beta_lower'], row['beta_upper'])
    error = _sub(beta, expansion)
    scaled = _mul(_interval(m**3), error)
    _contained(_interval(remainder['beta_infty_lower'], remainder['beta_infty_upper']),
               beta0, 'beta-infinity from rebuilt signed source')
    _contained(_interval(remainder['second_expansion_lower'], remainder['second_expansion_upper']),
               expansion, 'second expansion from rebuilt signed source')
    _contained(_interval(remainder['second_remainder_lower'], remainder['second_remainder_upper']),
               error, 'finite-count second remainder')
    _contained(_interval(remainder['scaled_cubic_lower'], remainder['scaled_cubic_upper']),
               scaled, 'cubically scaled finite-count remainder')
    assert remainder['remainder_sign_certified_positive'] == (F(remainder['second_remainder_lower']) > 0)
    assert remainder['signed_source_features'] == source['features']
    assert remainder['signed_source_q_sha256'] == source['q_sha256']
    assert remainder['finite_count_error_only'] is True


def _toy_checks():
    """Exact rational PSD examples, verified against the quadratic formula."""
    A = _Intervals(60)
    exact = lambda x: A.fraction(F(x), F(x))
    examples = [
        (F(3, 5), F(1, 10), F(1, 5), [F(1), F(1, 4)]),
        (F(3, 5), F(-1, 10), F(1, 5), [F(1), F(-1, 4)]),
        (F(1, 2), F(0), F(1, 4), [F(1), F(0)]),
    ]
    for a, b, c, q in examples:
        assert a > 0 and c > 0 and a*c-b*b > 0
        matrix = [[exact(a), exact(b)], [exact(b), exact(c)]]
        result = finite_trace_certificate(matrix, q, 60)
        trace = a+c
        discriminant = (a-c)**2+4*b*b
        lower = F(result['rayleigh'][0])
        upper = F(result['finite_upper'])
        # lambda+=(trace+sqrt(discriminant))/2, checked without rounded sqrt.
        assert 2*lower-trace >= 0 and (2*lower-trace)**2 <= discriminant
        assert 2*upper-trace >= 0 and (2*upper-trace)**2 >= discriminant
    rejected = 0
    for matrix, q in [
        ([[exact(1), exact(0)], [exact(0), exact(1)]], [F(1), F(0)]),
        ([[exact(F(3, 5)), exact(0)], [exact(0), exact(F(1, 5))]], [F(0), F(1)]),
    ]:
        try:
            finite_trace_certificate(matrix, q, 60)
        except ArithmeticError:
            rejected += 1
        else:
            raise AssertionError('an unresolved/nonpositive trace screen was accepted')
    assert rejected == 2
    return dict(exact_psd_toys=len(examples), unresolved_trace_screens_rejected=rejected)


def _precision_checks(case_rows, vectors, sources, signed_vectors):
    results = []
    for D, m in [('4', 80), ('6', 70)]:
        source_key = (D, '2', 40 if D == '4' else 50)
        source = sources[source_key]
        sv = signed_vectors[source_key]
        finer_source, _ = signed_second_certificate(D, source_key[2], sv['trial'], 80, '2')
        for order in (0, 1):
            row = next(r for r in case_rows if r['D'] == D and int(r['m']) == m
                       and int(r['predictor_order']) == order)
            p = int(row['features'])
            trial = vectors[(D, row['V'], p)]['trial']
            base, audit = predictor_certificate(D, m, p, trial, 60, row['V'], bool(order))
            fine, _ = predictor_certificate(D, m, p, trial, 80, row['V'], bool(order))
            for lo, hi in [('rho_lower', 'rho_upper'), ('beta_lower', 'beta_upper')]:
                assert F(base[lo]) <= F(fine[lo]) <= F(fine[hi]) <= F(base[hi])
            rem = second_remainder_certificate(base, source)
            rem_fine = second_remainder_certificate(fine, finer_source)
            for lo, hi in [('second_remainder_lower', 'second_remainder_upper'),
                           ('scaled_cubic_lower', 'scaled_cubic_upper')]:
                assert F(rem[lo]) <= F(rem_fine[lo]) <= F(rem_fine[hi]) <= F(rem[hi])
            with localcontext() as alien:
                alien.prec = 2
                alien.rounding = ROUND_CEILING
                again, again_audit = predictor_certificate(D, m, p, trial, 60, row['V'], bool(order))
                assert again == base and again_audit == audit
                assert second_remainder_certificate(again, source) == rem
            results.append(dict(D=D, m=m, features=p, predictor_order=order,
                                precision_nesting=True, ambient_context_invariance=True))
    return results


def main():
    assert 'numpy' not in sys.modules and 'scipy' not in sys.modules
    main_rows = _rows('predictor_residual')
    rank_rows = _rows('predictor_rank_refinement')
    vector_list = _json('predictor_vectors')
    saved_audits = _json('predictor_audits')
    vectors = {(str(v['D']), str(v['V']), int(v['features'])): v for v in vector_list}
    assert len(vectors) == len(vector_list), 'duplicate envelope predictor pair'
    all_rows = main_rows+rank_rows
    row_groups = {}
    for row in all_rows:
        row_groups.setdefault(_key(row), []).append(row)
    audit_lookup = {_key(a['summary']): a for a in saved_audits}
    assert len(audit_lookup) == len(saved_audits), 'duplicate saved certificate audit'
    assert set(row_groups) == set(audit_lookup), 'certificate rows/audits coverage mismatch'

    source_vectors = {(str(v['D']), str(v['V']), int(v['features'])): v
                      for v in _json('signed_second_vectors')}
    sources = {}
    for key in [('4', '2', 40), ('6', '2', 50)]:
        v = source_vectors[key]
        source, _ = signed_second_certificate(key[0], key[2], v['trial'], 60, key[1])
        # The study driver reads its signed source through CSV. Recreate that
        # serialization from the freshly certified source, never from CSV.
        sources[key] = {name: str(value) for name, value in source.items()}

    for key, saved_rows in sorted(row_groups.items()):
        D, V, m, p, order, precision = key
        assert precision == 60 and order in (0, 1)
        vector = vectors[(D, V, p)]
        trial = vector['trial']
        assert (trial['D'], trial['V'], trial['features']) == (D, V, p)
        record, audit = predictor_certificate(D, m, p, trial, precision, V, bool(order))
        source = sources[(D, V, 40 if D == '4' else 50)]
        remainder = second_remainder_certificate(record, source)
        for saved in saved_rows:
            _compare(record, saved, key)
            _compare(remainder, saved, ('second remainder', key))
            extras = set(saved)-set(record)-set(remainder)
            assert all(name.startswith('diagnostic_') or name.endswith('_float')
                       for name in extras), (key, 'unverified strict CSV fields', sorted(extras))
            _fraction_checks(saved, audit, trial)
            _remainder_fraction_checks(remainder, saved, source)
        saved_audit = audit_lookup[key]
        audit['summary'] = dict(record, **remainder)
        assert saved_audit == audit, ('full predictor audit', key)

    assert len(main_rows) == 40 and len(rank_rows) == 6
    assert len(row_groups) == 44
    for D in ('4', '6'):
        rows = [r for r in main_rows if r['D'] == D]
        assert len(rows) == 20 and {int(r['predictor_order']) for r in rows} == {0, 1}
        counts = [{int(r['m']) for r in rows if int(r['predictor_order']) == o} for o in (0, 1)]
        assert counts[0] == counts[1] == {16, 32, 64, 70, 80, 128, 256, 512, 1024, 2048}
        for m in counts[0]:
            plain = next(r for r in rows if int(r['m']) == m and r['predictor_order'] == '0')
            refined = next(r for r in rows if int(r['m']) == m and r['predictor_order'] == '1')
            assert F(refined['beta_width_upper']) < F(plain['beta_width_upper'])
            assert refined['remainder_sign_certified_positive'] == 'True'
    expected_rank = {('4', 80, p, 1) for p in (16, 24, 32)} | {
        ('6', 70, p, 1) for p in (24, 32, 48)}
    assert {(r['D'], int(r['m']), int(r['features']), int(r['predictor_order']))
            for r in rank_rows} == expected_rank
    toy = _toy_checks()
    checks = _precision_checks(main_rows, vectors, sources, source_vectors)
    saved_checks = _json('predictor_checks')
    expected_study_checks = [dict(D=D, m=m, features=p, precision_nesting=True,
        ambient_context_invariance=True, finite_count_only=True, uses_numpy_in_replay=False)
        for D, m, p in [('4', 80, 32), ('6', 70, 48)]]
    assert saved_checks == dict(self_checks=expected_study_checks,
        all_displayed_main_remainders_positive=True, unique_certificates=len(row_groups),
        finite_count_only=True, uniform_third_order_bound_replaced=False)
    assert 'numpy' not in sys.modules and 'scipy' not in sys.modules
    print(json.dumps(dict(main_rows=len(main_rows), rank_refinement_rows=len(rank_rows),
        unique_certificates=len(row_groups), complete_saved_audits_reproduced=True,
        signed_coefficient_sources_rebuilt_from_trials=True,
        independent_fraction_error_propagation=True, **toy, self_checks=checks,
        dependencies='Python standard library only; saved exact predictor/source trials'), indent=2))


if __name__ == '__main__':
    main()
