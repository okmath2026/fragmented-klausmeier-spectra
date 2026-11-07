"""Standard-library replay of the uniform Hilbert-space cubic certificates."""
from pathlib import Path
from fractions import Fraction
import csv
import json

from hilbert_remainder_study import selected_sources
from validated_hilbert_remainder import (hilbert_remainder_family,
    hilbert_remainder_self_check, hilbert_remainder_at_count)

ROOT = Path(__file__).resolve().parent.parent


def main():
    with (ROOT/'data/hilbert_remainders.csv').open(encoding='utf-8',newline='') as handle:
        saved = list(csv.DictReader(handle))
    replayed = []
    checks = []
    for source in selected_sources():
        rows, audit = hilbert_remainder_family(source['D'], source['p'],source['trial'],
            (70,80,160,320),60,source['V'])
        for row in rows:
            row['minimum_half_anchor'] = audit['minimum_half_anchor']
            match = next(x for x in saved if x['D'] == row['D'] and int(x['M']) == row['M'])
            assert set(row) == set(match)
            assert all(str(row[key]) == match[key] for key in row), (row['D'],row['M'])
            assert Fraction(row['eta_upper_exact']) < 1
            assert Fraction(row['anchor_error_upper_exact']) == Fraction(row['K3_upper_exact'])/row['M']**3
            at_count = hilbert_remainder_at_count(row,2*row['M'])
            assert Fraction(at_count['remainder_upper_exact']) == Fraction(row['K3_upper_exact'])/(2*row['M'])**3
            replayed.append(row)
        check = hilbert_remainder_self_check(source['D'],source['p'],source['trial'],60,source['V'])
        checks.append(check)
        print(json.dumps(check))
    assert len(replayed) == len(saved)
    print(json.dumps(dict(replayed_rows=len(replayed),all_checks_passed=True,
        standard_library_only=True,continuous_even_gap_replayed=True,
        uniform_cubic_bounds=True,finite_ratio_localization=False)))


if __name__ == '__main__':
    main()
