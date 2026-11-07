"""Replay the entire sharpened certification chain with the standard library."""
import csv
import json
from pathlib import Path
from validated_boundary import boundary_certificate, boundary_remainder
from validated_gap import gap_certificate, gap_self_check
from validated_remainder import sharp_remainder, sharp_resolution, sharp_eventual_sign, sharp_self_check


def main():
    root = Path(__file__).resolve().parents[1]
    def rows(name):
        with (root/"data"/(name+".csv")).open(newline="") as stream:
            return list(csv.DictReader(stream))
    vectors = json.loads((root/"data/boundary_vectors.json").read_text())
    baseline = rows("boundary_certificates")
    cache = {}
    for vector,row in zip(vectors,baseline,strict=True):
        actual = boundary_certificate(vector["D"],vector["N"],vector["q"],vector["precision"])
        for key,value in actual.items(): assert str(value) == row[key], (vector["D"],vector["N"],key)
        cache[actual["D"],actual["N"]] = actual
    oldremainders = rows("boundary_remainders")
    for row in oldremainders:
        actual = boundary_remainder(cache[row["D"],int(row["N"])],int(row["m"]))
        for key,value in actual.items(): assert str(value) == row[key],(row["D"],row["m"],key)
    oddvectors = json.loads((root/"data/sharp_vectors.json").read_text())
    gaps = rows("sharp_gaps")
    checks = []
    for vector,row in zip(oddvectors,gaps,strict=True):
        record = cache[vector["D"],vector["N"]]
        actual = gap_certificate(record,vector["q_odd"])
        for key,value in actual.items(): assert str(value) == row[key],(vector["D"],vector["N"],key)
        if vector["N"] == 80: checks.append(gap_self_check(record,vector["q_odd"]))
    remainders = rows("sharp_remainders")
    for row in remainders:
        actual = sharp_remainder(cache[row["D"],int(row["N"])],int(row["m"]))
        for key,value in actual.items(): assert str(value) == row[key],(row["D"],row["m"],key)
        try: old = boundary_remainder(cache[row["D"],int(row["N"])],int(row["m"]))["midpoint_coefficient_corrected_error_upper"]
        except ArithmeticError: old = "unavailable"
        assert row["baseline_corrected_bound"] == old
    saved = json.loads((root/"data/sharp_checks.json").read_text())
    for D in ["4","6"]:
        record = cache[D,640]
        checks.append(sharp_self_check(record,32))
        actual = dict(sharp_resolution=sharp_resolution(record),
                      baseline_resolution=record["m_resolution_min"],baseline_sign=record["m_sign_all"],
                      **sharp_eventual_sign(record))
        expected = next(r for r in saved["thresholds"] if r["D"] == D)
        assert actual == expected,(D,"uniform threshold")
    assert checks == saved["checks"]
    print(json.dumps(dict(baseline_boundary_rows=len(baseline),baseline_remainder_rows=len(oldremainders),
                         sharpened_gap_rows=len(gaps),sharpened_remainder_rows=len(remainders),
                         uniform_sign_thresholds_reproduced=True,precision_and_context_checks=True,
                         dependencies="Python standard library only; exact saved trial strings"),indent=2))


if __name__ == "__main__": main()
