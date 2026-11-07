"""Independently rerun every saved Decimal enclosure, without NumPy or BLAS."""
import csv
import json
from pathlib import Path
from validated_gaussian import gaussian_certificate, self_check


def main():
    root=Path(__file__).resolve().parents[1]
    vectors=json.loads((root/"data/certificate_vectors.json").read_text())
    by_key={(v["L"],v["N"]):v for v in vectors}
    assert len(by_key)==len(vectors)
    cache={};count=0
    for dataset in ("certificates","quadratic_certificates"):
        with (root/("data/"+dataset+".csv")).open(newline="",encoding="utf-8") as stream:
            rows=list(csv.DictReader(stream))
        for row in rows:
            key=(row["L"],int(row["N"]))
            vector=by_key[key]
            if key not in cache:
                cache[key]=gaussian_certificate(vector["L"],vector["N"],vector["q"],vector["precision"])
            actual=cache[key]
            assert set(actual).issubset(row)
            for field,value in actual.items():
                assert (str(value) if value is not None else "")==row[field],(dataset,key,field)
            count+=1
    assert len(cache)==len(vectors)
    print(json.dumps(dict(certificate_rows_reproduced=count,unique_vectors=len(cache),fields_compared=list(actual),
                          arithmetic_checks=self_check()),indent=2))


if __name__=="__main__":
    main()
