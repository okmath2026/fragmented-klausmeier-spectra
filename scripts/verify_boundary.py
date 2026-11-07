"""Replay every boundary/gap/remainder certificate using only the standard library."""
import csv
import json
from pathlib import Path
from validated_boundary import boundary_certificate,boundary_remainder


def main():
    root=Path(__file__).resolve().parents[1]
    vectors=json.loads((root/"data/boundary_vectors.json").read_text())
    with (root/"data/boundary_certificates.csv").open(newline="") as stream:
        rows=list(csv.DictReader(stream))
    assert len(rows)==len(vectors)
    cache={}
    for vector,row in zip(vectors,rows):
        assert (vector["D"],vector["N"])==(row["D"],int(row["N"]))
        actual=boundary_certificate(vector["D"],vector["N"],vector["q"],vector["precision"])
        for key,value in actual.items():assert str(value)==row[key],(vector["D"],vector["N"],key)
        cache[actual["D"],actual["N"]]=actual
    with (root/"data/boundary_remainders.csv").open(newline="") as stream:
        remainders=list(csv.DictReader(stream))
    for row in remainders:
        actual=boundary_remainder(cache[row["D"],int(row["N"])],int(row["m"]))
        for key,value in actual.items():assert str(value)==row[key],(row["D"],row["m"],key)
    print(json.dumps(dict(boundary_rows_reproduced=len(rows),remainder_rows_reproduced=len(remainders),
                         dependencies="Python standard library only; saved exact even trial vectors"),indent=2))


if __name__=="__main__":main()
