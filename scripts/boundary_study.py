"""Rerun boundary certificates and FFT checks while retaining the other experiments."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import csv
import json
from spectral_study import ROOT,DATA,TABLE,WIDTHS,RESIDUALS,boundary_experiments,boundary_figure


def main():
    with (DATA/"fragment_correction.csv").open(newline="") as stream:
        corrections=[{k:float(v) for k,v in row.items()} for row in csv.DictReader(stream)]
    certified,remainders,checks=boundary_experiments(corrections)
    boundary_figure(certified,remainders)
    path=ROOT/"output/pdf/shape_spectral_results.csv"
    with path.open(newline="") as stream:
        previous=[r for r in csv.DictReader(stream) if r["experiment"] not in ("boundary_certificates","boundary_remainders")]
    rows=previous+[dict(experiment="boundary_certificates",**r) for r in certified]+[dict(experiment="boundary_remainders",**r) for r in remainders]
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary=json.loads((DATA/"summary.json").read_text())
    summary.update(strict_boundary_certificate_rows=len(certified),strict_remainder_rows=len(remainders),
                   boundary_checks=checks,matrix_free_refinement_max=max(r["refinement_4_8"] for r in remainders),
                   experiment_rows=len(rows))
    summary["max_collatz_width"]=max(summary["max_collatz_width"],*WIDTHS,*[r["fft_collatz_width"] for r in remainders])
    summary["max_eigen_residual"]=max(summary["max_eigen_residual"],*RESIDUALS,*[r["fft_residual"] for r in remainders])
    (DATA/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (TABLE/"metrics.tex").write_text("\\newcommand{\\SmallSlope}{%.4f}\n\\newcommand{\\LargeSlope}{%.4f}\n\\newcommand{\\MaxPerronWidth}{%.2e}\n\\newcommand{\\MaxResidual}{%.2e}\n"%
            (summary["small_log_slope"],summary["large_log_slope"],summary["max_collatz_width"],summary["max_eigen_residual"]),encoding="ascii")
    print(json.dumps(dict(boundary_rows=len(certified),remainder_rows=len(remainders),result_rows=len(rows),checks=checks),indent=2))
    from sharp_study import main as sharpened_study
    sharpened_study()


if __name__=="__main__":main()
