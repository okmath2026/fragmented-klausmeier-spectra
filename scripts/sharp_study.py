"""Reproduce convergent odd-sector gaps and structured remainder bounds.

The saved baseline envelope certificates remain unchanged, allowing a paired
comparison with the original bounds. Float64 FFT values never enter a strict
certificate. Run verify_sharp.py for a standard-library-only reconstruction.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import csv
import json
from decimal import Decimal
from spectral_study import ROOT, DATA, FIG, TABLE, csvfile, table, plot, decimal_display, scientific_upper
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from validated_gap import odd_trial_vector, gap_certificate, gap_self_check
from validated_remainder import sharp_remainder, sharp_resolution, sharp_self_check, sharp_eventual_sign
from validated_boundary import boundary_remainder
from fragment_fft import fragment_fft


def read_rows(name):
    with (DATA/(name+".csv")).open(newline="") as stream:
        return list(csv.DictReader(stream))


def make_figure(gaps, remainders):
    blue, orange = colors.HexColor("#1f77b4"), colors.HexColor("#d95f02")
    green = colors.HexColor("#228833")
    c = canvas.Canvas(str(FIG/"sharpened_certification.pdf"), pagesize=(640,510))
    for D, box, title in [("4",(0,255,320,255),"(a) Gap widths, D=4"),
                           ("6",(320,255,320,255),"(b) Gap widths, D=6")]:
        rr = [r for r in gaps if r["D"] == D]
        plot(c,box,title,"Midpoint cells N","Certified gap width",
             [("Parity norm bound",[r["N"] for r in rr],[float(r["baseline_width_upper"]) for r in rr],orange),
              ("Odd Perron bound",[r["N"] for r in rr],[float(r["gamma_width_upper"]) for r in rr],blue)],
             True,True,legend_location="lower_left")
    for D, box, title in [("4",(0,0,320,255),"(c) Corrected errors, D=4"),
                           ("6",(320,0,320,255),"(d) Corrected errors, D=6")]:
        rr = [r for r in remainders if r["D"] == D]
        old = [r for r in rr if r["baseline_corrected_bound"] != "unavailable"]
        plot(c,box,title,"Patch count m","Error / certified upper bound",
             [("Original bound",[r["m"] for r in old],[float(r["baseline_corrected_bound"]) for r in old],orange),
              ("Structured bound",[r["m"] for r in rr],[float(r["midpoint_coefficient_corrected_error_upper"]) for r in rr],blue),
              ("Float error",[r["m"] for r in rr],[r["corrected_error_float"] for r in rr],green)],
             True,True,xticks=[32,128,512,1024],legend_location="lower_left")
    c.save()


def main():
    baseline = read_rows("boundary_certificates")
    corrections = read_rows("fragment_correction")
    gaps, remainders, vectors, checks = [], [], [], []
    for record in baseline:
        D, n = record["D"], int(record["N"])
        q = odd_trial_vector(D,n)
        actual = gap_certificate(record,q)
        oldwidth = Decimal(record["gamma_width_upper"])
        gaps.append(dict(**actual,baseline_width_upper=str(oldwidth),
                         width_gain_float=float(oldwidth)/float(actual["gamma_width_upper"]),
                         gap_float=float(record["gap_float"])))
        vectors.append(dict(D=D,N=n,precision=int(record["precision"]),q_odd=q))
        assert float(actual["gamma_lower"]) < float(record["gap_float"]) < float(actual["gamma_upper"])
        if n == 80:
            checks.append(gap_self_check(record,q))
        print(f'sharp gap D={D}, N={n}: completed',flush=True)
    thresholds = []
    for D in ["4","6"]:
        record = next(r for r in baseline if r["D"] == D and int(r["N"]) == 640)
        reference = next(r for r in corrections if float(r["span"]) == float(D))
        thresholds.append(dict(sharp_resolution=sharp_resolution(record),
                               baseline_resolution=int(record["m_resolution_min"]),
                               baseline_sign=int(record["m_sign_all"]),**sharp_eventual_sign(record)))
        checks.append(sharp_self_check(record,32))
        cache = {}
        for m in [32,64,128,256,512,1024]:
            actual = sharp_remainder(record,m)
            if m not in cache: cache[m] = fragment_fft(m,float(D),8)
            if 2*m not in cache: cache[2*m] = fragment_fft(2*m,float(D),8)
            fine, coarse = cache[m], fragment_fft(m,float(D),4)
            ce = abs(fine["beta"]-float(reference["limit_beta"])-float(actual["c_midpoint"])/m)
            re = abs(2*cache[2*m]["beta"]-fine["beta"]-float(reference["limit_beta"]))
            assert ce < float(actual["midpoint_coefficient_corrected_error_upper"])
            assert re < float(actual["richardson_error_upper"])
            assert float(actual["beta_m_lower"]) < fine["beta"] < float(actual["beta_m_upper"])
            assert abs(fine["beta"]-coarse["beta"]) < 2e-12
            try:
                old = boundary_remainder(record,m)["midpoint_coefficient_corrected_error_upper"]
                gain = float(old)/float(actual["midpoint_coefficient_corrected_error_upper"])
            except ArithmeticError:
                old, gain = "unavailable", "unavailable"
            remainders.append(dict(**actual,beta_fft=fine["beta"],corrected_error_float=ce,
                                   richardson_error_float=re,refinement_4_8=abs(fine["beta"]-coarse["beta"]),
                                   fft_collatz_width=fine["collatz_width"],fft_residual=fine["residual"],
                                   baseline_corrected_bound=old,bound_gain_float=gain))
    csvfile("sharp_gaps",gaps)
    csvfile("sharp_remainders",remainders)
    (DATA/"sharp_vectors.json").write_text(json.dumps(vectors,indent=2)+"\n",encoding="ascii")
    (DATA/"sharp_checks.json").write_text(json.dumps(dict(checks=checks,thresholds=thresholds),indent=2)+"\n",encoding="ascii")
    table("sharp_gaps",'Certified spectral gaps for the continuous operator, using its positive odd kernel on the half-interval. Endpoints are rounded outwards. The gain compares the enclosure widths with those from the parity norm bound.',
          ["$D$","$N$","$\\gamma_-$","$\\gamma_+$","Width $\\leq$","Gain"],
          [[r["D"],str(r["N"]),decimal_display(r["gamma_lower"],8),decimal_display(r["gamma_upper"],8,True),
            scientific_upper(r["gamma_width_upper"]),f'{r["width_gain_float"]:.1f}'] for r in gaps])
    table("sharp_remainders",'Certified error bounds for coefficient correction and Richardson extrapolation. The corrected bound includes the saved coefficient radius divided by $m$. The observed errors are float64 diagnostics.',
          ["$D$","$m$","Corr. bound $\\leq$","Observed error","Extr. bound $\\leq$","Gain"],
          [[r["D"],str(r["m"]),scientific_upper(r["midpoint_coefficient_corrected_error_upper"]),
            f'{r["corrected_error_float"]:.2e}',scientific_upper(r["richardson_error_upper"]),
            "--" if r["bound_gain_float"] == "unavailable" else f'{r["bound_gain_float"]:.1f}'] for r in remainders])
    table("sharp_budget",'Components of the remainder bound at $m=256$, $N=640$. Each error component is rounded upwards. The Schur denominator is rounded downwards.',
          ["$D$","Denominator $\\geq$","Spectral term","Taylor term","Cell term","Coeff. term"],
          [[r["D"],decimal_display(r["schur_denominator_lower"],6)]+[scientific_upper(r[k]) for k in
            ["spectral_remainder_upper","taylor_remainder_upper","cell_cancellation_upper","coefficient_conversion_upper"]]
           for r in remainders if r["m"] == 256])
    make_figure(gaps,remainders)
    path = ROOT/"output/pdf/shape_spectral_results.csv"
    with path.open(newline="") as stream:
        previous = [r for r in csv.DictReader(stream) if r["experiment"] not in ("sharp_gaps","sharp_remainders")]
    rows = previous+[dict(experiment="sharp_gaps",**r) for r in gaps]+[dict(experiment="sharp_remainders",**r) for r in remainders]
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary = json.loads((DATA/"summary.json").read_text())
    summary.update(sharp_gap_rows=len(gaps),sharp_remainder_rows=len(remainders),sharp_thresholds=thresholds,
                   sharp_checks=checks,experiment_rows=len(rows))
    for key, field in [("max_collatz_width","fft_collatz_width"),
                       ("max_eigen_residual","fft_residual"),
                       ("matrix_free_refinement_max","refinement_4_8")]:
        summary[key] = max(summary[key],*(r[field] for r in remainders))
    (DATA/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (TABLE/"metrics.tex").write_text("\\newcommand{\\SmallSlope}{%.4f}\n\\newcommand{\\LargeSlope}{%.4f}\n\\newcommand{\\MaxPerronWidth}{%.2e}\n\\newcommand{\\MaxResidual}{%.2e}\n" %
        (summary["small_log_slope"],summary["large_log_slope"],summary["max_collatz_width"],summary["max_eigen_residual"]),encoding="ascii")
    print(json.dumps(dict(sharp_gap_rows=len(gaps),sharp_remainder_rows=len(remainders),thresholds=thresholds,result_rows=len(rows)),indent=2))
    from coupled_study import main as coupled_study
    coupled_study()


if __name__ == "__main__": main()
