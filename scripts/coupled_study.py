"""Water-domain comparisons and certified global spectral-loss maxima.

Run after sharp_study.py, or use the saved baseline data for this focused run.
Every plotted value and proof input is saved. No empirical observations used.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import argparse
import csv
import json
import math
from fractions import Fraction
from decimal import Decimal
from spectral_study import ROOT,DATA,FIG,TABLE,csvfile,table,plot,decimal_display,scientific_upper
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from peak_trial import fragment_trial
from validated_peaks import (fragment_certificate,tail_certificate,global_peak_certificate,peak_self_check,
                             rayleigh_envelope_certificate,rayleigh_tail_certificate,rayleigh_self_check)
from validated_remainder import sharp_eventual_sign
from validated_water import rational_water_certificate,coupled_margin,synergy_certificate,first_water_only_m,water_self_check
from check_peak_geometry import independent_geometry_check


def read_rows(name):
    with (DATA/(name+".csv")).open(newline="") as stream:
        return list(csv.DictReader(stream))


def fd_water(length,dw,segments):
    """Independent float64 tridiagonal finite-difference comparison solve."""
    h=length/segments;k=dw/(h*h);n=segments-1
    diagonal=[1+2*k]*n;rhs=[1.]*n
    for i in range(1,n):
        multiplier=-k/diagonal[i-1]
        diagonal[i]-=multiplier*(-k);rhs[i]-=multiplier*rhs[i-1]
    z=[0.]*n;z[-1]=rhs[-1]/diagonal[-1]
    for i in range(n-2,-1,-1):z[i]=(rhs[i]+k*z[i+1])/diagonal[i]
    return max(z)


def make_figure(finite,tails,global_rows,water,fd):
    blue,orange,green=(colors.HexColor(v) for v in ("#1f77b4","#d95f02","#228833"))
    c=canvas.Canvas(str(FIG/"water_global_peaks.pdf"),pagesize=(640,510))
    for D,box,title in [("4",(0,255,320,255),"(a) Global spectral maximum, D=4"),
                         ("6",(320,255,320,255),"(b) Global spectral maximum, D=6")]:
        rr=[r for r in finite if r["D"]==D];tail=next(r for r in tails if r["D"]==D)
        winner=next(r for r in global_rows if r["D"]==D)
        M=int(tail["anchor_m"])
        plot(c,box,title,"Patch count m","Continuous beta1",
             [("Lower",[r["m"] for r in rr],[float(winner["winner_beta_lower"] if r["m"]==winner["winner_m"] else r["beta_lower"]) for r in rr],blue),
              ("Upper",[r["m"] for r in rr],[float(winner["winner_beta_upper"] if r["m"]==winner["winner_m"] else r["beta_upper"]) for r in rr],green),
              ("Tail bound",[M,2*M],[float(tail["all_m_ge_anchor_beta_upper"])]*2,orange)],
             True,xticks=[2,4,8,16] if D=="4" else [2,4,8,16,32],legend_location="above")
    plot(c,(0,0,320,255),"(c) Water-domain comparison","Patch count m","Comparison factor",
         [("Patch water",[r["m"] for r in water if r["D"]=="4" and r["water_domain"]=="component"],
           [float(r["q_upper"]) for r in water if r["D"]=="4" and r["water_domain"]=="component"],blue),
          ("Ambient D=4",[r["m"] for r in water if r["D"]=="4" and r["water_domain"]=="ambient"],
           [float(r["q_upper"]) for r in water if r["D"]=="4" and r["water_domain"]=="ambient"],orange),
          ("Ambient D=6",[r["m"] for r in water if r["D"]=="6" and r["water_domain"]=="ambient"],
           [float(r["q_upper"]) for r in water if r["D"]=="6" and r["water_domain"]=="ambient"],green)],
         True,xticks=[2,4,8,16],legend_location="above",ylim=(0,1))
    plot(c,(320,0,320,255),"(d) Water comparison solve","FD segments N","Maximum-value error",
         [(f'Case {case}',[r["segments"] for r in fd if r["case"]==case],
           [r["error_float"] for r in fd if r["case"]==case],col)
          for case,col in [(1,blue),(2,orange),(3,green)]],True,True,legend_location="lower_left")
    c.save()


def main(reuse_direct=None):
    baseline=read_rows("boundary_certificates")
    boundary_vectors=json.loads((DATA/"boundary_vectors.json").read_text())
    finite=[];support=[];vectors=[];refinement=[];envelopes=[];envelope_vectors=[]
    tails=[];dyadic=[];global_rows=[];checks=[];rayleigh_checks=[]
    reused=json.loads(open(reuse_direct).read()) if reuse_direct else None
    for D,winner_m,M in [("4",3,11),("6",4,17)]:
        principal_vector=next(v for v in boundary_vectors if v["D"]==D and v["N"]==640)
        envelope_vectors.append(principal_vector)
        envelope=rayleigh_envelope_certificate(D,640,principal_vector["q"],principal_vector["precision"])
        envelopes.append(envelope)
        tail=rayleigh_tail_certificate(envelope,M);tails.append(tail)
        rayleigh_checks.append(rayleigh_self_check(D,640,principal_vector["q"],M))
        # This auxiliary two-sided chain proves the separate doubling result.
        # Neither its gap nor its boundary coefficient enters the peak proof.
        record=next(r for r in baseline if r["D"]==D and int(r["N"])==640)
        dyadic.append(tail_certificate(record,sharp_eventual_sign(record)))
        for m in range(2,max(M,17)):
            if reused:
                row=next(r for r in reused["rows"] if r["D"]==D and int(r["m"])==m)
                vector=next(v for v in reused["vectors"] if v["D"]==D and int(v["m"])==m and int(v["cells_per_patch"])==4)
            else:
                q,diagnostic=fragment_trial(D,m,4)
                row=fragment_certificate(D,m,4,q)
                row=dict(**row,**diagnostic)
                vector=dict(D=D,m=m,cells_per_patch=4,precision=60,q=q)
            (finite if m<M else support).append(row)
            vectors.append(vector)
        print(f'direct interval D={D}: {M-2} peak cases completed',flush=True)
        winner_coarse=next(r for r in finite if r["D"]==D and r["m"]==winner_m)
        refinement.append(dict(winner_coarse))
        winner_vector=next(v for v in vectors if v["D"]==D and v["m"]==winner_m and v["cells_per_patch"]==4)
        checks.append(peak_self_check(D,winner_m,4,winner_vector["q"]))
        checks.append(independent_geometry_check(D,winner_m,4,winner_vector["q"]))
        for r in [8,16,32]:
            q,diagnostic=fragment_trial(D,winner_m,r)
            row=fragment_certificate(D,winner_m,r,q)
            refinement.append(dict(**row,**diagnostic))
            vectors.append(dict(D=D,m=winner_m,cells_per_patch=r,precision=60,q=q))
        proof_rows=[r for r in finite if r["D"]==D and r["m"]!=winner_m]+[refinement[-1]]
        global_rows.append(global_peak_certificate(proof_rows,tail,winner_m))
    thresholds=[first_water_only_m(dw) for dw in [".01",".1","1"]]
    water=[]
    for D in ["4","6"]:
        for m in range(2,17):
            direct=next(r for r in finite+support if r["D"]==D and r["m"]==m)
            if m==(3 if D=="4" else 4):direct=next(r for r in refinement if r["D"]==D and r["cells_per_patch"]==32)
            for model in ["component","ambient"]:
                length=Fraction(2,m) if model=="component" else Fraction(D)
                distance=Fraction(0) if model=="component" or m%2 else (Fraction(D)-2)/(2*(m-1))
                cap=rational_water_certificate(length,Fraction(1),distance)
                margin=coupled_margin(cap,direct["beta_lower"])
                water.append(dict(D=D,V="2",m=m,water_domain=model,
                                  spectral_cells_per_patch=direct["cells_per_patch"],**cap,**margin))
    direct=next(r for r in finite if r["D"]=="4" and r["m"]==2)
    cap=rational_water_certificate(Fraction(1),Fraction(1))
    synergy=[dict(D="4",V="2",m=2,water_domain="component",spectral_cells_per_patch=4,
                  **cap,**synergy_certificate(cap,direct["beta_lower"],direct["beta_upper"],".6"))]
    assert Decimal(synergy[0]["coupled_margin_lower"])>Decimal(".00327021")
    assert Decimal(synergy[0]["water_only_margin_upper"])<0 and Decimal(synergy[0]["crude_margin_upper"])<0
    fd=[]
    for case,L,dw in [(1,Fraction(1),Fraction(".1")),(2,Fraction(2,3),Fraction(1)),(3,Fraction(4),Fraction(1))]:
        cap=rational_water_certificate(L,dw);reference=(float(cap["q_lower"])+float(cap["q_upper"]))/2
        previous=None
        for segments in [32,64,128,256]:
            value=fd_water(float(L),float(dw),segments);error=abs(value-reference)
            order=math.log2(previous/error) if previous else None
            fd.append(dict(case=case,length_numerator=L.numerator,length_denominator=L.denominator,
                dw=str(dw),segments=segments,maximum_float=value,comparison_reference_float=reference,
                error_float=error,observed_order=order))
            if order is not None:assert 1.9<order<2.1
            previous=error
    experiments=[("direct_fragments",finite),("water_support",support),("peak_refinement",refinement),
                 ("peak_envelopes",envelopes),("peak_tails",tails),("dyadic_tails",dyadic),
                 ("global_peaks",global_rows),("water_thresholds",thresholds),("water_domains",water),
                 ("water_synergy",synergy),("water_convergence",fd)]
    for name,rows in experiments:
        csvfile(name,rows)
    (DATA/"peak_vectors.json").write_text(json.dumps(vectors,indent=2)+"\n",encoding="ascii")
    (DATA/"peak_envelope_vectors.json").write_text(json.dumps(envelope_vectors,indent=2)+"\n",encoding="ascii")
    check_record=dict(peak_checks=checks,water_checks=water_self_check(),
                      rayleigh_checks=rayleigh_checks,
                      gap_free_peak_chain=True,joint_gain_strictly_certified=True,
                      finite_coverage={D:[2,next(r["anchor_m"] for r in tails if r["D"]==D)-1] for D in ["4","6"]})
    (DATA/"coupled_checks.json").write_text(json.dumps(check_record,indent=2)+"\n",encoding="ascii")
    table("global_peaks","Strict global maxima of seed-dispersal spectral loss over every integer $m\\geq2$ in the prescribed equal-patch family. All displayed interval endpoints are rounded outwards.",
          ["$D$","$m_*$","$\\beta_{m_*}\\in$","Other finite $\\leq$","All $m\\geq M$ $\\leq$","$M$"],
          [[r["D"],str(r["winner_m"]),"["+decimal_display(r["winner_beta_lower"],7)+", "+decimal_display(r["winner_beta_upper"],7,True)+"]",
            decimal_display(r["competing_finite_upper"],7,True),decimal_display(r["infinite_tail_upper"],7,True),str(r["tail_first_m"])] for r in global_rows])
    table("rayleigh_tails",'Inputs and bounds for the one-sided Rayleigh tail estimate. The only spectral input is a lower bound for the principal eigenvalue of the continuous operator. No quantified envelope gap or coefficient interval is required.',
          ["$D$","$r_-\\geq$","$S^+\\leq$","$M$","$T_M^{\\rm R}\\leq$"],
          [[r["D"],decimal_display(r["rho_lower"],8),decimal_display(r["rayleigh_remainder_constant_upper"],8,True),
            str(r["anchor_m"]),decimal_display(r["all_m_ge_anchor_beta_upper"],8,True)] for r in tails])
    table("peak_refinement","Direct continuous spectral enclosures at the global maximisers. The cell-average bound is independent of the asymptotic expansion.",
          ["$D$","$m$","Cells/patch","$\\beta_-$","$\\beta_+$","Width $\\leq$"],
          [[r["D"],str(r["m"]),str(r["cells_per_patch"]),decimal_display(r["beta_lower"],8),decimal_display(r["beta_upper"],8,True),scientific_upper(r["beta_width_upper"])] for r in refinement])
    table("water_thresholds","Certified first water-only sufficient thresholds for $V=2$, $A_{\\max}=1.8$, $R=4$, $B=.45$. The preceding integer fails the water-only sufficient inequality.",
          ["$d_w$","First $m$","$q_m\\in$","$q_{m-1}\\geq$","Margin $\\geq$"],
          [[r["dw"],str(r["first_m"]),"["+decimal_display(r["q_first_lower"],8)+", "+decimal_display(r["q_first_upper"],8,True)+"]",
            decimal_display(r["q_previous_lower"],8),decimal_display(r["water_only_margin_lower"],8)] for r in thresholds])
    selected=[r for r in water if r["m"] in [2,3,4,8] and r["D"]=="4"]
    table("water_domains","Same plant habitat and spectrum, different water domains: $D=4$, $d_w=1$, $d_v=.1$. The margin is $d_v\\underline\\beta_m+B-A_{\\max}R\\tau^+$. A failed sufficient test is inconclusive.",
          ["$m$","Water domain","$\\tau^+\\leq$","Margin $\\geq$","Decision"],
          [[str(r["m"]),r["water_domain"],decimal_display(r["q_upper"],7,True),decimal_display(r["coupled_margin_lower"],7),r["coupled_decision"]] for r in selected])
    s=synergy[0]
    table("water_synergy",'Certified gain from combining the spectral and water bounds for $D=4$, $V=2$, $m=2$, $d_v=.6$, $d_w=1$, $A_{\\max}=1.8$, $R=4$ and $B=.45$, with water drainage at each patch boundary. Each interval encloses the margin in the sufficient test.',
          ["Sufficient test","Margin interval","Decision"],
          [[label,"["+decimal_display(s[left],8)+", "+decimal_display(s[right],8,True)+"]",s[decision]]
           for label,left,right,decision in [
               ("Crude reaction cap","crude_margin_lower","crude_margin_upper","crude_decision"),
               ("Water comparison alone","water_only_margin_lower","water_only_margin_upper","water_only_decision"),
               ("Water and dispersal","coupled_margin_lower","coupled_margin_upper","coupled_decision")]])
    table("water_convergence",'Finite-difference checks of the linear water comparison problem for $(\\ell,d_w)=(1,.1),(2/3,1),(4,1)$. The errors are float64 consistency diagnostics.',
          ["Case","Segments","Maximum-value error","Order"],
          [[str(r["case"]),str(r["segments"]),f'{r["error_float"]:.3e}',"--" if r["observed_order"] is None else f'{r["observed_order"]:.3f}'] for r in fd])
    make_figure(finite,tails,global_rows,water,fd)
    path=ROOT/"output/pdf/shape_spectral_results.csv"
    with path.open(newline="") as stream:
        rows=[r for r in csv.DictReader(stream) if r["experiment"] not in dict(experiments)]
    for name,values in experiments:rows.extend(dict(experiment=name,**r) for r in values)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    summary=json.loads((DATA/"summary.json").read_text())
    summary.update(experiment_rows=len(rows),direct_fragment_rows=len(finite),water_support_rows=len(support),
                   peak_envelope_rows=len(envelopes),global_peak_results=global_rows,
                   water_threshold_results=thresholds,water_synergy_results=synergy,coupled_checks=check_record)
    (DATA/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(result_rows=len(rows),global_peaks=global_rows,water_thresholds=thresholds),indent=2))
    from extension_study import main as extension_study
    extension_study()


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--reuse-direct",help="optional existing full certificate/vector JSON")
    main(parser.parse_args().reuse_direct)
