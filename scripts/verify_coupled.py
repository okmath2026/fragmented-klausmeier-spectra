"""Standard-library replay of water and all-integer spectral-maximum proofs.

The peak proof uses only direct continuous eigenvalue enclosures and a
one-sided Rayleigh tail. Boundary coefficients and quantified eigengaps
are replayed separately for the auxiliary dyadic-ordering corollary.
"""
import csv
import json
from pathlib import Path
from fractions import Fraction
from validated_peaks import (fragment_certificate,rayleigh_envelope_certificate,
    rayleigh_tail_certificate,global_peak_certificate,peak_self_check,rayleigh_self_check)
from validated_water import (rational_water_certificate,coupled_margin,
    synergy_certificate,first_water_only_m,water_self_check)
from check_peak_geometry import independent_geometry_check


def verify_auxiliary_dyadic(root,rows,compare):
    """This older two-sided proof is not an input to the peak certificate."""
    from validated_boundary import boundary_certificate
    from validated_remainder import sharp_eventual_sign
    from validated_peaks import tail_certificate
    vectors=json.loads((root/"data/boundary_vectors.json").read_text())
    baseline=rows("boundary_certificates");dyadic=rows("dyadic_tails")
    for D,threshold in [("4",64),("6",229)]:
        vector=next(v for v in vectors if v["D"]==D and v["N"]==640)
        record=boundary_certificate(D,640,vector["q"],vector["precision"])
        compare(record,next(r for r in baseline if r["D"]==D and int(r["N"])==640),"auxiliary boundary")
        actual=tail_certificate(record,sharp_eventual_sign(record))
        compare(actual,next(r for r in dyadic if r["D"]==D),"auxiliary two-sided dyadic tail")
        assert actual["dyadic_first_sufficient_m"]==threshold
        K=Fraction(actual["uniform_remainder_constant_upper"]);c=Fraction(record["c_lower"])
        assert c-K/(2*threshold-1)-2*K/(threshold-1)>0
    return len(dyadic)


def main():
    root=Path(__file__).resolve().parents[1]
    def rows(name):
        with (root/"data"/(name+".csv")).open(newline="") as stream:
            return list(csv.DictReader(stream))
    def compare(actual,saved,label):
        for key,value in actual.items():assert str(value)==saved[key],(label,key,str(value),saved.get(key))
    vectors=json.loads((root/"data/peak_vectors.json").read_text())
    cache={}
    for vector in vectors:
        key=(vector["D"],vector["m"],vector["cells_per_patch"])
        assert key not in cache
        cache[key]=fragment_certificate(vector["D"],vector["m"],vector["cells_per_patch"],vector["q"],vector["precision"])
    finite=rows("direct_fragments");support=rows("water_support");refinement=rows("peak_refinement")
    for row in finite+support+refinement:
        compare(cache[row["D"],int(row["m"]),int(row["cells_per_patch"])],row,"direct/refined fragment")
    assert len(finite)==24 and len(support)==6 and len(refinement)==8 and len(vectors)==36
    assert {(r["D"],int(r["m"])) for r in finite}==(
        {("4",m) for m in range(2,11)}|{("6",m) for m in range(2,17)})
    assert {(r["D"],int(r["m"])) for r in support}=={("4",m) for m in range(11,17)}
    envelope_vectors=json.loads((root/"data/peak_envelope_vectors.json").read_text())
    envelope_rows=rows("peak_envelopes");tail_rows=rows("peak_tails");global_rows=rows("global_peaks")
    checks=[];rayleigh_checks=[]
    for D,winner,M in [("4",3,11),("6",4,17)]:
        vector=next(v for v in envelope_vectors if v["D"]==D and v["N"]==640)
        envelope=rayleigh_envelope_certificate(D,640,vector["q"],vector["precision"])
        compare(envelope,next(r for r in envelope_rows if r["D"]==D),"principal-only envelope")
        tail=rayleigh_tail_certificate(envelope,M)
        compare(tail,next(r for r in tail_rows if r["D"]==D),"one-sided infinite tail")
        assert tail["uses_quantified_spectral_gap"] is False
        assert tail["uses_boundary_coefficient_interval"] is False
        # Independently check the sign-sensitive substitution using fractions.
        theta=Fraction(2,int(D));kappa=theta*(1-theta)
        factor=theta-kappa/(M-1)
        assert factor>0 and theta-kappa==theta*theta
        exact_tail=1-Fraction(envelope["rho_lower"])*factor+Fraction(envelope["rayleigh_remainder_constant_upper"])/(M*(M-1))
        assert exact_tail<=Fraction(tail["all_m_ge_anchor_beta_upper"])
        proof_rows=[cache[D,m,32 if m==winner else 4] for m in range(2,M)]
        actual=global_peak_certificate(proof_rows,tail,winner)
        compare(actual,next(r for r in global_rows if r["D"]==D),"unique global maximum")
        q=next(v["q"] for v in vectors if v["D"]==D and v["m"]==winner and v["cells_per_patch"]==4)
        checks.append(peak_self_check(D,winner,4,q));checks.append(independent_geometry_check(D,winner,4,q))
        rayleigh_checks.append(rayleigh_self_check(D,640,vector["q"],M))
    threshold_rows=rows("water_thresholds")
    for dw in [".01",".1","1"]:
        compare(first_water_only_m(dw),next(r for r in threshold_rows if r["dw"]==dw),"water threshold")
    water=rows("water_domains")
    for row in water:
        D,m,model=row["D"],int(row["m"]),row["water_domain"]
        assert model in ("component","ambient") and row["V"]=="2"
        length=Fraction(2,m) if model=="component" else Fraction(D)
        distance=Fraction(0) if model=="component" or m%2 else (Fraction(D)-2)/(2*(m-1))
        cap=rational_water_certificate(length,Fraction(1),distance)
        compare(cap,row,"water geometry cap")
        r=32 if m==(3 if D=="4" else 4) else 4
        assert int(row["spectral_cells_per_patch"])==r
        compare(coupled_margin(cap,cache[D,m,r]["beta_lower"]),row,"coupled exclusion margin")
    synergy=rows("water_synergy")
    assert len(synergy)==1
    row=synergy[0]
    assert (row["D"],row["V"],int(row["m"]),row["water_domain"],int(row["spectral_cells_per_patch"]))==("4","2",2,"component",4)
    cap=rational_water_certificate(Fraction(1),Fraction(1));direct=cache["4",2,4]
    compare(cap,row,"joint-gain water cap")
    compare(synergy_certificate(cap,direct["beta_lower"],direct["beta_upper"],".6"),row,"strict joint gain")
    # Negative upper endpoints certify failure of the separate sufficient
    # tests; a positive lower endpoint certifies their joint success.
    assert Fraction(row["water_only_margin_upper"])<0
    assert Fraction(row["crude_margin_upper"])<0
    exact_joint=Fraction(".6")*Fraction(direct["beta_lower"])-Fraction(cap["reaction_M_upper"])
    assert Fraction(".00327021")<Fraction(row["coupled_margin_lower"])<=exact_joint
    assert row["water_only_decision"]==row["crude_decision"]=="inconclusive"
    assert row["coupled_decision"]=="excluded"
    saved=json.loads((root/"data/coupled_checks.json").read_text())
    assert saved["peak_checks"]==checks
    assert saved["water_checks"]==water_self_check()
    assert saved["finite_coverage"]=={"4":[2,10],"6":[2,16]}
    if "rayleigh_checks" in saved:assert saved["rayleigh_checks"]==rayleigh_checks
    auxiliary_rows=verify_auxiliary_dyadic(root,rows,compare)
    print(json.dumps(dict(direct_fragment_rows=len(finite),water_support_rows=len(support),
        winner_refinement_rows=len(refinement),unique_direct_vectors=len(vectors),
        principal_only_envelope_rows=len(envelope_rows),infinite_tail_rows=len(tail_rows),
        global_maxima=len(global_rows),water_thresholds=len(threshold_rows),
        water_domain_rows=len(water),strict_joint_gain_rows=len(synergy),
        auxiliary_two_sided_dyadic_rows=auxiliary_rows,
        full_coordinate_fraction_sum_checks=2,precision_and_context_checks=True,
        gap_free_peak_chain=True,boundary_coefficient_free_peak_chain=True,
        dependencies="Python standard library only; exact saved trial vectors"),indent=2))


if __name__=="__main__":main()
