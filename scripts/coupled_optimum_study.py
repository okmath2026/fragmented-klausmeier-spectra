"""All-integer maxima of a coupled water/seed exclusion margin.

All proof inputs are exact saved decimals/rational geometry. Near-rounded
trial generation affects only tightness; replay uses no floating eigensolver.
"""
import argparse,csv,json
from decimal import Decimal,ROUND_FLOOR
from fractions import Fraction
from spectral_study import ROOT,DATA,FIG,table,csvfile,decimal_display,scientific_upper,plot
from fragment_principal_trial import principal_case
from validated_fragment_principal import (principal_margin_certificate,fragment_principal_self_check)
from validated_water import rational_water_certificate
from validated_coupled_optimum import (coupled_tail_certificate,global_coupled_certificate,
    coupled_optimum_self_check,asymptotic_scale_certificate)

CASES=[('4',80,240,('11.95','12.09'),(20,24,30)),
       ('6',70,200,('11.87','12.03'),(20,30,40))]


def scientific_lower(value,digits=3):
    value=Decimal(value)
    quantum=Decimal(1).scaleb(value.adjusted()-digits+1)
    return format(value.quantize(quantum,rounding=ROUND_FLOOR),f'.{digits-1}E')


def read_rows(name):
    with (DATA/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def figure(coarse,selected,refinement,tails,globals_):
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    blue,orange,green=(colors.HexColor(v) for v in ('#1f77b4','#d95f02','#228833'))
    c=canvas.Canvas(str(FIG/'coupled_optima.pdf'),pagesize=(640,510))
    for D,box,title in [('4',(0,255,320,255),'(a) Coupled global maximum, D=4'),
                         ('6',(320,255,320,255),'(b) Coupled global maximum, D=6')]:
        g=next(r for r in globals_ if r['D']==D);tail=next(r for r in tails if r['D']==D)
        rr=[r for r in selected if r['D']==D and r['m']!=g['winner_m']]
        lower=Fraction(g['winner_margin_lower'])
        gaps=[float(lower-Fraction(r['margin_upper'])) for r in rr]
        assert min(gaps)>0
        M=tail['anchor_m']
        plot(c,box,title,'Patch count m','Separation lower bound',
             [('Finite',[r['m'] for r in rr],gaps,blue),
              ('All remaining m',[M,int(1.25*M)],[float(g['tail_separation_lower'])]*2,orange)],
             False,True,legend_location='above',
             xticks=([2,80,160,240,300] if D=='4' else [2,70,140,200,250]))
    plot(c,(0,0,320,255),'(c) Winner interval refinement','Gaussian features p','Margin interval width',
         [('D='+D,[r['features'] for r in refinement if r['D']==D],
           [float(r['margin_width_upper']) for r in refinement if r['D']==D],col)
          for D,col in [('4',blue),('6',orange)]],False,True,legend_location='above')
    plot(c,(320,0,320,255),'(d) Certified ratio plateaus','(Q / dv) - 12','Envelope span D',
         [('m=80',[float(Fraction(t)-12) for t in CASES[0][3]],[4,4],blue),
          ('m=70',[float(Fraction(t)-12) for t in CASES[1][3]],[6,6],orange)],
         xticks=[-.1,-.05,0,.05],legend_location='above',ylim=(3.5,6.5))
    c.save()


def main():
    cache={};vectors=[];audits=[]
    def build(D,m,p):
        key=(D,m,p)
        if key not in cache:
            record,audit,q,diagnostic=principal_case(D,m,p)
            water=rational_water_certificate(Fraction(2,m),Fraction(1))
            margin=principal_margin_certificate(record,water)
            cache[key]=dict(**record,**margin,Q='7.2',dw='1',q_lower=water['q_lower'],q_upper=water['q_upper'],**diagnostic)
            vectors.append(dict(D=D,V='2',m=m,features=p,precision=60,q=q))
            audits.append(audit)
        return cache[key]
    coarse=[];refinement=[];competitors=[];selected=[];tails=[];global_rows=[];scales=[]
    checks=dict(selection={},targets={},principal_checks=[],global_checks=[],finite_coverage={})
    envelopes=read_rows('peak_envelopes');boundaries=read_rows('boundary_certificates')
    for D,winner,M,target,ranks in CASES:
        for p in ranks:refinement.append(build(D,winner,p))
        best=refinement[-1];proof=[]
        for m in range(2,M):
            row=build(D,m,20);coarse.append(row)
            if m==winner:row=best
            else:
                def passes(candidate):
                    for ratio in map(Fraction,(*target,'12')):
                        wlower=Fraction(best['beta_lower'])-ratio*Fraction(best['q_upper'])
                        cupper=Fraction(candidate['beta_upper'])-ratio*Fraction(candidate['q_lower'])
                        if cupper>=wlower:return False
                    return True
                if not passes(row):
                    for p in (30,40):
                        row=build(D,m,p);competitors.append(row)
                        if passes(row):break
                    if not passes(row):raise ArithmeticError(f'candidate/sector not separated: D={D},m={m}')
            proof.append(row)
            if m%40==0:print(f'coupled finite D={D},m={m}: completed',flush=True)
        envelope=next(r for r in envelopes if r['D']==D)
        tail=coupled_tail_certificate(envelope,M);tails.append(tail)
        result=global_coupled_certificate(proof,tail,winner);global_rows.append(result)
        assert Fraction(result['ratio_lower_required'])<Fraction(target[0])<Fraction(target[1])<Fraction(result['ratio_upper_allowed'])
        selected.extend(proof)
        checks['selection'][D]=[dict(m=r['m'],features=r['features']) for r in proof]
        checks['targets'][D]=list(target);checks['finite_coverage'][D]=[2,M-1]
        checks['global_checks'].append(coupled_optimum_self_check(proof,tail,winner))
        vector=next(v for v in vectors if (v['D'],v['m'],v['features'])==(D,winner,ranks[-1]))
        checks['principal_checks'].append(fragment_principal_self_check(D,winner,ranks[-1],vector['q']))
        boundary=next(r for r in boundaries if r['D']==D and r['N']=='640')
        scales.append(asymptotic_scale_certificate(boundary))
        print(f'coupled optimum D={D}: m={winner}, gap>{result["overall_separation_lower"]}',flush=True)
    experiments={'coupled_candidates':coarse,'coupled_refinement':refinement,'coupled_competitors':competitors,
                 'coupled_tails':tails,'coupled_optima':global_rows,'coupled_scale':scales}
    for name,rows in experiments.items():
        if not rows:raise ArithmeticError('expected a nonempty diagnostic/certificate dataset: '+name)
        csvfile(name,rows)
    for name,record in [('coupled_peak_vectors',vectors),('coupled_peak_audits',audits),('coupled_peak_checks',checks)]:
        (DATA/(name+'.json')).write_text(json.dumps(record,indent=2)+'\n',encoding='ascii')
    table('coupled_optima','Certified unique global maxima of $\\Gamma_m=.6\\beta_m+.45-7.2q_m$ for every integer $m\\geq2$, with $V=2,d_w=1$. Every displayed endpoint is rounded outwards.',
          ['$D$','$m_*$','$\\Gamma_{m_*}\\in$','Finite gap $\\geq$','Tail gap $\\geq$','$M$'],
          [[r['D'],str(r['winner_m']),'['+decimal_display(r['winner_margin_lower'],8)+', '+decimal_display(r['winner_margin_upper'],8,True)+']',
            scientific_lower(r['finite_separation_lower']),scientific_lower(r['tail_separation_lower']),str(r['tail_first_m'])] for r in global_rows])
    table('coupled_ratio_sectors','Closed intervals of the reaction-to-dispersal ratio lying strictly inside the certified open intervals. The kernel, geometry and $d_w=1$ are fixed; $B>0$ does not affect the ordering. The limiting finite counts are those obtained from the full saved interval endpoints.',
          ['$D$','$m_*$','Certified $Q/d_v$','Lower limiting $m$','Upper limiting $m$'],
          [[r['D'],str(r['winner_m']),'['+', '.join(checks['targets'][r['D']])+']',str(r['ratio_lower_limiting_m']),str(r['ratio_upper_limiting_m'])] for r in global_rows])
    table('coupled_refinement','Continuous intervals for the coupled margin at the certified maximisers. Displayed endpoints are rounded outwards; widths use the full saved endpoints. Adding Gaussian features controls the omitted operator. No spectral gap bound or odd-mode certificate is required.',
          ['$D$','$m$','$p$','$\\Gamma_- $','$\\Gamma_+$','Width $\\leq$'],
          [[r['D'],str(r['m']),str(r['features']),decimal_display(r['margin_lower'],9),decimal_display(r['margin_upper'],9,True),scientific_upper(r['margin_width_upper'])] for r in refinement])
    table('coupled_scale','Certified constants in the large-ratio scaling law, obtained from the boundary-coefficient enclosure. These limiting constants do not localise the optimum at a fixed ratio.',
          ['$D$','$c_D\\in$','$\\lim m_r/r\\in$','$\\lim r(\\max F_m-\\beta_\\infty)\\in$'],
          [[r['D'],'['+decimal_display(r['c_lower'],7)+', '+decimal_display(r['c_upper'],7,True)+']',
            '['+decimal_display(r['m_over_ratio_lower'],6)+', '+decimal_display(r['m_over_ratio_upper'],6,True)+']',
            '['+decimal_display(r['scaled_gain_limit_lower'],7)+', '+decimal_display(r['scaled_gain_limit_upper'],7,True)+']'] for r in scales])
    figure(coarse,selected,refinement,tails,global_rows)
    path=ROOT/'output/pdf/shape_spectral_results.csv'
    with path.open(newline='',encoding='utf-8') as stream:
        combined=[r for r in csv.DictReader(stream) if r['experiment'] not in experiments]
    for name,rows in experiments.items():combined.extend(dict(experiment=name,**row) for row in rows)
    fields=list(dict.fromkeys(k for row in combined for k in row))
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(combined)
    summary=json.loads((DATA/'summary.json').read_text())
    summary.update(experiment_rows=len(combined),numerical_experiment_count=len({r['experiment'] for r in combined}),
                   coupled_optimum_checks=checks,coupled_global_results=global_rows,
                   coupled_unique_vectors=len(vectors),coupled_refined_competitors=len(competitors))
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='ascii')
    print(json.dumps(dict(result_rows=len(combined),unique_vectors=len(vectors),
        experiment_rows={name:len(rows) for name,rows in experiments.items()},global_results=global_rows),indent=2),flush=True)
    from refined_study import main as refined_main
    refined_main()


if __name__=='__main__':main()
