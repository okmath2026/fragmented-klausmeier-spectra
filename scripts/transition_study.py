"""Certified local count transitions of the scalar coupled exclusion margin.

Writes only this extension's datasets/tables/figure. The parent numerical
entry point merges its CSV rows after all research extensions have completed.
"""
import csv,json
from fractions import Fraction
from pathlib import Path
from fragment_principal_trial import principal_case
from validated_fragment_principal import fragment_principal_self_check
from validated_water import rational_water_certificate
from validated_transitions import global_transition_certificate,transition_self_check
from spectral_study import ROOT,DATA,FIG,csvfile,table,plot,decimal_display,scientific_upper
from coupled_optimum_study import scientific_lower

CASES=[('4',(79,80,81),('11.9','12.15'),40),
       ('6',(69,70,71),('11.8','12.1'),50)]
EXPERIMENT_NAMES=('transition_candidates','transition_crossings','transition_global')


def _rows(name):
    with (DATA/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def _figure(candidates):
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    c=canvas.Canvas(str(FIG/'coupled_transitions.pdf'),pagesize=(640,275))
    for offset,(D,counts,window,p) in enumerate(CASES):
        rr={int(r['m']):r for r in candidates if r['D']==D}
        centre=rr[counts[1]]
        values=[Fraction(window[0]),(Fraction(window[0])+Fraction(window[1]))/2,Fraction(window[1])]
        series=[]
        for m,col in zip(counts,('#1f77b4','#228833','#d95f02')):
            row=rr[m];ys=[]
            for ratio in values:
                if m==counts[1]:ys.append(0.0);continue
                lower=Fraction(row['beta_lower'])-Fraction(centre['beta_upper'])-ratio*(Fraction(row['q_upper'])-Fraction(centre['q_lower']))
                upper=Fraction(row['beta_upper'])-Fraction(centre['beta_lower'])-ratio*(Fraction(row['q_lower'])-Fraction(centre['q_upper']))
                assert lower<=upper
                ys.append(float((lower+upper)/2))
            series.append(('m='+str(m),list(map(float,values)),ys,colors.HexColor(col)))
        plot(c,(offset*320,0,320,275),f'({chr(97+offset)}) Local all-count transitions, D={D}',
             'Reaction / dispersal ratio r','F(m,r) - F(middle,r)',series,
             xticks=([11.9,12,12.1] if D=='4' else [11.8,11.9,12,12.1]),
             legend_location='above',ylim=((-5e-7,5e-7) if D=='4' else (-8e-7,8e-7)))
    c.save()


def main():
    oldrows=_rows('coupled_candidates')+_rows('coupled_refinement')+_rows('coupled_competitors')
    lookup={(r['D'],int(r['m']),int(r['features'])):r for r in oldrows}
    oldchecks=json.loads((DATA/'coupled_peak_checks.json').read_text())
    tails=_rows('coupled_tails')
    candidates=[];crossings=[];global_rows=[];vectors=[];audits=[];global_audits=[]
    checks=dict(selection={},windows={},candidate_checks=[],global_checks=[],finite_coverage={})
    for D,counts,window,p in CASES:
        proof=[];new={}
        for m in counts:
            record,audit,q,diagnostic=principal_case(D,m,p)
            water=rational_water_certificate(Fraction(2,m),Fraction(1))
            row=dict(**record,dw='1',q_lower=water['q_lower'],q_upper=water['q_upper'],**diagnostic)
            candidates.append(row);new[m]=row;audits.append(audit)
            vectors.append(dict(D=D,V='2',m=m,features=p,precision=60,q=q))
            checks['candidate_checks'].append(fragment_principal_self_check(D,m,p,q))
        selection=[]
        for entry in oldchecks['selection'][D]:
            m=entry['m']
            if m in new:
                row=new[m];source='transition'
            else:
                row=dict(lookup[(D,m,entry['features'])]);row['dw']='1';source='coupled'
            proof.append(row)
            selection.append(dict(source=source,m=m,features=int(row['features'])))
        tail=next(r for r in tails if r['D']==D)
        summary,audit=global_transition_certificate(proof,tail,counts,window)
        crossings.extend(audit['pairs']);global_rows.append(summary);global_audits.append(audit)
        checks['global_checks'].append(transition_self_check(proof,tail,counts,window))
        checks['selection'][D]=selection;checks['windows'][D]=list(window)
        checks['finite_coverage'][D]=[2,int(tail['anchor_m'])-1]
        print(f'transitions D={D}: complete global phase {counts}, ratio window {window}',flush=True)
    experiments=dict(transition_candidates=candidates,transition_crossings=crossings,transition_global=global_rows)
    for name,rows in experiments.items():csvfile(name,rows)
    for name,value in [('transition_vectors',vectors),('transition_audits',audits),
                       ('transition_global_audits',global_audits),('transition_checks',checks)]:
        (DATA/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='ascii')
    table('transition_crossings','Certified switching ratios between consecutive patch counts. Displayed endpoints are rounded outwards; width bounds use the full saved intervals. Each switch is a transition between global maximisers within the verified ratio window.',
        ['$D$','Counts','True switching ratio $s\\in$','Width $\\leq$'],
        [[r['D'],str(r['left_m'])+' / '+str(r['right_m']),
          '['+decimal_display(r['crossing_lower'],16)+', '+decimal_display(r['crossing_upper'],16,True)+']',
          scientific_upper(r['crossing_width_upper'])] for r in crossings])
    table('transition_phases','Global maximising counts within the stated ratio windows for $F_m(r)=\\beta_m-rq_m$, with $V=2,d_w=1$. As $r$ increases, the three counts are uniquely optimal in the listed order. Exactly the two adjoining counts tie at each switch. The gaps bound normalised differences from all other counts throughout the window.',
        ['$D$','Ratio window','Winning counts','Small gap $\\geq$','Large gap $\\geq$','Tail gap $\\geq$'],
        [[r['D'],'['+r['ratio_window_lower']+', '+r['ratio_window_upper']+']',
          '/'.join(str(r[k]) for k in ('first_candidate','middle_candidate','last_candidate')),
          scientific_lower(r['minimum_smaller_gap_lower']),scientific_lower(r['minimum_larger_gap_lower']),
          scientific_lower(r['infinite_tail_gap_lower'])] for r in global_rows])
    _figure(candidates)
    result=dict(experiment_rows={name:len(rows) for name,rows in experiments.items()},
        unique_new_vectors=len(vectors),finite_cases_reused=436-6,all_integer_phase_diagrams=True,
        global_results=global_rows)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':main()
