"""Register the uniform, pointwise and profile-optimisation refinements."""
import csv
import json
from spectral_study import ROOT, DATA, FIG, plot
from predictor_study import main as predictor_main
from profile_optimum_study import main as profile_main
from hilbert_remainder_study import main as hilbert_main

NAMES=('predictor_residual','predictor_rank_refinement',
       'profile_optimum_diagnostics','hilbert_remainders')


def profile_figure(rows,metadata):
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    blue,orange,green,purple=[colors.HexColor(x) for x in ('#1f77b4','#d95f02','#228833','#7b3294')]
    c=canvas.Canvas(str(FIG/'profile_optimization.pdf'),pagesize=(640,510))
    for D,box,label in [(4,(0,255,320,255),'a'),(6,(320,255,320,255),'b')]:
        scan=[r for r in rows if float(r['D'])==D and float(r['r'])==12
              and r['study_kind']=='local_scan']
        if not scan:
            scan=[r for r in rows if float(r['D'])==D and float(r['r'])==12
                  and int(r['nodes_per_patch'])==8 and int(r['features'])==48]
        scan=sorted(scan,key=lambda x:int(x['m']))
        plot(c,box,f'({label}) Finite-window objectives, D={D}','Patch count m','Objective value',
             [('Full profile',[int(r['m']) for r in scan],[float(r['profile_objective_point']) for r in scan],blue),
              ('Scalar maximum',[int(r['m']) for r in scan],[float(r['scalar_objective_point']) for r in scan],orange)],
             xticks=[30,40,50,60,70,80],legend_location='above')
    scan=sorted([r for r in rows if float(r['D'])==4 and float(r['r'])==12
                 and int(r['nodes_per_patch'])==8 and int(r['features'])==48],key=lambda x:int(x['m']))
    plot(c,(0,0,320,255),'(c) Maximum versus mean water','Patch count m','m squared times water factor',
         [('Maximum',[int(r['m']) for r in scan],[int(r['m'])**2*float(r['water_max_point']) for r in scan],orange),
          ('Mean',[int(r['m']) for r in scan],[int(r['m'])**2*float(r['water_mean_point']) for r in scan],green)],
         xticks=[30,40,50,60,70,80],legend_location='above')
    series=[]
    for D,col1,col2 in [(4,blue,green),(6,orange,purple)]:
        cases=[r for r in metadata['local_winners'] if r['D']==D]
        ratios=[r['r'] for r in cases];limit=cases[0]['predicted_profile_count_scale']
        series.extend([(f'D{D} local',ratios,[r['m_over_r'] for r in cases],col1),
                       (f'D{D} limit',[min(ratios),max(ratios)],[limit,limit],col2)])
    plot(c,(320,0,320,255),'(d) Profile count-scale diagnostics','Reaction / dispersal ratio r','Local count divided by r',
         series,True,False,xticks=[3,6,12,24,48],legend_location='above')
    c.save()


def main():
    predictor_main()
    profile=profile_main([])
    hilbert_main()
    profile_figure(profile['experiments']['profile_optimum_diagnostics'],profile['checks'])
    experiments={}
    for name in NAMES:
        with (DATA/(name+'.csv')).open(newline='',encoding='utf-8') as f:
            experiments[name]=list(csv.DictReader(f))
    path=ROOT/'output/pdf/shape_spectral_results.csv'
    with path.open(newline='',encoding='utf-8') as f:
        combined=[r for r in csv.DictReader(f) if r['experiment'] not in NAMES]
    for name,rows in experiments.items():combined.extend(dict(experiment=name,**r) for r in rows)
    fields=list(dict.fromkeys(k for row in combined for k in row))
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(combined)
    summary=json.loads((DATA/'summary.json').read_text(encoding='utf-8'))
    summary.update(experiment_rows=len(combined),numerical_experiment_count=len({r['experiment'] for r in combined}),
        vector_figure_count=14,predictor_pointwise_error_certified=True,
        profile_global_winner_numerically_certified=False,profile_count_scale_ratio='2/3',
        hilbert_uniform_cubic_bound_certified=True,finite_ratio_localization=False)
    for name,rows in experiments.items():summary[name+'_rows']=len(rows)
    (DATA/'summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='ascii')
    print(json.dumps(dict(result_rows=len(combined),numerical_experiments=summary['numerical_experiment_count'],
                         new_experiments={k:len(v) for k,v in experiments.items()},vector_figures=14)),flush=True)
    return dict(experiments=experiments)


if __name__=='__main__':main()
