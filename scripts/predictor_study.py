"""Envelope-predictor residual certificates and finite-count error validation."""
import csv
import json
from fractions import Fraction
from spectral_study import ROOT, DATA, FIG, csvfile, table, plot, decimal_display, scientific_upper
from fragment_predictor_trial import prepare_fragment_predictor_trial
from validated_predictor import predictor_certificate, second_remainder_certificate, predictor_self_check

COUNTS = (16,32,64,70,80,128,256,512,1024,2048)
MAIN_FEATURES = {'4':32, '6':48}
RANK_FEATURES = {'4':(16,24,32), '6':(24,32,48)}
EXPERIMENT_NAMES = ('predictor_residual','predictor_rank_refinement')


def figure(rows, rank_rows):
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    blue,orange,green,purple = [colors.HexColor(x) for x in ('#1f77b4','#d95f02','#228833','#7b3294')]
    c = canvas.Canvas(str(FIG/'predictor_error_certification.pdf'), pagesize=(640,510))
    for D, box, label in [('4',(0,255,320,255),'a'),('6',(320,255,320,255),'b')]:
        rr = [r for r in rows if r['D']==D]
        baseline=[r for r in rr if r['predictor_order']==0]
        refined=[r for r in rr if r['predictor_order']==1]
        plot(c,box,f'({label}) Certified widths, D={D}','Patch count m','Continuous enclosure width',
             [('Envelope only',[r['m'] for r in baseline],[float(r['beta_width_upper']) for r in baseline],blue),
              ('First corrector',[r['m'] for r in refined],[float(r['beta_width_upper']) for r in refined],orange),
              ('Feature tail',[r['m'] for r in refined],[float(r['feature_tail_upper']) for r in refined],green)],
             True,True,xticks=[16,64,256,1024],legend_location='above')
    series=[]
    for D,col1,col2 in [('4',blue,green),('6',orange,purple)]:
        rr=[r for r in rows if r['D']==D and r['predictor_order']==1]
        for suffix,col in [('lower',col1),('upper',col2)]:
            series.append((f'D{D} {suffix}',[r['m'] for r in rr],[float(r['scaled_cubic_'+suffix]) for r in rr],col))
    plot(c,(0,0,320,255),'(c) Certified signed finite-count remainder','Patch count m','m cubed times remainder',
         series,True,False,xticks=[16,64,256,1024],legend_location='above')
    series=[]
    for D,col1,col2 in [('4',blue,green),('6',orange,purple)]:
        rr=[r for r in rank_rows if r['D']==D]
        series.extend([(f'D{D} width',[r['features'] for r in rr],[float(r['beta_width_upper']) for r in rr],col1),
                       (f'D{D} tail',[r['features'] for r in rr],[float(r['feature_tail_upper']) for r in rr],col2)])
    plot(c,(320,0,320,255),'(d) Explicit feature-tail floor','Taylor cutoff p','Width and tail bound',
         series,False,True,xticks=[16,24,32,48],legend_location='above')
    c.save()


def main():
    with (DATA/'signed_second.csv').open(newline='',encoding='utf-8') as f:
        sources=list(csv.DictReader(f))
    signed={D:max((r for r in sources if r['D']==D),key=lambda x:int(x['features'])) for D in MAIN_FEATURES}
    saved=[];trials={};diagnostics={};audits={};cache={}
    for D,features in RANK_FEATURES.items():
        for p in features:
            trial,diagnostic=prepare_fragment_predictor_trial(D,p)
            trials[D,p]=trial;diagnostics[D,p]=diagnostic
            saved.append(dict(D=D,V='2',features=p,trial=trial,diagnostic=diagnostic))
    def calculate(D,m,p,order):
        key=(D,m,p,order)
        if key not in cache:
            row,audit=predictor_certificate(D,m,p,trials[D,p],corrected=bool(order))
            row.update(second_remainder_certificate(row,signed[D]))
            audit['summary']=dict(row)
            cache[key]=row;audits[key]=audit
        return cache[key]
    rows=[]
    for D,p in MAIN_FEATURES.items():
        for m in COUNTS:
            for order in (0,1):
                row=calculate(D,m,p,order)
                if order:assert row['remainder_sign_certified_positive']
                rows.append(row)
            print(f'predictor D={D},m={m}: width {rows[-1]["beta_width_upper"]}',flush=True)
    rank_rows=[]
    for D,features in RANK_FEATURES.items():
        m=80 if D=='4' else 70
        rank_rows.extend(calculate(D,m,p,1) for p in features)
    checks=[predictor_self_check(D,80 if D=='4' else 70,p,trials[D,p],signed[D]) for D,p in MAIN_FEATURES.items()]
    for D,p in MAIN_FEATURES.items():
        for m in COUNTS:
            old=calculate(D,m,p,0);new=calculate(D,m,p,1)
            assert Fraction(new['beta_width_upper'])<Fraction(old['beta_width_upper'])
    for name,values in [('predictor_vectors',saved),('predictor_audits',list(audits.values())),
                        ('predictor_checks',dict(self_checks=checks,all_displayed_main_remainders_positive=True,
                            unique_certificates=len(cache),finite_count_only=True,uniform_third_order_bound_replaced=False))]:
        (DATA/(name+'.json')).write_text(json.dumps(values,indent=2)+'\n',encoding='ascii')
    experiments=dict(predictor_residual=rows,predictor_rank_refinement=rank_rows)
    for name,values in experiments.items():csvfile(name,values)
    selected=[r for r in rows if r['predictor_order']==1 and r['m'] in (32,80 if r['D']=='4' else 70,256,2048)]
    table('predictor_residual',
          'Continuous spectral enclosures from trial vectors constructed on the envelope. The trace and residual give rigorous widths. The last column bounds the signed second-order remainder at the listed counts; it gives no bound outside that set.',
          ['$D$','$m$','$p$','Envelope width','Predictor width','$m^3$ remainder'],
          [[r['D'],str(r['m']),str(r['features']),scientific_upper(calculate(r['D'],r['m'],r['features'],0)['beta_width_upper']),
            scientific_upper(r['beta_width_upper']),
            '['+decimal_display(r['scaled_cubic_lower'],6)+', '+decimal_display(r['scaled_cubic_upper'],6,True)+']'] for r in selected])
    table('predictor_rank_refinement',
          'Contributions of the omitted features and residual at the reference counts. Increasing $p$ reduces the feature truncation error. The residual also depends on the accuracy of the trial vector.',
          ['$D$','$m$','$p$','Quadratic term $\leq$','Feature tail $\leq$','Width $\leq$'],
          [[r['D'],str(r['m']),str(r['features']),scientific_upper(r['quadratic_correction_upper']),
            scientific_upper(r['feature_tail_upper']),scientific_upper(r['beta_width_upper'])] for r in rank_rows])
    figure(rows,rank_rows)
    print(json.dumps(dict(experiments={k:len(v) for k,v in experiments.items()},unique_certificates=len(cache),checks=checks)),flush=True)
    return dict(experiments=experiments,checks=checks)


if __name__=='__main__':main()
