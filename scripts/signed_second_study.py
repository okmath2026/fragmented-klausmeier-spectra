"""Strict signed second coefficients and their asymptotic centre expressions."""
import csv,json
from fractions import Fraction
from spectral_study import ROOT,DATA,FIG,csvfile,table,plot,decimal_display,scientific_upper
from signed_second_trial import signed_second_case
from validated_signed_second import (signed_second_self_check,derived_second_centers)

EXPERIMENT_NAMES=('signed_second','signed_second_centers')
CASES=[('4',(20,30,40)),('6',(30,40,50))]


def figure(rows):
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    blue,orange=[colors.HexColor(s) for s in ('#1f77b4','#d95f02')]
    c=canvas.Canvas(str(FIG/'signed_second_certification.pdf'),pagesize=(640,510))
    for box,title,prefix,label in [
        ((0,255,320,255),'(a) Signed second coefficient','d','d_D interval width'),
        ((320,255,320,255),'(b) Boundary eigenfunction','u_edge','u(b) interval width'),
        ((0,0,320,255),'(c) Boundary derivative','u_edge_derivative',"u prime(b) interval width"),
        ((320,0,320,255),'(d) Reduced resolvent energy','energy','Energy interval width')]:
        plot(c,box,title,'Gaussian features p',label,
            [('D='+D,[row['features'] for row in rows if row['D']==D],
              [float(Fraction(row[prefix+'_upper'])-Fraction(row[prefix+'_lower']))
               for row in rows if row['D']==D],col) for D,col in [('4',blue),('6',orange)]],
             False,True,xticks=[20,30,40,50],legend_location='above')
    c.save()


def main():
    rows=[];vectors=[];audits=[];self_checks=[];centers=[]
    with (DATA/'boundary_certificates.csv').open(newline='',encoding='utf-8') as stream:
        boundary=list(csv.DictReader(stream))
    for D,features in CASES:
        for p in features:
            record,audit,trial,diagnostic=signed_second_case(D,p)
            assert record['d_sign_certified_positive'] and record['signed_second_coefficient_certified']
            assert Fraction(record['u_edge_derivative_upper'])<0
            row=dict(**record,**diagnostic);rows.append(row);audits.append(audit)
            vectors.append(dict(D=D,V='2',features=p,precision=60,trial=trial))
            self_checks.append(signed_second_self_check(D,p,trial))
            baseline=next(r for r in boundary if r['D']==D and r['N']=='640')
            assert Fraction(baseline['c_lower'])<=Fraction(record['c_lower'])
            assert Fraction(record['c_upper'])<=Fraction(baseline['c_upper'])
            print(f'signed second D={D},p={p}: d in [{record["d_lower"]}, {record["d_upper"]}]',flush=True)
        centers.append(derived_second_centers(rows[-1]))
    experiments=dict(signed_second=rows,signed_second_centers=centers)
    for name,records in experiments.items():csvfile(name,records)
    checks=dict(self_checks=self_checks,signed_second_coefficient_certified=True,
        all_signed_coefficients_positive=True,complete_feature_coverage=True,
        old_boundary_intervals_contain_new=True,finite_ratio_localization=False)
    for name,value in [('signed_second_vectors',vectors),('signed_second_audits',audits),
                       ('signed_second_checks',checks)]:
        (DATA/(name+'.json')).write_text(json.dumps(value,indent=2)+'\n',encoding='ascii')
    table('signed_second',
        'Certified intervals for the second coefficient of the continuous problem. Displayed endpoints are rounded outwards; widths use the full saved endpoints. All six signs are certified without fitting fragment eigenvalues.',
        ['$D$','$p$','$d_D$ lower','$d_D$ upper','Width $\\leq$'],
        [[r['D'],str(r['features']),decimal_display(r['d_lower'],15),
          decimal_display(r['d_upper'],15,True),scientific_upper(r['d_width_upper'])] for r in rows])
    table('signed_second_endpoints',
        'Certified endpoint values and energy intervals for the reduced equation at the finest tested resolution. Every displayed interval is rounded outwards. The saved records include exact trial vectors and the full residual and arithmetic error bounds.',
        ['$D$','Quantity','Lower','Upper'],
        [[r['D'],label,decimal_display(r[prefix+'_lower'],13),decimal_display(r[prefix+'_upper'],13,True)]
         for r in rows if (r['D'],r['features']) in [('4',40),('6',50)]
         for prefix,label in [('u_edge','$u(b)$'),('u_edge_derivative',"$u'(b)$"),('energy','$\\mathcal E$')]])
    table('signed_second_centers',
        'Certified asymptotic expressions at $r=12$, $d_w=1$. The intervals enclose the indicated continuous expressions. They need not contain the integer optimiser.',
        ['$D$','Offset $2d_D/c_D$','Second centre $2(br-d_D)/c_D$'],
        [[r['D'],'['+decimal_display(r['offset_lower'],9)+', '+decimal_display(r['offset_upper'],9,True)+']',
          '['+decimal_display(r['second_center_lower'],9)+', '+decimal_display(r['second_center_upper'],9,True)+']']
         for r in centers])
    figure(rows)
    result=dict(experiment_rows={k:len(v) for k,v in experiments.items()},
        unique_vectors=len(vectors),checks=checks)
    print(json.dumps(result,indent=2),flush=True)
    return result


if __name__=='__main__':main()
