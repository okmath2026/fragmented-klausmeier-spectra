"""Second fragmentation coefficient from the envelope reduced resolvent.

Coefficient/eigenvalue point estimates are FLOAT64 DIAGNOSTICS, not interval
certificates. The independent measure-moment dataset uses exact fractions.
No coefficient is fitted to finite-fragment data.
"""
import csv,json,math,os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import numpy as np
from fragment_fft import fragment_fft
from spectral_study import ROOT,DATA,FIG,csvfile,table,plot


def coefficient_case(D,N,V=2.):
    t,ww=np.polynomial.legendre.leggauss(N)
    x=t*D/2;weights=ww*D/2;sw=np.sqrt(weights)
    J=lambda z:np.exp(-np.asarray(z)**2/2)/math.sqrt(2*math.pi)
    S=sw[:,None]*J(x[:,None]-x[None,:])*sw[None,:]
    eigenvalues,eigenvectors=np.linalg.eigh(S)
    rho=float(eigenvalues[-1]);p=eigenvectors[:,-1]
    if p.sum()<0:p=-p
    u=p/sw;theta=V/D;Delta=D-V;edge=D/2
    ub=float((weights*J(edge-x))@u/rho)
    dub=float((weights*(-(edge-x))*J(edge-x))@u/rho)
    C=theta*(1-theta)*rho*(D*ub*ub-1)
    A1u=(1-theta)*(V/2*ub*(J(x-edge)+J(x+edge))-theta*rho*u)
    forcing=sw*A1u-C*p
    forcing-=p*float(p@forcing)
    rest=eigenvectors[:,:-1]
    z=rest@((rest.T@forcing)/(theta*(rho-eigenvalues[:-1])))
    w=z/sw
    g_edge=(1-theta)*(V/2*ub*(float(J(D))+float(J(0)))-theta*rho*ub)-C*ub
    wb=float((g_edge+theta*(weights*J(edge-x))@w)/(theta*rho))
    kappa=V*D*(1-theta)*(1-2*theta)/12
    direct=theta*C+4*kappa*rho*ub*dub
    resolvent=theta*Delta*rho*ub*wb
    L2=direct+resolvent
    residual=float(np.linalg.norm(theta*rho*z-theta*S@z-forcing))
    c=-C;d=-L2;b=V*V/8
    return dict(D=D,V=V,N=N,rho_float=rho,beta_infty_float=1-theta*rho,
        c_float=c,d_float=d,u_edge_float=ub,u_edge_derivative_float=dub,
        w_edge_float=wb,kappa_float=kappa,lambda2_direct_float=direct,
        lambda2_resolvent_float=resolvent,linear_residual_float=residual,
        orthogonality_float=abs(float(p@z)),
        leading_center_r12_float=2*b*12/c,
        second_center_r12_float=2*(b*12-d)/c,
        diagnostic='float64 envelope quadrature and reduced resolvent; not interval-certified; no fitting')


def figure(convergence):
    from reportlab.pdfgen import canvas
    from reportlab.lib import colors
    blue,orange,green=[colors.HexColor(s) for s in ('#1f77b4','#d95f02','#228833')]
    c=canvas.Canvas(str(FIG/'second_order_fragmentation.pdf'),pagesize=(640,510))
    for D,box in [(4.,(0,255,320,255)),(6.,(320,255,320,255))]:
        rr=[r for r in convergence if r['D']==D and r['nodes_per_patch']==8]
        plot(c,box,f'({"a" if D==4 else "b"}) Second coefficient, D={int(D)}',
            'Patch count m','Scaled first-order residual',
            [('Computed',[r['m'] for r in rr],[r['scaled_second_coefficient_float'] for r in rr],blue),
             ('Resolvent d',[rr[0]['m'],rr[-1]['m']],[rr[0]['d_float']]*2,orange)],
            True,False,xticks=[16,32,64,128,256,512],legend_location='above')
    plot(c,(0,0,320,255),'(c) Second-corrected errors','Patch count m','Absolute error',
        [('D='+str(int(D)),[r['m'] for r in convergence if r['D']==D and r['nodes_per_patch']==8],
          [r['second_corrected_error_float'] for r in convergence if r['D']==D and r['nodes_per_patch']==8],col)
         for D,col in [(4.,blue),(6.,orange)]],True,True,
         xticks=[16,32,64,128,256,512],legend_location='above')
    plot(c,(320,0,320,255),'(d) Cubic residual diagnostic','Patch count m','m^3 times absolute error',
        [('D='+str(int(D)),[r['m'] for r in convergence if r['D']==D and r['nodes_per_patch']==8],
          [r['scaled_cubic_error_float'] for r in convergence if r['D']==D and r['nodes_per_patch']==8],col)
         for D,col in [(4.,blue),(6.,orange)]],True,False,
         xticks=[16,32,64,128,256,512],legend_location='above')
    c.save()


def main():
    from validated_fragment_measure import measure_moment_certificate,fragment_measure_self_check
    from validated_second_budget import second_budget_certificate,second_budget_self_check
    coefficients=[coefficient_case(D,N) for D in (4.,6.) for N in (80,160,320,640)]
    convergence=[]
    for D in (4.,6.):
        coefficient=next(r for r in coefficients if r['D']==D and r['N']==640)
        prev={}
        for m in (16,32,64,128,256,512):
            for nodes in (4,8):
                spectral=fragment_fft(m,D,nodes)
                beta0,c0,d0=(coefficient[k] for k in ('beta_infty_float','c_float','d_float'))
                error1=abs(spectral['beta']-beta0-c0/m)
                error2=abs(spectral['beta']-beta0-c0/m-d0/m**2)
                order=math.log2(prev[nodes]/error2) if nodes in prev else ''
                convergence.append(dict(D=D,V=2,m=m,nodes_per_patch=nodes,
                    beta_float=spectral['beta'],beta_infty_float=beta0,c_float=c0,d_float=d0,
                    first_corrected_error_float=error1,second_corrected_error_float=error2,
                    scaled_second_coefficient_float=m*m*(spectral['beta']-beta0-c0/m),
                    scaled_cubic_error_float=m**3*error2,observed_second_error_order=order,
                    fft_residual_float=spectral['residual'],collatz_width_float=spectral['collatz_width'],
                    diagnostic='float64 quadrature and FFT; resolution diagnostic, not a rigorous remainder bound'))
                prev[nodes]=error2
    measure=[measure_moment_certificate(D,V,m,n)
        for D,V in [('4','2'),('6','2'),('3.7','2.1'),('5','1')]
        for n in (0,2,4,6,8) for m in (2,4,8,16,32,64)]
    def saved(name):
        with (DATA/(name+'.csv')).open(newline='',encoding='utf-8') as stream:
            return list(csv.DictReader(stream))
    envelopes=saved('peak_envelopes');gaps=saved('sharp_gaps');budgets=[];budget_checks=[]
    for D in ('4','6'):
        envelope=next(row for row in envelopes if row['D']==D and row['N']=='640')
        gap=next(row for row in gaps if row['D']==D and row['N']=='640')
        budgets.append(second_budget_certificate(envelope,gap))
        budget_checks.append(second_budget_self_check(envelope,gap))
    experiments=dict(second_order_coefficients=coefficients,second_order_convergence=convergence,
                     weak_measure=measure,second_budgets=budgets)
    for name,rows in experiments.items():csvfile(name,rows)
    table('second_order_coefficients',
        'Second fragmentation coefficients computed from the envelope and corrected count centres at $r=12$. The entries are float64 diagnostics, not interval enclosures. No coefficient is fitted to data from fragmented domains.',
        ['$D$','$N$','$c_D$','$d_D$','$2br/c_D$','$2(br-d_D)/c_D$'],
        [[str(int(r['D'])),str(r['N']),f"{r['c_float']:.9f}",f"{r['d_float']:.9f}",
          f"{r['leading_center_r12_float']:.5f}",f"{r['second_center_r12_float']:.5f}"]
         for r in coefficients if r['N'] in (80,640)])
    table('second_order_convergence',
        'Second-order correction evaluated with eight Gauss points per patch and FFT multiplication. The last column reports an observed order, not a certified convergence rate.',
        ['$D$','$m$','First corr. error','Second corr. error','$m^3$ error','Order'],
        [[str(int(r['D'])),str(r['m']),f"{r['first_corrected_error_float']:.3e}",
          f"{r['second_corrected_error_float']:.3e}",f"{r['scaled_cubic_error_float']:.5f}",
          ('--' if r['observed_second_error_order']=='' else f"{r['observed_second_error_order']:.3f}")]
         for r in convergence if r['nodes_per_patch']==8])
    from spectral_study import scientific_upper
    table('second_budgets',
        'Conservative cubic error bounds from the certified principal eigenvalue and full spectral gap. They do not give a practical count localisation at $r=12$. This calculation does not enclose the signed second coefficient.',
        ['$D$','$M$','$K_3\\leq$','$|d_D|\\leq$'],
        [[r['D'],str(r['anchor_M']),scientific_upper(r['K3_upper']),
          scientific_upper(r['d_abs_upper'])] for r in budgets])
    figure(convergence)
    checks=dict(coefficients_not_fitted=True,coefficient_intervals_certified=False,
        polynomial_cases=len(measure),all_polynomial_bounds=all(r['bound_verified'] for r in measure),
        fragment_measure_self_check=fragment_measure_self_check(),second_budget_checks=budget_checks,
        max_linear_residual=max(r['linear_residual_float'] for r in coefficients),
        max_orthogonality=max(r['orthogonality_float'] for r in coefficients),
        coefficient_resolution_d={str(int(D)):max(abs(r['d_float']-coefficients[3 if D==4 else 7]['d_float'])
            for r in coefficients if r['D']==D) for D in (4.,6.)},
        gauss4_vs8_max=max(abs(a['beta_float']-b['beta_float'])
            for a,b in zip(convergence[::2],convergence[1::2])))
    assert checks['all_polynomial_bounds']
    assert checks['max_linear_residual']<1e-11 and checks['max_orthogonality']<1e-11
    (DATA/'second_order_checks.json').write_text(json.dumps(checks,indent=2)+'\n',encoding='ascii')
    print(json.dumps(dict(rows={k:len(v) for k,v in experiments.items()},checks=checks),indent=2))
    return dict(experiments=experiments,checks=checks)


if __name__=='__main__':main()
