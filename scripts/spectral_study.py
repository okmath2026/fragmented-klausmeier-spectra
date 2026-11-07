"""Reproducible spectral study: NumPy + ReportLab; no randomness."""
import csv, json, math, platform, os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
from decimal import Decimal, ROUND_FLOOR, ROUND_CEILING
from pathlib import Path
import numpy as np
import reportlab
from reportlab.pdfgen import canvas
from reportlab.lib import colors
ROOT=Path(__file__).resolve().parents[1]
DATA,FIG,TABLE=(ROOT/x for x in ("data","figures","generated"))
for path in (DATA,FIG,TABLE): path.mkdir(exist_ok=True)
PI=math.pi
WIDTHS=[]; RESIDUALS=[]

def gauss(n,a,b):
    x,w=np.polynomial.legendre.leggauss(n)
    return a+(x+1)*(b-a)/2,w*(b-a)/2

def interval(L,n,mid=False):
    if mid: return (-L+(np.arange(n)+.5)*2*L/n)[:,None],np.full(n,2*L/n)
    x,w=gauss(n,-L,L)
    return x[:,None],w

def domain(name,n):
    if name in ("disk","ellipse"):
        r,wr=gauss(n,0,1); t=2*PI*np.arange(4*n)/(4*n)
        rr,tt=np.meshgrid(r,t,indexing="ij")
        p=np.column_stack(((rr*np.cos(tt)).ravel(),(rr*np.sin(tt)).ravel()))
        w=np.repeat(wr*r,len(t))*2*PI/len(t)
        if name=="ellipse": p*= [1.5,1/1.5]
    else:
        a,b=(math.sqrt(PI)/2,math.sqrt(PI)/2)
        if name=="rectangle": a,b=math.sqrt(2*PI)/2,math.sqrt(PI/2)/2
        x,wx=gauss(2*n,-a,a); y,wy=gauss(2*n,-b,b)
        xx,yy=np.meshgrid(x,y,indexing="ij")
        p=np.column_stack((xx.ravel(),yy.ravel())); w=np.outer(wx,wy).ravel()
    assert abs(w.sum()-PI)<2e-13
    return p,w

def kernel(p,kind="gaussian",Q=None,theta=0.):
    d=p[:,None,:]-p[None,:,:]
    if p.shape[1]==1:
        z=d[:,:,0]; G=np.exp(-z*z/2)/math.sqrt(2*PI)
        if kind=="gaussian": return G
        A=np.exp(-math.sqrt(2)*abs(z))/math.sqrt(2)
        return A if kind=="laplace" else (1-theta)*G+theta*A
    q=(d*d).sum(axis=2) if Q is None else np.einsum("...i,ij,...j->...",d,np.linalg.inv(Q),d)
    det=1 if Q is None else np.linalg.det(Q)
    return np.exp(-q/2)/(2*PI*math.sqrt(det))

def perron(S):
    q=np.ones(len(S))/math.sqrt(len(S))
    for _ in range(30000):
        y=S@q; ratios=y/q; width=float(np.ptp(ratios))
        if width<2e-12: break
        q=y/np.linalg.norm(y)
    else: raise RuntimeError("Perron iteration did not converge")
    rho=float(q@y); res=float(np.linalg.norm(y-rho*q))
    WIDTHS.append(width); RESIDUALS.append(res)
    return rho,q,width,res

def solve(p,w,**kw):
    S=np.sqrt(w[:,None]*w[None,:])*kernel(p,**kw)
    rho,q,width,res=perron(S)
    return 1-rho,S,q,width,res

def exact_laplace(L):
    a=math.sqrt(2); lo,hi=0.,PI/(2*L)
    for _ in range(100):
        k=(lo+hi)/2
        if k*math.tan(k*L)<a: lo=k
        else: hi=k
    k=(lo+hi)/2
    return k*k/(a*a+k*k)

def csvfile(name,rows):
    with (DATA/(name+".csv")).open("w",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def table(name,caption,headers,rows):
    s=[r"\begin{table}[htbp]",r"\centering",r"\caption{"+caption+"}",r"\label{tab:"+name+"}",
       r"\begin{tabular}{"+len(headers)*"l"+"}",r"\toprule"," & ".join(headers)+r" \\",r"\midrule"]
    s+=[" & ".join(row)+r" \\" for row in rows]
    s += [r"\bottomrule",r"\end{tabular}",r"\end{table}"]
    (TABLE/(name+".tex")).write_text("\n".join(s)+"\n",encoding="ascii")

def refinement_text(value):
    return "$<2\\times10^{-12}$" if value<2e-12 else f"{value:.2e}"

def decimal_display(value,places,upper=False):
    return str(Decimal(value).quantize(Decimal(1).scaleb(-places),
               rounding=ROUND_CEILING if upper else ROUND_FLOOR))

def scientific_upper(value,digits=3):
    value=Decimal(value)
    quantum=Decimal(1).scaleb(value.adjusted()-digits+1)
    return format(value.quantize(quantum,rounding=ROUND_CEILING),f'.{digits-1}E')

def patch_solve(m,g,n):
    """Equal patches, total length 2; g is the edge-to-edge gap."""
    ell=2/m;x,w=interval(ell/2,n)
    centres=(np.arange(m)-(m-1)/2)*(ell+g)
    p=np.concatenate([x+c for c in centres]);weights=np.tile(w,m)
    b,S,q,cw,res=solve(p,weights)
    Z=np.zeros((m*n,m))
    for j in range(m):Z[j*n:(j+1)*n,j]=np.sqrt(w/ell)
    rho_average=float(np.linalg.eigvalsh(Z.T@S@Z)[-1])
    assert rho_average<=1-b+3e-12
    return b,1-b-rho_average,cw,res

def connectivity_experiments():
    network=[];envelopes=[];LJ=1/math.sqrt(2*PI*math.e)
    for m in [1,2,4,8]:
        for g in [0.,.25,.5,1.,2.]:
            coarse=patch_solve(m,g,24)[0]
            b,ae,cw,res=patch_solve(m,g,48)
            network.append(dict(patches=m,gap=g,total_length=2.,span=2+(m-1)*g,
                                beta=b,average_error=ae,refinement=abs(b-coarse),
                                collatz_width=cw,residual=res))
        vals=[r["beta"] for r in network if r["patches"]==m]
        assert all(y>=x-3e-12 for x,y in zip(vals,vals[1:]))
    baseline=network[0]["beta"]
    assert max(abs(r["beta"]-baseline) for r in network if r["gap"]==0)<5e-12
    for D in [4.,6.]:
        envelope_beta=solve(*interval(D/2,200))[0]
        limit=1-(2/D)*(1-envelope_beta)
        for m in [2,3,4,5,8,16,32,64]:
            g=(D-2)/(m-1)
            coarse=patch_solve(m,g,12)[0]
            b,ae,cw,res=patch_solve(m,g,24)
            bound=4*(D-2)*LJ/m
            assert abs(b-limit)<bound+5e-12
            envelopes.append(dict(span=D,patches=m,gap=g,total_length=2.,beta=b,
                                  limit_beta=limit,limit_error=abs(b-limit),
                                  schur_bound=bound,refinement=abs(b-coarse),
                                  collatz_width=cw,residual=res))
    return network,envelopes

def correction_experiments(envelopes):
    rows=[]
    for D in [4.,6.]:
        p,w=interval(D/2,220);beta,S,q,_,_=solve(p,w)
        rho=1-beta;theta=2/D;delta=D-2;z=p[:,0,None]-p[None,:,0]
        coefficient=theta*(delta/D)*float(q@((z*z*S)@q))
        # Independently evaluate the endpoint trace through the integral equation.
        endpoint=float((np.exp(-.5*(D/2-p[:,0])**2)/math.sqrt(2*PI))@ (np.sqrt(w)*q))/rho
        trace_coefficient=theta*(delta/D)*rho*(1-D*endpoint**2)
        assert abs(coefficient-trace_coefficient)<5e-12 and coefficient>0
        previous=None
        for m in [4,8,16,32,64]:
            r=next(r for r in envelopes if r["span"]==D and r["patches"]==m)
            b=r["beta"];limit=1-theta*rho
            corrected=b-coefficient/m;ce=abs(corrected-limit)
            centred=patch_solve(m,(D-2)/m,24)[0]
            doubled=next((x["beta"] for x in envelopes if x["span"]==D and x["patches"]==2*m),None)
            extrapolated=2*doubled-b if doubled is not None else float("nan")
            rows.append(dict(span=D,patches=m,beta=b,limit_beta=limit,
                             coefficient=coefficient,trace_coefficient=trace_coefficient,
                             scaled_error=m*(b-limit),corrected_beta=corrected,corrected_error=ce,
                             corrected_order=math.log2(previous/ce) if previous else float("nan"),
                             extrapolated_beta=extrapolated,extrapolated_error=abs(extrapolated-limit),
                             centred_beta=centred,centred_span=D-(D-2)/m,centred_error=abs(centred-limit)))
            previous=ce
    return rows

def boundary_experiments(corrections):
    from validated_boundary import boundary_certificate,boundary_remainder,boundary_self_check
    from fragment_fft import fragment_fft,self_check as fft_self_check
    certified=[];vectors=[];remainders=[];checks=[]
    fft_checks=fft_self_check()
    for D in ["4","6"]:
        _,S,_,_,_=solve(*interval(float(D)/2,180))
        eigenvalues=np.linalg.eigvalsh(S)
        gap_float=(2/float(D))*float(eigenvalues[-1]-eigenvalues[-2])
        reference=next(r for r in corrections if r["span"]==float(D))
        for n in [80,160,320,640]:
            _,_,q,_,_=solve(*interval(float(D)/2,n,True))
            q=(q+q[::-1])/2
            qstrings=[str(float(x)) for x in q]
            vector=dict(D=D,N=n,precision=60,q=qstrings)
            record=boundary_certificate(D,n,qstrings)
            print(f'boundary certificate D={D}, N={n}: completed',flush=True)
            assert float(record["c_lower"])<reference["coefficient"]<float(record["c_upper"])
            assert float(record["gamma_lower"])<gap_float<float(record["gamma_upper"])
            certified.append(dict(**record,coefficient_float=reference["coefficient"],gap_float=gap_float))
            vectors.append(vector)
            if n==80:checks.append(boundary_self_check(D,n,qstrings))
        finest=certified[-1];cache={}
        for m in [256,512,1024]:
            if m not in cache:cache[m]=fragment_fft(m,float(D),8)
            if 2*m not in cache:cache[2*m]=fragment_fft(2*m,float(D),8)
            coarse=fragment_fft(m,float(D),4);fine=cache[m]
            record=boundary_remainder(finest,m)
            corrected_error=abs(fine["beta"]-reference["limit_beta"]-float(record["c_midpoint"])/m)
            rich_error=abs(2*cache[2*m]["beta"]-fine["beta"]-reference["limit_beta"])
            assert corrected_error<float(record["midpoint_coefficient_corrected_error_upper"])
            assert rich_error<float(record["richardson_error_upper"])
            assert float(record["beta_m_lower"])<fine["beta"]<float(record["beta_m_upper"])
            remainders.append(dict(**record,beta_fft=fine["beta"],
                                   corrected_error_float=corrected_error,richardson_error_float=rich_error,
                                   refinement_4_8=abs(fine["beta"]-coarse["beta"]),
                                   fft_collatz_width=fine["collatz_width"],fft_residual=fine["residual"],
                                   fft_iterations=fine["iterations"],fft_N=fine["N"]))
    csvfile("boundary_certificates",certified);csvfile("boundary_remainders",remainders)
    (DATA/"boundary_vectors.json").write_text(json.dumps(vectors,indent=2)+"\n",encoding="ascii")
    (DATA/"boundary_checks.json").write_text(json.dumps(dict(interval_checks=checks,fft_checks=fft_checks),indent=2)+"\n",encoding="ascii")
    table("boundary_certificates","Outward-rounded boundary coefficients $c_D=-C_D$ and continuous spectral gaps. Every displayed endpoint is rounded outwards.",
          ["$D$","$N$","$c_-$","$c_+$","$\\gamma_-$","$\\gamma_+$"],
          [[r["D"],str(r["N"]),decimal_display(r["c_lower"],8),decimal_display(r["c_upper"],8,True),
            decimal_display(r["gamma_lower"],8),decimal_display(r["gamma_upper"],8,True)] for r in certified])
    table("boundary_budget",'Error bounds at $N=640$. The norms measure errors in the continuous envelope eigenfunction and its endpoint pairing.',
          ["$D$","$\\varepsilon$","$d$","$E_N$","Functional width","Trace width","Final width"],
          [[r["D"]]+[scientific_upper(r[k]) for k in ["projection_error_upper","direction_error_upper","edge_error_upper",
             "c_functional_width_upper","c_trace_width_upper","c_width_upper"]] for r in certified if r["N"]==640])
    table("boundary_remainders","Certified remainder bounds and matrix-free numerical checks. The coefficient uncertainty is included in the corrected bound.",
          ["$D$","$m$","Corr. bound $\\leq$","Observed error","Extr. bound $\\leq$","Observed error"],
          [[r["D"],str(r["m"]),decimal_display(r["midpoint_coefficient_corrected_error_upper"],8,True),
            f'{r["corrected_error_float"]:.2e}',decimal_display(r["richardson_error_upper"],8,True),
            f'{r["richardson_error_float"]:.2e}'] for r in remainders])
    return certified,remainders,checks

def plot(c,box,title,xlabel,ylabel,series,logx=False,logy=False,xticks=None,legend_location="upper_left",ylim=None):
    x0,y0,w,h=box; left,bottom=x0+75,y0+43;pw,ph=w-93,h-87
    tx=(lambda x:math.log10(x)) if logx else (lambda x:x)
    ty=(lambda x:math.log10(x)) if logy else (lambda x:x)
    xx=[tx(x) for _,xs,ys,_ in series for x in xs];yy=[ty(y) for _,xs,ys,_ in series for y in ys]
    xmin,xmax=min(xx),max(xx); ymin,ymax=min(yy),max(yy);dy=max(ymax-ymin,1e-12)
    ymin-=.08*dy
    ymax+=min(.4,.25*dy) if logy else (.42 if len(series)==3 else .25)*dy
    if ylim is not None:ymin,ymax=ty(ylim[0]),ty(ylim[1])
    if ylabel=="beta1" and not logy:ymax=min(ymax,1.)
    px=lambda x:left+(tx(x)-xmin)/(xmax-xmin or 1)*pw
    py=lambda y:bottom+(ty(y)-ymin)/(ymax-ymin)*ph
    c.setFillColor(colors.black);c.setFont("Helvetica-Bold",11);c.drawString(left,y0+h-15,title)
    c.setStrokeColor(colors.grey);c.setLineWidth(.5);c.rect(left,bottom,pw,ph)
    c.setFont("Helvetica",10)
    for k in range(5):
        x=xmin+(xmax-xmin)*k/4;y=ymin+(ymax-ymin)*k/4
        xp,yp=left+pw*k/4,bottom+ph*k/4
        c.line(left,yp,left-3,yp)
        c.drawRightString(left-6,yp-2,f'{10**y if logy else y:.3g}')
    if xticks is None:
        unique=sorted(set(x for _,xs,ys,_ in series for x in xs))
        xticks=unique if logx and len(unique)<=5 else [10**(xmin+(xmax-xmin)*k/4) if logx else xmin+(xmax-xmin)*k/4 for k in range(5)]
    for value in xticks:
        xp=px(value);c.line(xp,bottom,xp,bottom-3)
        label=str(int(value)) if abs(value)<10000 and float(value).is_integer() else f'{value:.3g}'
        c.drawCentredString(xp,bottom-13,label)
    c.setFont("Helvetica",10);c.drawCentredString(left+pw/2,y0+8,xlabel)
    c.saveState();c.translate(x0+10,bottom+ph/2);c.rotate(90);c.drawCentredString(0,0,ylabel);c.restoreState()
    for j,(label,xs,ys,col) in enumerate(series):
        c.setStrokeColor(col);c.setFillColor(col);c.setLineWidth(1.2)
        p=c.beginPath()
        for k,(x,y) in enumerate(zip(xs,ys)):
            if k==0:p.moveTo(px(x),py(y))
            else:p.lineTo(px(x),py(y))
        c.drawPath(p)
        for x,y in zip(xs,ys):c.circle(px(x),py(y),2,fill=1,stroke=0)
        if legend_location=="above":legend_y=bottom+ph+8
        elif legend_location=="lower_left":legend_y=bottom+8+12*(len(series)-1-j)
        else:legend_y=bottom+ph*.70-12*j if len(series)==4 else bottom+ph-12-12*j
        legend_x=left+5
        if legend_location=="above":legend_x=left+sum(c.stringWidth(s[0],"Helvetica",10)+10 for s in series[:j])
        c.setFont("Helvetica",10);c.drawString(legend_x,legend_y,label)
    c.setFillColor(colors.black)

def boundary_figure(boundary,strict_remainders):
    blue,orange,green=colors.HexColor("#1f77b4"),colors.HexColor("#d95f02"),colors.HexColor("#228833")
    c=canvas.Canvas(str(FIG/"boundary_certification.pdf"),pagesize=(640,510))
    plot(c,(0,255,320,255),"(a) Certified coefficient widths","Midpoint cells N","Coefficient enclosure width",
         [(f'D={D}',[r["N"] for r in boundary if r["D"]==D],[float(r["c_width_upper"]) for r in boundary if r["D"]==D],col)
          for D,col in [("4",blue),("6",orange)]],True,True,legend_location="lower_left")
    for D,box,title in [("4",(320,255,320,255),"(b) Gap certificate, D=4"),("6",(0,0,320,255),"(c) Gap certificate, D=6")]:
        rr=[r for r in boundary if r["D"]==D]
        plot(c,box,title,"Midpoint cells N","Continuous spectral gap",
             [("Lower bound",[r["N"] for r in rr],[float(r["gamma_lower"]) for r in rr],blue),
              ("Upper bound",[r["N"] for r in rr],[float(r["gamma_upper"]) for r in rr],orange),
              ("Float reference",[r["N"] for r in rr],[r["gap_float"] for r in rr],green)],True)
    plot(c,(320,0,320,255),"(d) Certified corrected errors","Patch count m","Corrected error upper bound",
         [(f'D={D}',[r["m"] for r in strict_remainders if r["D"]==D],
           [float(r["midpoint_coefficient_corrected_error_upper"]) for r in strict_remainders if r["D"]==D],col)
          for D,col in [("4",blue),("6",orange)]],True,True,legend_location="lower_left")
    c.save()

def main():
    rows=[];exact=exact_laplace(1.);gref=solve(*interval(1.,160))[0];prev=None
    for n in [20,40,80,160,320]:
        p,w=interval(1.,n,True);b,S,q,cw,res=solve(p,w,kind="laplace");bg=solve(p,w)[0]
        err=abs(b-exact);order=math.log2(prev/err) if prev else float("nan")
        rows.append(dict(N=n,h=2/n,beta=b,exact=exact,error=err,order=order,
                         gaussian_error=abs(bg-gref),schur_bound=4/n,collatz_width=cw,residual=res));prev=err
    csvfile("convergence",rows)
    table("convergence","Midpoint errors on $(-1,1)$; Laplace uses an analytic root, Gaussian a 160-point Gauss reference.",
          ["$N$","$\\beta_h$","Laplace error","Order","Gaussian error"],
          [[str(r["N"]),f'{r["beta"]:.9f}',f'{r["error"]:.2e}',"--" if math.isnan(r["order"]) else f'{r["order"]:.3f}',f'{r["gaussian_error"]:.2e}'] for r in rows])
    assert rows[-1]["error"]<rows[0]["error"]/200
    shapes=[]
    for name in ["disk","square","rectangle","ellipse"]:
        for n in [6,10,16]:
            p,w=domain(name,n);b,S,q,cw,res=solve(p,w)
            shapes.append(dict(shape=name,order=n,N=len(w),area=float(w.sum()),beta=b,collatz_width=cw,residual=res))
        print(name,[r["beta"] for r in shapes if r["shape"]==name],flush=True)
    csvfile("shapes",shapes)
    table("shapes","Boundary-fitted equal-area calculations: every domain has area $\\pi$.",
          ["Shape","$\\beta_6$","$\\beta_{10}$","$\\beta_{16}$","Difference"],
          [[name]+[f'{r["beta"]:.9f}' for r in shapes if r["shape"]==name]+[refinement_text(abs([r["beta"] for r in shapes if r["shape"]==name][-1]-[r["beta"] for r in shapes if r["shape"]==name][-2]))] for name in ["disk","square","rectangle","ellipse"]])
    anis=[]
    for a in [0.,22.5,45.,67.5,90.]:
        for n in [10,16]:
            p,w=domain("disk",n);p*= [math.sqrt(2),1/math.sqrt(2)]
            t=math.radians(a);R=np.array([[math.cos(t),-math.sin(t)],[math.sin(t),math.cos(t)]])
            b,S,q,cw,res=solve(p@R.T,w,Q=np.diag([4.,1.]))
            anis.append(dict(angle=a,order=n,N=len(w),area=float(w.sum()),beta=b,collatz_width=cw,residual=res))
    csvfile("anisotropy",anis)
    table("anisotropy","Rotation of the area-$\\pi$ ellipse with semiaxes $\\sqrt{2},1/\\sqrt{2}$; $\\Sigma=\\operatorname{diag}(4,1)$.",
          ["Angle","$\\beta_{10}$","$\\beta_{16}$","Difference"],
          [[f'{a:g}']+[f'{r["beta"]:.9f}' for r in anis if r["angle"]==a]+[refinement_text(abs([r["beta"] for r in anis if r["angle"]==a][0]-[r["beta"] for r in anis if r["angle"]==a][1]))] for a in [0.,22.5,45.,67.5,90.]])
    frag=[]
    for ell in [1.,.5,.25,.125,.0625]:
        x,w=interval(ell/2,64);p=np.concatenate([x-1,x+1]);ww=np.tile(w,2)
        b,S,q,cw,res=solve(p,ww);Z=np.zeros((len(ww),2))
        Z[:64,0]=np.sqrt(w/ell);Z[64:,1]=np.sqrt(w/ell)
        rhog=float(np.linalg.eigvalsh(Z.T@S@Z)[-1]);rhoc=ell*(1+math.exp(-2))/math.sqrt(2*PI)
        rho=1-b;ec=abs(rho-rhoc);ea=abs(rho-rhog);bound=4*ell**2/math.sqrt(2*PI*math.e)
        assert ec<=bound and rhog<=rho+2e-12
        frag.append(dict(ell=ell,centre_distance=2.,rho=rho,rho_center=rhoc,rho_average=rhog,
                         center_error=ec,average_error=ea,bound=bound,relative_center_error=ec/rho))
    csvfile("fragmentation",frag)
    table("fragmentation","Shrinking intervals at centre distance $2$: full eigenvalue and errors of reduced matrices.",
          ["$\\ell$","Full $\\rho_h$","Centre error","Average error","Schur bound"],
          [[f'{r["ell"]:g}',f'{r["rho"]:.8f}',f'{r["center_error"]:.2e}',f'{r["average_error"]:.2e}',f'{r["bound"]:.2e}'] for r in frag])
    mix=[]
    for n in [320,640]:
        p,w=interval(2.,n,True);ends=[solve(p,w,kind="mixture",theta=t)[0] for t in [0.,1.]]
        for t in [0.,.25,.5,.75,1.]:
            b,S,q,cw,res=solve(p,w,kind="mixture",theta=t);gap=b-((1-t)*ends[0]+t*ends[1])
            assert gap>=-2e-11
            mix.append(dict(N=n,theta=t,beta=b,chord_gap=gap,collatz_width=cw))
    csvfile("mixtures",mix)
    table("mixtures","Gaussian--Laplace mixture on $(-2,2)$ and the gap above its endpoint chord.",
          ["$\\vartheta$","$\\beta_{320}$","$\\beta_{640}$","Gap ($640$)"],
          [[f'{t:.2f}']+[f'{r["beta"]:.9f}' for r in mix if r["theta"]==t]+[f'{[r["chord_gap"] for r in mix if r["theta"]==t][-1]:.2e}'] for t in [0.,.25,.5,.75,1.]])
    scale=[]
    for L in [.025,.05,.1,.2,2.,4.,8.,16.,32.,64.]:
        if L<=.2:b=solve(*interval(L,80))[0]
        else:
            n=max(64,int(L/.1));x=(np.arange(n)+.5)*L/n
            S=(np.exp(-.5*(x[:,None]-x[None,:])**2)+np.exp(-.5*(x[:,None]+x[None,:])**2))/math.sqrt(2*PI)*L/n
            b=1-float(np.linalg.eigvalsh(S)[-1])
        scale.append(dict(L=L,beta=b,rho=1-b,small_ratio=(1-b)/(2*L/math.sqrt(2*PI)),large_ratio=b/(PI**2/(8*L**2))))
    csvfile("scaling",scale)
    from validated_gaussian import gaussian_certificate, arithmetic_audit
    certificates=[];certificate_vectors=[];certificate_cache={}
    for L,n in [("1.5",160),("2",80),("2",160),("2",320),("2",640),("2.2",640),("3",320)]:
        b,S,q,cw,res=solve(*interval(float(L),n,True))
        record=dict(beta_float=b,**gaussian_certificate(L,n,q))
        certificates.append(record);certificate_cache[L,n]=record
        certificate_vectors.append(dict(L=L,N=n,precision=60,
                                        q=[str(Decimal(str(float(value)))) for value in q]))
    csvfile("certificates",certificates)
    quadratic=[]
    for n in [10,20,40,80,160,320,640]:
        if ("2",n) not in certificate_cache:
            b,S,q,cw,res=solve(*interval(2.,n,True))
            certificate_cache["2",n]=dict(beta_float=b,**gaussian_certificate("2",n,q))
            certificate_vectors.append(dict(L="2",N=n,precision=60,q=[str(Decimal(str(float(v)))) for v in q]))
        quadratic.append(certificate_cache["2",n])
    csvfile("quadratic_certificates",quadratic)
    (DATA/"certificate_vectors.json").write_text(json.dumps(certificate_vectors,indent=2)+"\n",encoding="ascii")
    network,envelopes=connectivity_experiments()
    csvfile("connectivity",network);csvfile("fixed_span",envelopes)
    corrections=correction_experiments(envelopes);csvfile("fragment_correction",corrections)
    boundary,strict_remainders,boundary_checks=boundary_experiments(corrections)
    table("certificates","Outward-rounded Gaussian enclosures: $M/d_v=0.15$; margins use $d_v=45$, $M=6.75$.",
          ["$L$","$N$","$E_h^+$","$\\underline\\beta_h$","Margin $\\geq$","Decision"],
          [[r["L"],str(r["N"]),decimal_display(r["cell_bound"],5,True),
            # Floor displayed lower bounds; printing to nearest could overstate them.
            decimal_display(r["beta_lower"],6),
            decimal_display(r["nominal_margin_lower"],4),
            "exclude" if r["decision"]=="excluded" else "undecided"] for r in certificates])
    table("quadratic_certificates","Gaussian $(-2,2)$: rigorous first- and second-order lower bounds and the second-order enclosure width.",
          ["$N$","First lower","Second lower","Width $\\leq$","Ratio","Decision"],
          [[str(r["N"]),decimal_display(r["beta_lower"],6),decimal_display(r["quadratic_beta_lower"],6),
            decimal_display(r["quadratic_width_upper"],8,True),
            "--" if i==0 else f'{float(quadratic[i-1]["quadratic_width_upper"])/float(r["quadratic_width_upper"]):.3f}',
            "exclude" if r["quadratic_decision"]=="excluded" else "undecided"] for i,r in enumerate(quadratic)])
    assert quadratic[1]["quadratic_robust_decision"]=="excluded"
    table("fragment_correction",'First-order boundary correction, Richardson extrapolation and a comparison using centred cells. Errors are relative to $\\beta_\\infty$.',
          ["$D$","$m$","$m(\\beta_m-\\beta_\\infty)$","$c_D$","Corr. error","Extr. error","Centred error"],
          [[f'{r["span"]:g}',str(r["patches"]),f'{r["scaled_error"]:.6f}',f'{r["coefficient"]:.6f}',
            f'{r["corrected_error"]:.2e}',f'{r["extrapolated_error"]:.2e}',f'{r["centred_error"]:.2e}']
           for r in corrections if r["patches"] in [8,16,32]])
    table("connectivity","Fixed total length $2$: equal patches and edge gaps $g$. The last column is a discrete Ritz error at $g=0.5$.",
          ["$m$","$\\beta_h(0)$","$\\beta_h(0.5)$","$\\beta_h(2)$","Average error"],
          [[str(m)]+[f'{next(r["beta"] for r in network if r["patches"]==m and r["gap"]==g):.9f}' for g in [0.,.5,2.]]+
           [f'{next(r["average_error"] for r in network if r["patches"]==m and r["gap"]==.5):.2e}'] for m in [1,2,4,8]])
    table("fixed_span","Fixed total length $2$ and envelope span $D$: numerical patch-count dependence.",
          ["$m$","$\\beta_h$, $D=4$","$\\beta_h$, $D=6$","Error, $D=4$"],
          [[str(m)]+[f'{next(r["beta"] for r in envelopes if r["patches"]==m and r["span"]==D):.9f}' for D in [4.,6.]]+
           [f'{next(r["limit_error"] for r in envelopes if r["patches"]==m and r["span"]==4.):.2e}'] for m in [2,3,4,8,16,32,64]])
    combined=[]
    for experiment,items in [("convergence",rows),("shapes",shapes),("anisotropy",anis),
                              ("fragmentation",frag),("mixtures",mix),("scaling",scale),
                              ("certificates",certificates),("connectivity",network),("fixed_span",envelopes),
                              ("quadratic_certificates",quadratic),("fragment_correction",corrections),
                              ("boundary_certificates",boundary),("boundary_remainders",strict_remainders)]:
        combined.extend([dict(experiment=experiment,**item) for item in items])
    combined_path=ROOT/"output/pdf/shape_spectral_results.csv"
    combined_path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for item in combined for k in item))
    with combined_path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(combined)
    small=[r for r in scale if r["L"]<=.2];large=[r for r in scale if r["L"]>=8.]
    ss=float(np.polyfit(np.log([r["L"] for r in small]),np.log([r["rho"] for r in small]),1)[0])
    sl=float(np.polyfit(np.log([r["L"] for r in large]),np.log([r["beta"] for r in large]),1)[0])
    table("scaling","Small and large Gaussian intervals: ratios to $2LJ(0)$ and $\\pi^2/(8L^2)$.",
          ["$L$","$\\beta_h$","Small ratio","Large ratio"],
          [[f'{r["L"]:g}',f'{r["beta"]:.8f}',f'{r["small_ratio"]:.6f}' if r["L"]<=.2 else "--",f'{r["large_ratio"]:.6f}' if r["L"]>=2 else "--"] for r in scale])
    blue,orange,green=colors.HexColor("#1f77b4"),colors.HexColor("#d95f02"),colors.HexColor("#228833")
    c=canvas.Canvas(str(FIG/"spectral_validation.pdf"),pagesize=(640,510))
    plot(c,(0,255,320,255),"(a) Exact benchmark","Midpoint cells N","Absolute eigenvalue error",
         [("Laplace analytic",[r["N"] for r in rows],[r["error"] for r in rows],blue),("Gaussian Gauss reference",[r["N"] for r in rows],[r["gaussian_error"] for r in rows],orange)],True,True)
    plot(c,(320,255,320,255),"(b) Shrinking fragments","Patch length ell","Absolute error in rho",
         [("Centre matrix",[r["ell"] for r in frag],[r["center_error"] for r in frag],blue),("Averaged Galerkin",[r["ell"] for r in frag],[r["average_error"] for r in frag],orange),("Schur bound",[r["ell"] for r in frag],[r["bound"] for r in frag],green)],True,True)
    plot(c,(0,0,320,255),"(c) Area-exact shapes","Quadrature order n","beta1",
         [(name,[r["order"] for r in shapes if r["shape"]==name],[r["beta"] for r in shapes if r["shape"]==name],col) for name,col in zip(["disk","square","rectangle","ellipse"],[blue,orange,green,colors.purple])])
    plot(c,(320,0,320,255),"(d) Rotation without clipping","Rotation (degrees)","beta1",
         [(f'n={n}',[r["angle"] for r in anis if r["order"]==n],[r["beta"] for r in anis if r["order"]==n],col) for n,col in [(10,blue),(16,orange)]])
    c.save()
    c=canvas.Canvas(str(FIG/"quantitative_refinements.pdf"),pagesize=(640,510))
    plot(c,(0,255,320,255),"(a) Validated enclosure width","Midpoint cells N","Spectral enclosure width",
         [("Cell bound",[r["N"] for r in quadratic],[float(r["beta_upper"])-float(r["beta_lower"]) for r in quadratic],blue),
          ("Projection bound",[r["N"] for r in quadratic],[float(r["quadratic_width_upper"]) for r in quadratic],orange)],True,True,
         xticks=[10,20,40,80,160,320,640],legend_location="lower_left")
    plot(c,(320,255,320,255),"(b) Stationary exclusion","Midpoint cells N","Lower bound on beta1",
         [("Cell bound",[r["N"] for r in quadratic if r["N"]>=20],[float(r["beta_lower"]) for r in quadratic if r["N"]>=20],blue),
          ("Projection bound",[r["N"] for r in quadratic if r["N"]>=20],[float(r["quadratic_beta_lower"]) for r in quadratic if r["N"]>=20],orange),
          ("Threshold",[20,640],[.15,.15],green)],True,xticks=[20,40,80,160,320,640])
    plot(c,(0,0,320,255),"(c) Predicted boundary coefficient","Patch count m","m (beta_m-beta_inf) / c_D",
         [(f'D={D:g}',[r["patches"] for r in corrections if r["span"]==D],
           [r["scaled_error"]/r["coefficient"] for r in corrections if r["span"]==D],col) for D,col in [(4.,blue),(6.,orange)]]+
         [("Predicted limit",[4,64],[1.,1.],green)],True,xticks=[4,8,16,32,64])
    chosen=[r for r in corrections if r["span"]==4. and r["patches"]<=32]
    plot(c,(320,0,320,255),"(d) Bias removal, D=4","Patch count m","Absolute limit error",
         [("Uncorrected",[r["patches"] for r in chosen],[abs(r["beta"]-r["limit_beta"]) for r in chosen],blue),
          ("Coefficient corrected",[r["patches"] for r in chosen],[r["corrected_error"] for r in chosen],orange),
          ("Richardson",[r["patches"] for r in chosen],[r["extrapolated_error"] for r in chosen],green)],True,True,
         legend_location="lower_left")
    c.save()
    boundary_figure(boundary,strict_remainders)
    c=canvas.Canvas(str(FIG/"connectivity_certificates.pdf"),pagesize=(640,510))
    selected=[r for r in certificates if r["L"]=="2"]
    plot(c,(0,255,320,255),"(a) Validated lower bounds","Midpoint cells N","Lower bound on beta1",
         [("Gaussian L=2",[r["N"] for r in selected],[float(r["beta_lower"]) for r in selected],blue),
          ("Exclusion threshold",[r["N"] for r in selected],[.15]*len(selected),orange)],True)
    plot(c,(320,255,320,255),"(b) Gaps at fixed length","Edge gap g","beta1",
         [(f'm={m}',[r["gap"] for r in network if r["patches"]==m],
           [r["beta"] for r in network if r["patches"]==m],col) for m,col in [(2,blue),(4,orange),(8,green)]])
    plot(c,(0,0,320,255),"(c) Fixed envelope span","Patch count m","beta1",
         [(f'D={D:g}',[r["patches"] for r in envelopes if r["span"]==D],
           [r["beta"] for r in envelopes if r["span"]==D],col) for D,col in [(4.,blue),(6.,orange)]],True,xticks=[2,4,8,16,32,64])
    plot(c,(320,0,320,255),"(d) Fine fragmentation limit","Patch count m","Absolute limit error",
         [(f'D={D:g}',[r["patches"] for r in envelopes if r["span"]==D],
           [r["limit_error"] for r in envelopes if r["span"]==D],col) for D,col in [(4.,blue),(6.,orange)]],True,True,xticks=[2,4,8,16,32,64])
    c.save()
    c=canvas.Canvas(str(FIG/"kernel_scaling.pdf"),pagesize=(640,255))
    plot(c,(0,0,320,255),"(a) Large interval limit","Half-width L","beta1",
         [("Computed",[r["L"] for r in scale if r["L"]>=2],[r["beta"] for r in scale if r["L"]>=2],blue),("pi^2/(8 L^2)",[r["L"] for r in scale if r["L"]>=2],[PI**2/(8*r["L"]**2) for r in scale if r["L"]>=2],green)],True,True)
    plot(c,(320,0,320,255),"(b) Kernel mixture","Laplace fraction theta","Concavity gap",
         [(f'N={n}',[r["theta"] for r in mix if r["N"]==n],[max(0,r["chord_gap"]) for r in mix if r["N"]==n],col) for n,col in [(320,blue),(640,orange)]])
    c.save()
    summary=dict(python=platform.python_version(),numpy=np.__version__,reportlab=reportlab.Version,
                 seed="none: deterministic",laplace_exact_L1=exact,max_collatz_width=max(WIDTHS),
                 max_eigen_residual=max(RESIDUALS),small_log_slope=ss,large_log_slope=sl,
                 area_error_max=max(abs(r["area"]-PI) for r in shapes+anis),
                 decimal_audit=arithmetic_audit(),
                 certificates_passed=sum(r["decision"]=="excluded" for r in certificates),
                 network_refinement_max=max(r["refinement"] for r in network+envelopes),
                 quadratic_certificates_passed=sum(r["quadratic_decision"]=="excluded" for r in quadratic),
                 saved_certificate_vectors=len(certificate_vectors),
                 boundary_coefficient_trace_error=max(abs(r["coefficient"]-r["trace_coefficient"]) for r in corrections),
                 strict_boundary_certificate_rows=len(boundary),
                 strict_remainder_rows=len(strict_remainders),boundary_checks=boundary_checks,
                 matrix_free_refinement_max=max(r["refinement_4_8"] for r in strict_remainders),
                 experiment_rows=len(combined))
    (DATA/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    (TABLE/"metrics.tex").write_text("\\newcommand{\\SmallSlope}{%.4f}\n\\newcommand{\\LargeSlope}{%.4f}\n\\newcommand{\\MaxPerronWidth}{%.2e}\n\\newcommand{\\MaxResidual}{%.2e}\n"%(ss,sl,max(WIDTHS),max(RESIDUALS)),encoding="ascii")
    b,S,_,_,_=solve(*interval(1.,40),kind="laplace")
    assert abs(1-b-np.linalg.eigvalsh(S)[-1])<5e-12
    print(json.dumps(summary,indent=2))
    from sharp_study import main as sharpened_study
    sharpened_study()

if __name__=="__main__":main()
