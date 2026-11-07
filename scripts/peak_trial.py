"""Generate positive exact trial strings for direct fragment certificates.

NumPy is used only to prepare proof inputs. Their accuracy is not assumed by
the independent standard-library interval calculation in validated_peaks.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
os.environ.setdefault("OMP_NUM_THREADS","1")
import math
import numpy as np


def fragment_trial(D,m,r=4,V=2.):
    D=float(D);ell=V/m;h=ell/r;gap=(D-V)/(m-1)
    centres=(np.arange(m)-(m-1)/2)*(ell+gap)
    local=-ell/2+(np.arange(r)+.5)*h
    x=(centres[:,None]+local[None,:]).ravel()
    z=x[:,None]-x[None,:]
    S=h*np.exp(-z*z/2)/math.sqrt(2*math.pi)
    q=np.ones(len(S))/math.sqrt(len(S))
    for iteration in range(30000):
        y=S@q;ratios=y/q
        if float(np.ptp(ratios))<2e-14:break
        q=y/np.linalg.norm(y)
    else:raise ArithmeticError("positive trial iteration did not converge")
    q=(q+q[::-1])/2
    strings=[str(float(value)) for value in q]
    assert strings==strings[::-1] and min(q)>0
    rho=float(q@(S@q)/(q@q))
    return strings,dict(beta_midpoint_float=1-rho,
                        collatz_width_float=float(np.ptp((S@q)/q)),
                        trial_iterations=iteration+1)
