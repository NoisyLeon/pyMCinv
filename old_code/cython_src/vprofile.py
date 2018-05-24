# -*- coding: utf-8 -*-
"""
Module for 1D profile inversion

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

import numpy as np
import invsolver
# import matplotlib.pyplot as plt
import matplotlib
import multiprocessing
from functools import partial
import os

# def to_percent(y, position):
#     # Ignore the passed in position. This has the effect of scaling the default
#     # tick locations.
#     s = str(100. * y)
# 
#     # The percent symbol needs escaping in latex
#     if matplotlib.rcParams['text.usetex'] is True:
#         return s + r'$\%$'
#     else:
#         return s + '%'
# 
class vprofile1d(object):
    """
    An object for 1D velocity profile inversion
    =====================================================================================================================
    ::: parameters :::
    indata              - object storing input data
    model               - object storing 1D model
    eigkR, eigkL        - eigenkernel objects storing Rayleigh/Love eigenfunctions and sensitivity kernels
    hArr                - layer array 
    disprefR, disprefL  - flags indicating existence of sensitivity kernels for reference model
    =====================================================================================================================
    """
    def __init__(self):
        self.solver     = invsolver.invsolver1d()
        return
    
    def mc_inv_iso_mp(self, outdir, maxstep=10000, maxsubstep=2000, pfx='MC', dispdtype='ph',\
                    wdisp=1., rffactor=40., monoc=1, nprocess=2):
        if not os.path.isdir(outdir):
         os.makedirs(outdir)
        indLst      = []
        indArr      = np.zeros(3, np.int32)
        
        maxstep     = (np.ceil(maxstep/maxsubstep))*maxsubstep
        for i in range(int(maxstep/maxsubstep)):
            indArr[0]   = i*maxsubstep
            indArr[1]   = (i+1)*maxsubstep
            indArr[2]   = i+1
            tempLst = []
            tempLst.append(indArr.copy())
            solver = invsolver.invsolver1d()
            solver.readdisp('../synthetic_iso_inv/disp_ray.txt')
            solver.readrf('../synthetic_iso_inv/rf.txt')
            
            solver.readmod('../old_code/TEST/Q22A.mod1')
            
            # solver.readpara('../old_code/TEST/in.para')
            solver.getpara()
            solver.update_mod_interface()
            solver.get_vmodel_interface()
            solver.get_period_interface()
            tempLst.append(solver)
            indLst.append(tempLst)
        print 'Start MC inverion for isotropic model, multiprocessing'
        MCINV = partial(mcinviso4mp, outdir=outdir, pfx=pfx, dispdtype=dispdtype,
                 wdisp=wdisp, rffactor=rffactor, monoc=monoc)
        pool = multiprocessing.Pool(processes=nprocess)
        pool.map(MCINV, indLst) 
        pool.close() 
        pool.join()
        
        
def mcinviso4mp(ind, outdir, pfx, dispdtype,
                 wdisp, rffactor, monoc):
    # print iArr[2], iArr[1]
    iArr = ind[0]
    solver = ind[1]
    solver.mc_inv_iso_singel_thread(outdir=outdir, ind0=iArr[0], ind1=iArr[1], indid=iArr[2], pfx=pfx, dispdtype=dispdtype, wdisp=wdisp,
                                    rffactor=rffactor, monoc=monoc)
    return
    
            
    
    