import multiprocessing 
from time import sleep
import numpy as np
from functools import partial
import vprofile
import time

def mp4mc_tti_inv(monoc, stime, vpr):
# initializations
    vpr.get_period(dtype = 'ph')
    vpr.update_mod(mtype = 'tti')
    vpr.model.ttimod.get_rho()
    # vpr.get_vmodel(mtype = 'tti')
    # # initial run
    # if not vpr.compute_tcps(wtype='ray'):
    #     raise ValueError('Error in computing reference Rayleigh dispersion for initial model!')
    # if not vpr.compute_tcps(wtype='love'):
    #     raise ValueError('Error in computing reference Love dispersion for initial model!')
    # vpr.perturb_from_kernel(wtype='ray')
    # vpr.perturb_from_kernel(wtype='love')
    # print vpr.counter
    # vpr.get_misfit_tti()
    
vprlst = []
for i in xrange(10):
    vpr = vprofile.vprofile1d()
    vpr.readdisp(infname='./synthetic_inv/disp_lov.txt', wtype = 'l')
    vpr.readdisp(infname='./synthetic_inv/disp_ray.txt', wtype = 'r')
    vpr.readaziamp(infname='./synthetic_inv/aziamp.ray.txt', wtype = 'r')
    vpr.readaziphi(infname='./synthetic_inv/aziphi.ray.txt', wtype = 'r')
    
    vpr.readmod(infname='mod_-112.0.36.0.mod', mtype='tti')
    vpr.getpara(mtype='tti')
    vpr.counter = i
    vprlst.append(vpr)
    
MCINV   = partial(mp4mc_tti_inv, monoc=True, stime=time.time())
pool    = multiprocessing.Pool(processes=5)
pool.map(MCINV, vprlst) #make our results with a map call
pool.close() #we are not adding any more processes
pool.join() #tell it to wait un
    
    
