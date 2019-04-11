# -*- coding: utf-8 -*-
import surfdbase
import copy
import matplotlib.pyplot as plt
import numpy as np

dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190320_no_ocsi_vti.h5')
    

# 
# vpr1 = dset.mc_inv_vti(use_ref=True, outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190325_150000_vti',
#                 numbrun=150000, nprocess=35, verbose=False, group=False, outlon=-142., outlat = 60.)
vpr = dset.mc_inv_vti(use_ref=True, outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190325_150000_vti',
                numbrun=150000, nprocess=35, verbose=False, group=False, outlon=-150., outlat = 65.)
dset.close()
vpr.get_period()
vpr.update_mod(mtype = 'vti')
vpr.get_vmodel(mtype = 'vti')
vpr.model.vtimod.mod2para()
vpr.get_period()
vpr.compute_reference_vti(wtype='ray')
vpr.compute_reference_vti(wtype='lov')

oldvpr      = copy.deepcopy(vpr)
N           = 1000
maxdiffR    = np.zeros(N)
maxdiffR2   = np.zeros(N)
maxdA       = np.zeros(N)
maxdC       = np.zeros(N)
maxdF       = np.zeros(N)
maxdL       = np.zeros(N)
maxdN       = np.zeros(N)
maxdiffL    = np.zeros(N)
maxdiffL2   = np.zeros(N)
maxdphR     = np.zeros(N)
maxdphL     = np.zeros(N)

Nr          = vpr.data.dispR.pvelp.size
Nl          = vpr.data.dispL.pvelp.size
CR          = np.zeros((Nr, N))
CL          = np.zeros((Nl, N)) 
# crmin    = vpr.data.dispR.pvelo.min() - 0.5
# crmax    = vpr.data.dispR.pvelo.max() + 0.5
# clmin    = vpr.data.dispL.pvelo.min() - 0.5
# clmax    = vpr.data.dispL.pvelo.max() + 0.5
for i in range(N):
    print i
    vpr.model.vtimod.para.new_paraval(0)
    vpr.model.vtimod.para2mod()
    vpr.model.vtimod.update()
    while not vpr.model.vtimod.isgood(0, 1, 1, 0):
        vpr.model.vtimod.para.new_paraval(0)
        vpr.model.vtimod.para2mod()
        vpr.model.vtimod.update()
        
    vpr.model.get_vti_vmodel()
    newvpr = copy.deepcopy(vpr)
    newvpr2 = copy.deepcopy(vpr)
    vpr.perturb_from_kernel_vti(wtype='ray')
    vpr.perturb_from_kernel_vti(wtype='lov')
    newvpr2.compute_fsurf(wtype='ray')
    newvpr2.compute_fsurf(wtype='lov')
    crmin   = newvpr2.data.dispR.pvelp.min() - 0.1
    crmax   = newvpr2.data.dispR.pvelp.max() + 0.1
    clmin   = newvpr2.data.dispL.pvelp.min() - 0.1
    clmax   = newvpr2.data.dispL.pvelp.max() + 0.1
    newvpr.compute_reference_vti(wtype='ray', cmin=crmin, cmax=crmax)
    newvpr.compute_reference_vti(wtype='lov', cmin=clmin, cmax=clmax)
    # while newvpr.data.dispR.pvelp
    
    
    if np.any(newvpr.data.dispR.pvelp == 0.) or np.any(newvpr.data.dispL.pvelp == 0.):
        newvpr.compute_reference_vti(wtype='ray', cmin=crmin, cmax=crmax)
        newvpr.compute_reference_vti(wtype='lov', cmin=clmin, cmax=clmax)
    if np.any(newvpr.data.dispR.pvelp == 0.) or np.any(newvpr.data.dispL.pvelp == 0.):
        break
    maxdiffR[i]     = (abs(newvpr.data.dispR.pvelp - vpr.data.dispR.pvelp)/newvpr.data.dispR.pvelp).max()
    
    maxdiffR2[i]    = (abs(newvpr.data.dispR.pvelp - newvpr2.data.dispR.pvelp)/newvpr.data.dispR.pvelp).max()
    
    maxdiffL[i]     = (abs(newvpr.data.dispL.pvelp - vpr.data.dispL.pvelp)/newvpr.data.dispL.pvelp).max()
    
    maxdiffL2[i]    = (abs(newvpr.data.dispL.pvelp - newvpr2.data.dispL.pvelp)/newvpr.data.dispL.pvelp).max()
    CR[:, i]        =  newvpr.data.dispR.pvelp
    CL[:, i]        =  newvpr.data.dispL.pvelp
    maxdphR[i]      = (abs(vpr.data.dispR.pvelp - oldvpr.data.dispR.pvelp)/newvpr.data.dispR.pvelp).max()
    maxdphL[i]      = (abs(vpr.data.dispL.pvelp - oldvpr.data.dispL.pvelp)/newvpr.data.dispL.pvelp).max()
    
