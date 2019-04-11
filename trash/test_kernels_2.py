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
vpr.compute_reference_vti(wtype='ray')
vpr.compute_reference_vti(wtype='lov')
# vpr.model.A += 0.05 * vpr.model.A
vpr.model.C += 0.02 * vpr.model.C
# vpr.model.F += 0.05 * vpr.model.F
vpr.model.L += 0.02 * vpr.model.L
vpr.model.N += 0.05 * vpr.model.N
vpr.model.love2vel()

refv = vpr.data.dispL.pvelref
# vpr.perturb_from_kernel_vti(ivellove=1)
vpr.perturb_from_kernel_vti('lov', ivellove=1)
perv1 = vpr.data.dispL.pvelp
vpr.perturb_from_kernel_vti('lov',ivellove=2)
perv2 = vpr.data.dispL.pvelp

# c1 = vpr.data.dispR.pvelp
vpr.compute_disp_vti(solver_type=1)
# c2 = vpr.data.dispR.pvelp
# plt.plot(c1, 'ro')
# plt.plot(c2, 'ko')