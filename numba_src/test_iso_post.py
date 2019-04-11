import vprofile
import modparam
import numpy as np


vpr = vprofile.vprofile1d()
# vpr.readdisp(infname='old_code/TEST/Q22A.com.txt')
# vpr.readrf(infname='old_code/TEST/in.rf')

vpr.readdisp(infname='synthetic_iso_inv/disp_ray.txt')
vpr.readrf(infname='synthetic_iso_inv/rf.txt')

vpr.readmod(infname='old_code/TEST/Q22A.mod1')
vpr.readpara(infname='old_code/TEST/in.para')
# vpr.mc_inv_iso('synthetic_iso_inv_result')

# 
vpr.read_iso_inv(indir='/work3/leon/mc_inv/synthetic_iso_inv_result_wdisp_0.2')
vpr.read_paraval('synthetic_iso_inv/paraval.txt')

vpr.get_iso_acc_ind(threshhold=2.0)

paraval0=np.loadtxt('synthetic_iso_inv/paraval_ref.txt')

# vpr.read_iso_disp(indir='/work3/leon/mc_inv/synthetic_iso_inv_result_wdisp_0.5')
# vpr.read_iso_rf(indir='/work3/leon/mc_inv/synthetic_iso_inv_result_wdisp_0.5')
# vpr.read_iso_mod(indir='/work3/leon/mc_inv/synthetic_iso_inv_result_wdisp_0.5')


# vpr.read_iso_inv(indir='/work3/leon/mc_inv/synthetic_iso_inv_result_only_surf')
# vpr.read_paraval('synthetic_iso_inv/paraval.txt')
# 
# vpr.get_iso_acc_ind(threshhold=5.0)
# vpr.read_iso_disp(indir='/work3/leon/mc_inv/synthetic_iso_inv_result_only_surf')
# vpr.read_iso_rf(indir='/work3/leon/mc_inv/synthetic_iso_inv_result_only_surf')
# vpr.read_iso_mod(indir='/work3/leon/mc_inv/synthetic_iso_inv_result_only_surf')