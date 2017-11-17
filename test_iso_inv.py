import vprofile
import modparam

vpr = vprofile.vprofile1d()
# vpr.readdisp(infname='old_code/TEST/Q22A.com.txt')
# vpr.readrf(infname='old_code/TEST/in.rf')

vpr.readdisp(infname='synthetic_iso_inv/disp_ray.txt')
vpr.readrf(infname='synthetic_iso_inv/rf.txt')

vpr.readmod(infname='old_code/TEST/Q22A.mod1')
vpr.readpara(infname='old_code/TEST/in.para')
vpr.mc_inv_iso('/work3/leon/mc_inv/synthetic_iso_inv_result_wdisp_0.4', wdisp=0.4)
