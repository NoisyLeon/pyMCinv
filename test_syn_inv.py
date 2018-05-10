import vprofile
import modparam, copy

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='synthetic_iso_inv/in.disp')
vpr.readrf(infname='synthetic_iso_inv/in.rf')
vpr.readmod(infname='./old_code/TEST/Q22A.mod1')
vpr.getpara()

vpr.mc_joint_inv_iso_mp(nprocess=4, outdir='synthetic_working', pfx='CU.LF', verbose=True, numbrun=150000)