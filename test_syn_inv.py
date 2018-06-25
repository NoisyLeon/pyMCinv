import vprofile
import modparam, copy

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='synthetic_iso_inv/in.disp')
vpr.readrf(infname='synthetic_iso_inv/in.rf')
vpr.readmod(infname='./old_code/weisen_old_code/TEST/Q22A.mod1')
vpr.getpara()

vpr.mc_joint_inv_iso_mp(nprocess=12, outdir='synthetic_working', pfx='CU.LF', verbose=True, numbrun=45000, step4uwalk=1500, wdisp=1.)