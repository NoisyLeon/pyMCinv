#
# test the fitness from real model parameters
#

import vprofile
import modparam, copy
import numpy as np
vpr = vprofile.vprofile1d()
vpr.readdisp(infname='synthetic_iso_inv/in.disp')
vpr.readrf(infname='synthetic_iso_inv/in.rf')
vpr.readmod(infname='./old_code/weisen_old_code/TEST/Q22A.mod1')
vpr.getpara()

vpr.get_period()
vpr.update_mod()
vpr.get_vmodel()
vpr.model.isomod.mod2para()


real_paraval = np.loadtxt('synthetic_iso_inv/real_para.txt')
vpr.model.get_para_model(paraval=real_paraval)

vpr.compute_fsurf()
vpr.compute_rftheo()