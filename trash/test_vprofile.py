import vmodel
import param
import numpy as np

para = param.para()
para.read('../CODE_NEW/TEST/in.para')

vpr=vmodel.vprofile()
vpr.readmod('../CODE_NEW/TEST/Q22A.mod1')
vpr.update()
vpr.readdisp('../CODE_NEW/TEST/Q22A.com.txt')
vpr.readrf('../CODE_NEW/TEST/in.rf')
vpr.readpara('../CODE_NEW/TEST/in.para')
    
# vpr.compute_disp()
# vpr.compute_rf()
# vpr.compute_misfit()
# ppara = vpr.mod2para()
# vpr.write_model('LF', './')

# for x in xrange(1000):
#     # vpr.compute_disp()
#     vpr.compute_rf()
# vpr.tolist()