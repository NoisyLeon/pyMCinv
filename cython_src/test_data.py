import data
import numpy as np
import multiprocessing
from functools import partial
disp = data.disp()
disp.readdisptxt('../disp_-112.0_36.0_ray.txt')
disp.pvelp  = np.float32(np.random.randn(20)) + disp.pvelo
disp.readaziphitxt('../aziphi_-112.0_36.0.txt')
disp.pphip  = np.float32(np.random.randn(20)) + disp.pphio
disp.readaziamptxt('../aziamp_-112.0_36.0.txt')
disp.pampp  = np.float32(np.random.randn(20)) + disp.pampo


rf = data.rf()
rf.readrftxt('../old_code/TEST/in.rf')

data1d = data.data1d()
data1d.rfr.readrftxt('../old_code/TEST/in.rf')

data1d.dispR.readdisptxt('../disp_-112.0_36.0_ray.txt')

rf.tp= rf.to[:]
rf.rfp= rf.rfo[:]

# 
# import time
# def testmp(i, rf):
#     print i
#     rf.get_misfit_incompatible()
#     
# ilst = (np.arange(10000000)).tolist()
# TESTmp = partial(testmp, rf=rf)
# pool = multiprocessing.Pool(processes=4)
# pool.map(TESTmp, ilst) #make our results with a map call
# pool.close() #we are not adding any more processes
# pool.join() #tell it to wait until all threads are done before going on


