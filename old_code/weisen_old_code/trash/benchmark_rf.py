import INIT
import param
import numpy as np

rflf=param.rf()
inArr = np.loadtxt('in.rf')
rflf.tn=inArr[:,0]
rflf.rfn=inArr[:,1]
rflf.stretch()

rfcv=INIT.rf()
inArr = np.loadtxt('in.rf')
rfcv.tn=inArr[:,0]
rfcv.rfn=inArr[:,1]

INIT.rf.stretch(rfcv)

print np.allclose(rfcv.tnn, rflf.tnn)
print np.allclose(rfcv.rfnn, rflf.rfnn)