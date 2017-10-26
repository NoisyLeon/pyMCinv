import INIT
import param
import numpy as np
gplf = param.group(numbp=10, flagBs=-1, flag=5, thickness=10., nlay=20, vpvs=1.75)
gplf.ratio=np.random.rand(10)
gplf.value=np.random.rand(10)
gplf.update()

gpcv = INIT.group()
gpcv.np=10
gpcv.flag=5
gpcv.thick=10.
gpcv.value = gplf.value.tolist()
gpcv.ratio = gplf.ratio.tolist()
# INIT.group.updateBs(gpcv)
INIT.group.update(gpcv)

print np.allclose(gpcv.thick1, gplf.thick1)
print np.allclose(gpcv.value1, gplf.value1)