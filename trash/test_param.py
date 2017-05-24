import param
import numpy as np
gp = param.group(numbp=10, flagBs=-1, flag=2, thickness=10., nlay=20, vpvs=1.75)
# gp1=param.group(flag=1, numbp=1)
a=np.arange(10)
# gp.ttt(a)
gp.update2()