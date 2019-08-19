import numpy as np

Nr      = int(1e7)
Ndata   = 123
Psmall  = np.zeros(Nr)
for i in range(Nr):
    psi1    = np.random.rand(Ndata) * 180.
    psi2    = np.random.rand(Ndata) * 180.
    dpsi    = abs(psi1 - psi2)
    dpsi[dpsi>90.]  = 180. -  dpsi[dpsi>90.]
    Psmall[i] = np.where(dpsi < 30.)[0].size/float(Ndata)