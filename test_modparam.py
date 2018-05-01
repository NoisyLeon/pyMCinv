import modparam
import numpy as np
# 
# para = modparam.para1d()

isomod = modparam.isomod()
# isomod.readmodtxt('./old_code/TEST/Q22A.mod1')

# isomod.init_arr(3)
# isomod.thickness= np.array([0.2, 29.8, 170.])
# 
# 
# isomod.numbp    = np.array([2, 4, 5])
# isomod.mtype    = np.array([4, 2, 2])
# isomod.vpvs     = np.array([2., 1.75, 1.75])
# isomod.update()

import h5py
dset    = h5py.File('./CU_SDT1.0.mod.h5')
data    = dset['118.0_-22.0'].value
isomod.parameterize_input(zarr=data[:, 0], vsarr=data[:, 1], mohodepth=30., seddepth=0.2, maxdepth=200.)

