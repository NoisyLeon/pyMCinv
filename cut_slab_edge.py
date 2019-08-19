import numpy as np

inarr   = np.loadtxt('SlabE325.dat')

mindep  = 5.
maxdep  = 200.

outfname= 'SlabE325_'+str(int(mindep))+'_'+str(int(maxdep))+'.dat'

depth   = -inarr[:, 2]
index   = (depth>=mindep)*(depth<=maxdep)

outarr  = inarr[index, :]

np.savetxt(outfname, outarr, fmt='%g')