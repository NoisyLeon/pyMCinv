import generate_Bs
import bspline_basis
import scipy.interpolate
import numpy as np
import scipy.signal
# nBs, order, 0, group1.thick, 2., group1.nlay
# cvbs=generate_Bs.cv_B_spline(nBs=5, degBs=4, zmin_Bs=0, zmax_Bs=10., disfacBs=2., npts=21)

t = bspline_basis.Bspline(nBs=5, degBs=4, zmin_Bs=0, zmax_Bs=10., disfacBs=2., npts=21)

# lfbs=scipy.interpolate.BSpline(t=np.arange(10), c=np.ones(10), k=4)
# aa=lfbs.basic_element()
# lfbs=scipy.signal.bspline(x=np.arange(10)+1, n=2)

