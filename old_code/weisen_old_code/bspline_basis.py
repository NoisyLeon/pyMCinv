# generating cubic B-splines
# read the B-spline number, ratio, points
# read the uper bound and lower bound
# return a 1-D array containing basis B-splines
from __future__ import absolute_import
from numba import jit, float32, int32
import numpy as np
import math

@jit(float32[:,:](int32, int32, int32, int32, int32, int32))
def Bspline(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts):
    """
    Compute cubic B-spline basis
    
    """
    Bs_basis    = np.array([])
    m           = nBs-1+degBs
    t           = np.zeros(m+1)
    for i in range (degBs):
        temp = zmin_Bs + i*(zmax_Bs-zmin_Bs)/10000.
        t[i] = temp
    for i in range (degBs,m+1-degBs):
        n_temp = m+1-degBs-degBs+1
        if (disfacBs !=1):
            temp = (zmax_Bs-zmin_Bs)*(disfacBs-1)/(math.pow(disfacBs,n_temp)-1)
        else:
            temp = (zmax_Bs-zmin_Bs)/n_temp
        t[i] = temp*math.pow(disfacBs,(i-degBs)) + zmin_Bs
    for i in range (m+1-degBs,m+1):
        t[i] = zmax_Bs-(zmax_Bs-zmin_Bs)/10000.*(m-i)
    ##### numpy
    # # # # Left bound
    # # # tempArr     = np.arange(degBs)
    # # # t[:degBs]   = zmin_Bs + tempArr*(zmax_Bs-zmin_Bs)/10000.
    # # # # middle
    # # # n_temp      = m+1-degBs-degBs+1
    # # # tempArr     = degBs + np.arange(m+1-2*degBs)
    # # # if disfacBs !=1: temp = (zmax_Bs-zmin_Bs)*(disfacBs-1)/(np.power(disfacBs,n_temp)-1)
    # # # else: temp = (zmax_Bs-zmin_Bs)/n_temp
    # # # t[degBs:m+1-degBs]  = temp*np.power(disfacBs, (tempArr-degBs)) + zmin_Bs
    # # # # Right bound
    # # # tempArr             = m+1-degBs + np.arange(degBs)
    # # # t[m+1-degBs:m+1]    = zmax_Bs-(zmax_Bs-zmin_Bs)/10000.*(m-tempArr) 
    step    = (zmax_Bs-zmin_Bs)/(npts-1)
    obasis  = np.zeros((m, npts))
    nbasis  = np.zeros((m, npts))
    depth   = np.arange(npts) * step + zmin_Bs
    for i in range (m):
        for j in range (npts):
            if (depth[j] >=t[i] and depth[j]<t[i+1]): obasis[i][j] = 1
            else: obasis[i][j] = 0
    for pp in range (1,degBs):
        for i in range (m-pp):
            for j in range (npts):
                temp = (depth[j]-t[i])/(t[i+pp]-t[i])*obasis[i][j]+(t[i+pp+1]-depth[j])/(t[i+pp+1]-t[i+1])*obasis[i+1][j]
                nbasis[i][j] = temp
        for i in range (m-pp):
            for j in range (npts):
                obasis[i][j] = nbasis[i][j]
    nbasis[0][0] = 1
    nbasis[nBs-1][npts-1] = 1
    # # # for i in range (m-pp):
    # # #     ff = open("lf_Bs.%d.dat" % i, "w");
    # # #     for j in range (npts):
    # # #         temp = nbasis[i][j]
    # # #         ff.write("%g %g\n" % ( depth[j],nbasis[i][j]));
    # # #                     # Bs_basis.append(temp);
    # # #     ff.close();            
    return  nbasis
