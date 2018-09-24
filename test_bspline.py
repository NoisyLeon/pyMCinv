import scipy.interpolate
import numpy as np
import math

def bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts):
    """
    function that generate B spline basis
    """
    #-------------------------------- 
    # defining the knot vector
    #--------------------------------
    m           = nBs-1+degBs
    t           = np.zeros(m+1, dtype=np.float64)
    for i in range(degBs):
        t[i]    = zmin_Bs + i*(zmax_Bs-zmin_Bs)/10000.
    for i in range(degBs,m+1-degBs):
        n_temp  = m+1-degBs-degBs+1
        if (disfacBs !=1):
            temp= (zmax_Bs-zmin_Bs)*(disfacBs-1)/(math.pow(disfacBs,n_temp)-1)
        else:
            temp= (zmax_Bs-zmin_Bs)/n_temp
        t[i]    = temp*math.pow(disfacBs,(i-degBs)) + zmin_Bs
    for i in range(m+1-degBs,m+1):
        t[i]    = zmax_Bs-(zmax_Bs-zmin_Bs)/10000.*(m-i)
    # depth array
    step        = (zmax_Bs-zmin_Bs)/(npts-1)
    depth       = np.zeros(npts, dtype=np.float64)
    for i in range(npts):
        depth[i]= np.float64(i) * np.float64(step) + np.float64(zmin_Bs)
    # arrays for storing B spline basis
    obasis      = np.zeros((np.int64(m), np.int64(npts)), dtype = np.float64)
    nbasis      = np.zeros((np.int64(m), np.int64(npts)), dtype = np.float64)
    #-------------------------------- 
    # computing B spline basis functions
    #--------------------------------
    for i in range (m):
        for j in range (npts):
            if (depth[j] >=t[i] and depth[j]<t[i+1]):
                obasis[i][j]= 1
            else:
                obasis[i][j]= 0
    for pp in range (1,degBs):
        for i in range (m-pp):
            for j in range (npts):
                nbasis[i][j]= (depth[j]-t[i])/(t[i+pp]-t[i])*obasis[i][j] + \
                        (t[i+pp+1]-depth[j])/(t[i+pp+1]-t[i+1])*obasis[i+1][j]
        for i in range (m-pp):
            for j in range (npts):
                obasis[i][j]= nbasis[i][j]
    nbasis[0][0]            = 1
    nbasis[nBs-1][npts-1]   = 1
    return nbasis, t, depth

out1 = bspl_basis(5, 4, 0, 100, 2., 60)

spl0 = scipy.interpolate.BSpline(out1[1], [1, 0, 0, 0, 0], k=3)
spl1 = scipy.interpolate.BSpline(out1[1], [0, 1, 0, 0, 0], k=3)
spl2 = scipy.interpolate.BSpline(out1[1], [0, 0, 1, 0, 0], k=3)
spl3 = scipy.interpolate.BSpline(out1[1], [0, 0, 0, 1, 0], k=3)
spl4 = scipy.interpolate.BSpline(out1[1], [0, 0, 0, 0, 1], k=3)

