# -*- coding: utf-8 -*-
"""
Module for handling splines

Numba is used for speeding up of the code.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import numpy as np
import math
import numba


####################################################
# Predefine the parameters for the disp object
####################################################
spec_isospl = [
        # number of control points
        ('numbp',       numba.int32),
        ('isspl',       numba.boolean),
        ('stype',       numba.int32),
        ('thickness',   numba.float32),
        ('nlay',        numba.int32),
        ('vpvs',        numba.float32),      
        # arrays
        ('spl',         numba.float32[:, :]),
        ('ratio',       numba.float32[:]),
        ('vel',         numba.float32[:]),
        ('cvel',        numba.float32[:]),
        ('hArr',        numba.float32[:]),
        ('t',           numba.float32[:])
        ]

@numba.jit(numba.float32(numba.float32, numba.int32, numba.int32, numba.float32[:]))
def B(x, k, i, t):
    if k == 0:
       return 1.0 if t[i] <= x < t[i+1] else 0.0
    if t[i+k] == t[i]:
       c1 = 0.0
    else:
       c1 = (x - t[i])/(t[i+k] - t[i]) * B(x, k-1, i, t)
    if t[i+k+1] == t[i+1]:
       c2 = 0.0
    else:
       c2 = (t[i+k+1] - x)/(t[i+k+1] - t[i+1]) * B(x, k-1, i+1, t)
    return c1 + c2


@numba.jitclass(spec_isospl)
class isospl(object):
    
    def __init__(self):
        self.isspl      = False
        self.numbp      = 0
        self.nlay       = 20
        self.vpvs       = 1.75
        self.thickness  = 0.
        self.stype      = -1
        return
    
    def bspline(self):
        """
        Compute B-spline basis
        The actual degree is k = degBs - 1
        e.g. nBs = 5, degBs = 4, k = 3, cubic B spline
        ::: output :::
        self.spl    - (nBs+k, npts)
                        [:nBs, :] B spline basis for nBs control points
                        [nBs:,:] can be ignored
        """
        # initialize
        nBs         = self.numbp
        if nBs < 4:
            degBs   = 3
        else:
            degBs   = 4
        zmin_Bs     = 0.
        zmax_Bs     = self.thickness
        disfacBs    = 2.
        npts        = self.nlay
        #-------------------------------- 
        # defining the knot vector
        #--------------------------------
        m           = nBs-1+degBs
        t           = np.zeros(m+1, dtype=np.float32)
        for i in xrange(degBs):
            t[i] = zmin_Bs + i*(zmax_Bs-zmin_Bs)/10000.
        for i in range(degBs,m+1-degBs):
            n_temp  = m+1-degBs-degBs+1
            if (disfacBs !=1):
                temp= (zmax_Bs-zmin_Bs)*(disfacBs-1)/(math.pow(disfacBs,n_temp)-1)
            else:
                temp= (zmax_Bs-zmin_Bs)/n_temp
            t[i] = temp*math.pow(disfacBs,(i-degBs)) + zmin_Bs
        for i in range(m+1-degBs,m+1):
            t[i] = zmax_Bs-(zmax_Bs-zmin_Bs)/10000.*(m-i)
        # depth array
        step    = (zmax_Bs-zmin_Bs)/(npts-1)
        depth   = np.zeros(npts, dtype=np.float32)
        for i in xrange(npts):
            depth[i]    = np.float32(i) * np.float32(step) + np.float32(zmin_Bs)
        # arrays for storing B spline basis
        obasis  = np.zeros((np.int64(m), np.int64(npts)), dtype = np.float32)
        nbasis  = np.zeros((np.int64(m), np.int64(npts)), dtype = np.float32)
        #-------------------------------- 
        # computing B spline basis
        #--------------------------------
        for i in xrange (m):
            for j in xrange (npts):
                if (depth[j] >=t[i] and depth[j]<t[i+1]):
                    obasis[i][j] = 1
                else:
                    obasis[i][j] = 0
        for pp in range (1,degBs):
            for i in xrange (m-pp):
                for j in xrange (npts):
                    nbasis[i][j] = (depth[j]-t[i])/(t[i+pp]-t[i])*obasis[i][j] + \
                            (t[i+pp+1]-depth[j])/(t[i+pp+1]-t[i+1])*obasis[i+1][j]
            for i in xrange (m-pp):
                for j in xrange (npts):
                    obasis[i][j] = nbasis[i][j]
        nbasis[0][0]            = 1
        nbasis[nBs-1][npts-1]   = 1
        self.spl                = nbasis
        self.isspl              = True
        self.t                  = t
        return
        
    
    def bspline2(self):
        """
        For benchmark of bspline
        """
        # initialize
        nBs         = self.numbp
        if nBs < 4:
            degBs   = 3
        else:
            degBs   = 4
        zmin_Bs     = 0.
        zmax_Bs     = self.thickness
        disfacBs    = 2.
        npts        = self.nlay
        # 
        m           = nBs-1+degBs
        t           = np.zeros(m+1, dtype=np.float32)
        
        for i in xrange(degBs):
            t[i] = zmin_Bs + i*(zmax_Bs-zmin_Bs)/10000.
 
        for i in range(degBs,m+1-degBs):
            n_temp  = m+1-degBs-degBs+1
            if (disfacBs !=1):
                temp= (zmax_Bs-zmin_Bs)*(disfacBs-1)/(math.pow(disfacBs,n_temp)-1)
            else:
                temp= (zmax_Bs-zmin_Bs)/n_temp
            t[i] = temp*math.pow(disfacBs,(i-degBs)) + zmin_Bs
        
        for i in range(m+1-degBs,m+1):
            t[i] = zmax_Bs-(zmax_Bs-zmin_Bs)/10000.*(m-i)
        
        step    = (zmax_Bs-zmin_Bs)/(npts-1)
        nbasis  = np.zeros((np.int64(m), np.int64(npts)), dtype = np.float32)
        depth   = np.zeros(npts, dtype=np.float32)
        for i in xrange(npts):
            depth[i]    = np.float32(i) * np.float32(step) + np.float32(zmin_Bs)
        for pp in range (1,degBs):
            for i in xrange (m-pp):
                for j in xrange (npts):
                    nbasis[i][j] = B(x=depth[j], k=pp, i=i, t=t)
        nbasis[0][0]            = 1
        nbasis[nBs-1][npts-1]   = 1
        self.spl    = nbasis
        return
    
    def updateBspl(self):
        if self.thickness >= 150:
            self.nlay = 60
        elif self.thickness < 10:
            self.nlay = 5
        elif self.thickness < 20:
            self.nlay = 10
        self.bspline()
        return
    
isolst_type=numba.typeof([isospl()])

spec_isosplLst = [
        # number of control points
        ('spls',        isolst_type)
        ]

@numba.jitclass(spec_isosplLst)
class isosplLst(object):
    def __init__(self):
        return
        # self.spls=[]
    #     if isinstance(spls, isospl):
    #         spls = [spls]
    #     if spls:
    #         self.spls.extend(spls)
    #     self.nisospl = 0
    # 
    # def __add__(self, other):
    #     """
    #     Add two isosplLst with self += other.
    #     """
    #     if isinstance(other, isospl):
    #         other = isosplLst([other])
    #     if not isinstance(other, isosplLst):
    #         raise TypeError
    #     spls = self.spls + other.spls
    #     return self.__class__(spls=spls)
    # 
    # def __len__(self):
    #     """
    #     Return the number of spls in the isosplLst object.
    #     """
    #     return len(self.spls)
    # 
    # def __getitem__(self, index):
    #     """
    #     __getitem__ method of isosplLst objects.
    #     :return: isospl objects
    #     """
    #     if isinstance(index, slice):
    #         return self.__class__(spls=self.spls.__getitem__(index))
    #     else:
    #         return self.spls.__getitem__(index)
    # 
    # def append(self, inspl):
    #     """
    #     Append a single isospl object to the current isosplLst object.
    #     """
    #     if isinstance(inspl, isospl):
    #         self.spls.append(inspl)
    #     else:
    #         msg = 'Append only supports a single isospl object as an argument.'
    #         raise TypeError(msg)
    #     return self
    