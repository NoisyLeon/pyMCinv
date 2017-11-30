# -*- coding: utf-8 -*-
"""
Module for handling 1D velocity model objects.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

from __future__ import division

from libc.math cimport sqrt, exp, log, pow, fmax, fmin, fabs, floor
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
cimport cython
cimport modparam

import numpy as np
cimport numpy as np



cdef class model1d:
    """
    An object for handling a 1D Earth model
    =====================================================================================================================
    ::: parameters :::
    VsvArr, VshArr, - Vsv, Vsh, Vpv, Vph velocity (unit - m/s)
    VpvArr, VphArr  
    rhoArr          - density (kg/m^3)
    etaArr          - eta(F/(A-2L)) dimensionless
    AArr, CArr, FArr- Love parameters (unit - Pa)
    LArr, NArr
    rArr            - radius array (unit - m), sorted from the rmin to rmax(6371000. m)
    zArr            - depth array (unit - km), sorted as rArr
    flat            - = 0 spherical Earth, = 1 flat Earth (default)
                        Note: different from CPS
    arrays with *E  - Love parameters for effective VTI tensor
    arrays with *R  - Love parameters and density arrays after Earth flattening transformation for PSV motion
    arrays with *L  - Love parameters and density arrays after Earth flattening transformation for SH motion
    rArrS           - radius array after Earth flattening transformation
    dipArr,strikeArr- dip/strike angles, used for tilted hexagonal symmetric media
    CijArr          - elastic tensor given rotational angles(dip, strike) (unit - Pa)
    CijAA           - azimuthally anisotropic elastic tensor (unit - Pa)
    =====================================================================================================================
    """
    def __init__(self):
        self.flat   = 0
        self.tilt   = 0
        self.isomod = modparam.isomod()
        self.nlay   = 0
        self.ngrid  = 0
#        self.ttimod = modparam.ttimod()
        return
    
    def read_model(self, str infname, float unit=1., int isotropic=1, int tilt=0,
            int indz=0, int indvpv=1, int indvsv=2, int indrho=3, int indvph=4, int indvsh=5, 
            int indeta=6, int inddip=7, int indstrike=8):
        """
        Read model in txt format
        ===========================================================================================================
        ::: input parameters :::
        infname                     - input txt file name
        unit                        - unit of input, default = 1., means input has units of km
        isotropic                   - whether the input is isotrpic or not
        indz, indvpv, indvsv, indrho- column id(index) for depth, vpv, vsv, rho, vph, vsh, eta
        indvph, indvsh, indeta
        reverse                     - revert the arrays or not
        ===========================================================================================================
        """
        cdef np.ndarray inArr
        cdef float[:] z, vpv, vsv, rho, vsh, vph, eta, dip, strike
        cdef int N
                  
        inArr   = np.loadtxt(infname, dtype=np.float32)
        z       = inArr[:, indz]
        rho     = inArr[:, indrho]*unit
        vpv     = inArr[:, indvpv]*unit
        vsv     = inArr[:, indvsv]*unit
        N      = inArr.shape[0]
        if isotropic:
            vph     = inArr[:, indvpv]*unit
            vsh     = inArr[:, indvsv]*unit
            eta     = np.ones(N, dtype=np.float32)
        else:
            vph     = inArr[:, indvph]*unit
            vsh     = inArr[:, indvsh]*unit
        if tilt and isotropic:
            dip     = inArr[:, inddip]
            strike  = inArr[:, indstrike]
        else:
            dip     = np.ones(N, dtype=np.float32)
            strike  = np.ones(N, dtype=np.float32)
        self.get_model_vel(vsv=vsv, vsh=vsh, vpv=vpv, vph=vph,\
                      eta=eta, rho=rho, z=z, dip=dip, strike=strike, tilt=tilt, N=N)
        return
    
    def write_model(self, str outfname, int isotropic=1):
        """
        Write model in txt format
        ===========================================================================================================
        ::: input parameters :::
        outfname                    - output txt file name
        unit                        - unit of output, default = 1., means output has units of km
        isotropic                   - whether the input is isotrpic or not
        ===========================================================================================================
        """
        cdef np.ndarray outArr, z, vsv, vsh, vpv, vph, eta, rho, dip, strike
        cdef int N
        cdef str header
        z       = np.array(self.zArr, dtype=np.float32)
        vsv     = np.array(self.VsvArr, dtype=np.float32)
        vsh     = np.array(self.VshArr, dtype=np.float32)
        vpv     = np.array(self.VpvArr, dtype=np.float32)
        vph     = np.array(self.VphArr, dtype=np.float32)
        eta     = np.array(self.etaArr, dtype=np.float32)
        rho     = np.array(self.rhoArr, dtype=np.float32)
        dip     = np.array(self.dipArr, dtype=np.float32)
        strike  = np.array(self.strikeArr, dtype=np.float32)
        
        outArr  = np.append(z[:self.ngrid], vsv[:self.ngrid])
        if not isotropic:
            outArr  = np.append(outArr, vsh[:self.ngrid])
        outArr  = np.append(outArr, vpv[:self.ngrid])
        if not isotropic:
            outArr  = np.append(outArr, vph[:self.ngrid])
            outArr  = np.append(outArr, eta[:self.ngrid])
            if self.tilt:
                outArr  = np.append(outArr, dip[:self.ngrid])
                outArr  = np.append(outArr, strike[:self.ngrid])
        outArr  = np.append(outArr, rho[:self.ngrid])
        if isotropic:
            N       = 4
            header  = 'depth vs vp rho'
        else:
            if self.tilt:
                N       = 9
                header  = 'depth vsv vsh vpv vph eta dip strike rho'
            else:
                N       = 7
                header  = 'depth vsv vsh vpv vph eta rho'
        outArr  = outArr.reshape((N, self.ngrid))
        outArr  = outArr.T
        np.savetxt(outfname, outArr, fmt='%g', header=header)
        return 

    @cython.boundscheck(False)
    cdef void get_model_vel(self, float[:] vsv, float[:] vsh, float[:] vpv, float[:] vph,
                      float[:] eta, float[:] rho, float[:] z, float[:] dip, float[:] strike, int tilt, int N) nogil:
        """
        Get model data given velocity/density/depth arrays
        """
        cdef Py_ssize_t i
        if N > 1024:
            printf('Number of grid points %d is larger than 1024!', N)
        for i in range(N):
            self.zArr[i]        = z[i]
            self.VsvArr[i]      = vsv[i]
            self.VshArr[i]      = vsh[i]
            self.VpvArr[i]      = vpv[i]
            self.VphArr[i]      = vph[i]
            self.etaArr[i]      = eta[i]
            self.rhoArr[i]      = rho[i]
            if tilt==1:
                self.dipArr[i]      = dip[i]
                self.strikeArr[i]   = strike[i]
        self.ngrid = N
        self.vel2love()
        return
    
    @cython.boundscheck(False)
    cdef void vel2love(self) nogil:
        """
        velocity parameters to Love parameters
        """
        cdef Py_ssize_t i
        for i in range(self.ngrid):
            self.AArr[i]= self.rhoArr[i] * (self.VphArr[i])**2
            self.CArr[i]= self.rhoArr[i] * (self.VpvArr[i])**2
            self.LArr[i]= self.rhoArr[i] * (self.VsvArr[i])**2
            self.FArr[i]= self.etaArr[i] * (self.AArr[i] - 2.* self.LArr[i])
            self.NArr[i]= self.rhoArr[i] * (self.VshArr[i])**2
        for i in range(self.nlay):
            self.A[i]   = self.rho[i] * (self.vph[i])**2
            self.C[i]   = self.rho[i] * (self.vpv[i])**2
            self.L[i]   = self.rho[i] * (self.vsv[i])**2
            self.F[i]   = self.eta[i] * (self.A[i] - 2.* self.L[i])
            self.N[i]   = self.rho[i] * (self.vsh[i])**2
        return
        
    @cython.boundscheck(False)
    cdef void love2vel(self) nogil:
        """
        Love parameters to velocity parameters
        """
        cdef Py_ssize_t i
        for i in range(self.ngrid):
            self.VphArr[i]  = sqrt(self.AArr[i]/self.rhoArr[i])
            self.VpvArr[i]  = sqrt(self.CArr[i]/self.rhoArr[i])
            self.VshArr[i]  = sqrt(self.NArr[i]/self.rhoArr[i])
            self.VsvArr[i]  = sqrt(self.LArr[i]/self.rhoArr[i])
            self.etaArr[i]  = self.FArr[i]/(self.AArr[i] - 2.* self.LArr[i])
        for i in range(self.nlay):
            self.vph[i]     = sqrt(self.A[i]/self.rho[i])
            self.vpv[i]     = sqrt(self.C[i]/self.rho[i])
            self.vsh[i]     = sqrt(self.N[i]/self.rho[i])
            self.vsv[i]     = sqrt(self.L[i]/self.rho[i])
            self.eta[i]     = self.F[i]/(self.A[i] - 2.* self.L[i])
        return
    
    @cython.boundscheck(False)
    cdef int grid2layer(self) nogil:
        """
        Convert grid point model to layerized model
        """
        cdef Py_ssize_t i, j
        if not self.is_layer_model():
            return 0
        self.nlay = int(self.ngrid/2)
        j   = 0
        for i in range(self.ngrid):
            if i == 0:
                self.vsv[j]     = self.VsvArr[i]
                self.vsh[j]     = self.VshArr[i]
                self.vpv[j]     = self.VpvArr[i]
                self.vph[j]     = self.VphArr[i]
                self.eta[j]     = self.etaArr[i]
                self.rho[j]     = self.rhoArr[i]
                self.dip[j]     = self.dipArr[i]
                self.strike[j]  = self.strikeArr[i]
                self.qs[j]      = self.qsArr[i]
                self.qp[j]      = self.qpArr[i]
                self.h[j]       = self.zArr[i+1]
                j += 1
                continue
            if i % 2 != 0: 
                continue
            self.vsv[j]     = self.VsvArr[i]
            self.vsh[j]     = self.VshArr[i]
            self.vpv[j]     = self.VpvArr[i]
            self.vph[j]     = self.VphArr[i]
            self.eta[j]     = self.etaArr[i]
            self.rho[j]     = self.rhoArr[i]
            self.dip[j]     = self.dipArr[i]
            self.strike[j]  = self.strikeArr[i]
            self.qs[j]      = self.qsArr[i]
            self.qp[j]      = self.qpArr[i]
            self.h[j]       = self.zArr[i+1] - self.zArr[i]
            j += 1
        return 1
    
    def grid2layer_interface(self):
        return self.grid2layer()

    @cython.boundscheck(False)
    cdef int is_iso(self) nogil:
        """Check if the model is isotropic at each point.
        """
        cdef float tol = 1e-5
        cdef Py_ssize_t i
        for i in range(self.ngrid):
            if fabs(self.AArr[i] - self.CArr[i])> tol or fabs(self.LArr[i] - self.NArr[i])> tol\
                   or fabs(self.FArr[i] - (self.AArr[i]- 2.*self.LArr[i]) )> tol:
                return 0
        return 1
    
    @cython.boundscheck(False)
    cdef void get_iso_vmodel(self) nogil:
        """
        get the isotropic model from isomod
        """
        cdef float[512] vs, vp, rho, qs, qp, hArr, z
        cdef int N
        cdef Py_ssize_t i, j, k
        cdef float depth = 0.
        
        N = self.isomod.get_vmodel(vs, vp, rho, qs, qp, hArr)
        # store layerized model
        for i in range(N):
            depth       = depth + hArr[i]
            z[i]        = depth
            self.vsv[i] = vs[i] 
            self.vsh[i] = vs[i]
            self.vpv[i] = vp[i] 
            self.vph[i] = vp[i] 
            self.eta[i] = 1. 
            self.rho[i] = rho[i] 
            self.h[i]   = hArr[i] 
            self.qs[i]  = qs[i]
            self.qp[i]  = qp[i]
        self.nlay   = N
        # store grid point model
        for i in range(2*N):
            if i == 0:
                self.VsvArr[i]  = vs[i]
                self.VshArr[i]  = vs[i]
                self.VpvArr[i]  = vp[i]
                self.VphArr[i]  = vp[i]
                self.qsArr[i]   = qs[i]
                self.qpArr[i]   = qp[i]
                self.rhoArr[i]  = rho[i]
                self.etaArr[i]  = 1.
                continue
            j   = int(i/2)
            k   = i%2
            self.zArr[i]    = z[j+k-1]
            self.VsvArr[i]  = vs[j]
            self.VshArr[i]  = vs[j]
            self.VpvArr[i]  = vp[j]
            self.VphArr[i]  = vp[j]
            self.qsArr[i]   = qs[j]
            self.qpArr[i]   = qp[j]
            self.rhoArr[i]  = rho[j]
            self.etaArr[i]  = 1.
        self.ngrid  = 2*N
        self.vel2love()
        return 
    
    def get_iso_vmodel_interface(self):
        self.get_iso_vmodel()
        return
    
    @cython.boundscheck(False)
    cdef int is_layer_model(self) nogil:
        """
        Check if the grid point model is a layerized one or not
        """
        cdef Py_ssize_t i
        cdef float z0, z1, A0, A1, C0, C1, F0, F1, L0, L1, N0, N1, 
        cdef float d0, d1, s0, s1
        if self.ngrid %2 !=0:
            return 0
        self.vel2love()
        
        for i in range(self.ngrid):
            if i == 0: 
                continue
            if i % 2 != 0: 
                continue
        
            z0 = self.zArr[i-1];  z1 = self.zArr[i]
            if z0 != z1:
                return 0
            A0  = self.AArr[i-2]; A1 = self.AArr[i-1]
            if A0 != A1:
                return 0
            C0  = self.CArr[i-2]; C1 = self.CArr[i-1]
            if C0 != C1:
                return 0
            F0  = self.FArr[i-2]; F1 = self.FArr[i-1]
            if F0 != F1:
                return 0
            L0  = self.LArr[i-2]; L1 = self.LArr[i-1]
            if L0 != L1:
                return 0
            N0  = self.NArr[i-2]; N1 = self.NArr[i-1]
            if N0 != N1:
                return 0
            # check tilted angles of anisotropic axis
            if self.tilt: 
                d0  = self.dipArr[i-2]; d1 = self.dipArr[i-1]
                if d0 != d1:
                    return 0
                s0  = self.strikeArr[i-2]; s1 = self.strikeArr[i-1]
                if s0 != s1:
                    return 0
        return 1
    
    
    
    
    

        
    
    
    
        
        
    
    
    
    
    
