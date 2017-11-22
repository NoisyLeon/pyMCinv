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

from libcpp cimport bool
from libc.math cimport sqrt, exp, log, pow, fmax, fmin, fabs
from libc.stdlib cimport malloc, free
from libc.stdio cimport printf
cimport cython
cimport modparam
#from cython.view cimport array as cvarray

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
        self.flat   = False
        self.isomod = modparam.isomod()
#        self.ttimod = modparam.ttimod()
        return
    
    def read_model(self, str infname, float unit=1., bool isotropic=True, bool tilt=False,
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
    
    
    def write_model(self, str outfname, bool isotropic=True):
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
                      float[:] eta, float[:] rho, float[:] z, float[:] dip, float[:] strike, bool tilt, int N) nogil:
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
            if tilt:
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
        return
    
    @cython.boundscheck(False)
    cdef bool is_iso(self) nogil:
        """Check if the model is isotropic at each point.
        """
        cdef float tol = 1e-5
        cdef Py_ssize_t i
        for i in range(self.ngrid):
            if fabs(self.AArr[i] - self.CArr[i])> tol or fabs(self.LArr[i] - self.NArr[i])> tol\
                   or fabs(self.FArr[i] - (self.AArr[i]- 2.*self.LArr[i]) )> tol:
                return False
        return True
    
    def get_iso_vmodel(self):
        """
        get the isotropic model from isomod
        """
        cdef vector[float] vs, vp, rho, qs, qp, hArr, z
        cdef int N
        cdef Py_ssize_t i
        self.get_vmodel(vs, vp, rho, qs, qp, hArr)
        N               = hArr.size()
        for i in range(N):
            z.push_back(hArr[i])
#        
#        hArr, vs, vp, rho, qs, qp = self.isomod.get_vmodel()
#        zArr            = hArr.cumsum()
#        N               = zArr.size
#        self.zArr       = np.zeros(2*N, dtype=np.float32)
#        self.VsvArr     = np.zeros(2*N, dtype=np.float32)
#        self.VshArr     = np.zeros(2*N, dtype=np.float32)
#        self.VpvArr     = np.zeros(2*N, dtype=np.float32)
#        self.VphArr     = np.zeros(2*N, dtype=np.float32)
#        self.qsArr      = np.zeros(2*N, dtype=np.float32)
#        self.qpArr      = np.zeros(2*N, dtype=np.float32)
#        self.rhoArr     = np.zeros(2*N, dtype=np.float32)
#        self.rArr       = np.zeros(2*N, dtype=np.float32)
#        for i in xrange(2*N):
#            if i == 0:
#                self.VsvArr[i]  = vs[i]*1000.
#                self.VshArr[i]  = vs[i]*1000.
#                self.VpvArr[i]  = vp[i]*1000.
#                self.VphArr[i]  = vp[i]*1000.
#                self.qsArr[i]   = qs[i]
#                self.qpArr[i]   = qp[i]
#                self.rhoArr[i]  = rho[i]*1000.
#                continue
#            j   = int(i/2)
#            k   = i%2
#            self.zArr[i]    = zArr[j+k-1]
#            self.VsvArr[i]  = vs[j]*1000.
#            self.VshArr[i]  = vs[j]*1000.
#            self.VpvArr[i]  = vp[j]*1000.
#            self.VphArr[i]  = vp[j]*1000.
#            self.qsArr[i]   = qs[j]
#            self.qpArr[i]   = qp[j]
#            self.rhoArr[i]  = rho[j]*1000.
#        self.zArr       = self.zArr[::-1]
#        self.VsvArr     = self.VsvArr[::-1]
#        self.VshArr     = self.VshArr[::-1]
#        self.VpvArr     = self.VpvArr[::-1]
#        self.VphArr     = self.VphArr[::-1]
#        self.etaArr     = np.ones(2*N, dtype=np.float32)
#        self.qsArr      = self.qsArr[::-1]
#        self.qpArr      = self.qpArr[::-1]
#        self.rhoArr     = self.rhoArr[::-1]
#        self.rArr       = (np.float32(6371000.) - self.zArr*np.float32(1000.))
#        self.vel2love()
#        return hArr, vs, vp, rho, qs, qp
    
    
    

        
        
        
        
#    
#    cdef get_data_vel(self, vsv, vsh, vpv, vph, eta, rho, radius):
#        """
#        Get model data given velocity/density/radius arrays
#        """
#        self.rArr   = radius
#        self.rhoArr = rho
#        if radius[-1] != 6371000.:
#            raise ValueError('Last element of radius array should be 6371000. meter !')
#        if np.any(vsv<500.) or np.any(vsh<500.) or np.any(vpv<500.) or np.any(vph<500.) or np.any(rho<500.):
#            raise ValueError('Wrong unit for model parameters!')
#        if np.any(radius< 10000.):
#            raise ValueError('Wrong unit for radius!')
#        ###
#        # assign velocities
#        ###
#        self.VsvArr = vsv
#        self.VshArr = vsh
#        self.VpvArr = vpv
#        self.VphArr = vph
#        self.etaArr = eta
#        ###
#        # compute Love parameters
#        ###
#        self.AArr   = rho * vph**2
#        self.CArr   = rho * vpv**2
#        self.LArr   = rho * vsv**2
#        self.FArr   = eta * (self.AArr - np.float32(2.)* self.LArr)
#        self.NArr   = rho * vsh**2
#        return
    
    
    
        
        
    
    
    
    
    
