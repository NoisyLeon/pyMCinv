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
from libc.math cimport sqrt, exp, log, pow, fmax, fmin
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
#    
#    def read_model(self, str infname, float unit=1000., bool isotropic=True, bool tilt=False,
#            Py_ssize_t indz=0, Py_ssize_t indvpv=1, Py_ssize_t indvsv=2, Py_ssize_t indrho=3,
#            Py_ssize_t indvph=4, Py_ssize_t indvsh=5, Py_ssize_t indeta=6, Py_ssize_t inddip=7, 
#            Py_ssize_t indstrike=8, bool reverse=True):
#        """
#        Read model in txt format
#        ===========================================================================================================
#        ::: input parameters :::
#        infname                     - input txt file name
#        unit                        - unit of input, default = 1000., means input has units of km
#        isotropic                   - whether the input is isotrpic or not
#        indz, indvpv, indvsv, indrho- column id(index) for depth, vpv, vsv, rho, vph, vsh, eta
#        indvph, indvsh, indeta
#        reverse                     - revert the arrays or not
#        ===========================================================================================================
#        """
#        cdef np.ndarray inArr
#        cdef 
#        inArr   = np.loadtxt(infname, dtype=np.float32)
#        z       = inArr[:, indz]
#        radius  = (6371.-z)*unit
#        rho     = inArr[:, indrho]*unit
#        vpv     = inArr[:, indvpv]*unit
#        vsv     = inArr[:, indvsv]*unit
#        if isotropic:
#            vph     = inArr[:, indvpv]*unit
#            vsh     = inArr[:, indvsv]*unit
#            eta     = np.ones(vph.size, dtype=np.float32)
#        else:
#            vph     = inArr[:, indvph]*unit
#            vsh     = inArr[:, indvsh]*unit
#            if tilt:
#                dip     = inArr[:, inddip]
#                srike   = inArr[:, indstrike]
#        if reverse:
#            vsv     = vsv[::-1]
#            vsh     = vsh[::-1]
#            vpv     = vpv[::-1]
#            vph     = vph[::-1]
#            eta     = eta[::-1]
#            rho     = rho[::-1]
#            radius  = radius[::-1]
#            if tilt:
#                dip     = dip[::-1]
#                strike  = strike[::-1]
#        ind     = radius > 3700000.
#        vsv     = vsv[ind]
#        vsh     = vsh[ind]
#        vpv     = vpv[ind]
#        vph     = vph[ind]
#        eta     = eta[ind]
#        rho     = rho[ind]
#        radius  = radius[ind]
#        self.get_data_vel(vsv, vsh, vpv, vph, eta, rho, radius)
#        if tilt:
#            self.init_tilt()
#            self.dipArr    = dip
#            self.strikArr  = strike
#        return 
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
    
    
    
        
        
    
    
    
    
    
