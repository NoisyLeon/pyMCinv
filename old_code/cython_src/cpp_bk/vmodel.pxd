# -*- coding: utf-8 -*-
"""
Module for handling 1D velocity model objects.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""


from libcpp cimport bool
from libcpp.vector cimport *
cimport modparam

#
#cdef enum int:
#    maxgrid = 1000

# import vector from c++ vector.h
#cdef extern from "<vector>" namespace "std":
#    cdef cppclass vector[T] nogil:
#        cppclass iterator:
#            T operator*() 
#            iterator operator++() 
#            bint operator==(iterator) 
#            bint operator!=(iterator) 
#        vector() 
#        void push_back(T&) 
#        T& operator[](int) 
#        T& at(int) 
#        iterator begin() 
#        iterator end() 
    
cdef class model1d:
    cdef public:
        bool flat, tilt
        int ngrid, nlay
        float rmin, zmax    
        modparam.isomod isomod
    # grid point model
    cdef public:
        float[1024] VsvArr, VpvArr, VshArr, VphArr, etaArr, rhoArr
        # Q factor
        float[1024] qsArr, qpArr
        # radius/depth array
        float[1024] rArr, zArr
        # Love parameters 
        float[1024] AArr, CArr, LArr, FArr, NArr
        # effective Love parameters
        float[1024] AArrE, CArrE, LArrE, FArrE, NArrE
        # 2 theta azimuthal term
        float[1024] BcArr, BsArr, GcArr, GsArr, HcArr, HsArr
        # 4-theta azimuthal terms
        float[1024] CcArr, CsArr
        # Dip/strike angles of anisotropy, see Xie et al.(2015) Fig. 1 for details.
        float[1024] dipArr, strikeArr
        # Voigt matrix for tilted hexagonal symmetric media
        float CijArr[6][6][1024]
        # Voigt matrix for AA(azimuthally independent)/ETI(effective transversely isotropic) part
        float CijAAArr[6][6][1024], CijETIArr[6][6][1024]
    # layerized model
    cdef public:
        float[512] vsv, vpv, vsh, vph, eta, rho
        # Q factor
        float[512] qs, qp
        # radius/depth array
        float[512] r, h
        # Love parameters 
        float[512] A, C, L, F, N
        # effective Love parameters
        float[512] AE, CE, LE, FE, NE
        # 2 theta azimuthal term
        float[512] Bc, Bs, Gc, Gs, Hc, Hs
        # 4-theta azimuthal terms
        float[512] Cc, Cs
        # Dip/strike angles of anisotropy, see Xie et al.(2015) Fig. 1 for details.
        float[512] dip, strike
        # Voigt matrix for tilted hexagonal symmetric media
        float Cij[6][6][512]
        # Voigt matrix for AA(azimuthally independent)/ETI(effective transversely isotropic) part
        float CijAA[6][6][512], CijETI[6][6][512]
    
    cdef void get_model_vel(self, float[:] vsv, float[:] vsh, float[:] vpv, float[:] vph,
              float[:] eta, float[:] rho, float[:] z, float[:] dip, float[:] strike, bool tilt, int N) nogil
    cdef void vel2love(self) nogil
    cdef void love2vel(self) nogil
    cdef bool is_layer_model(self) nogil
    cdef bool grid2layer(self) nogil
    cdef bool is_iso(self) nogil
    cdef void get_iso_vmodel(self) nogil
    
        
    
    
    
    
    
