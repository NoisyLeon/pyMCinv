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
        int nbgrid
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
        # 4th order elastic tensor/Voigt matrix for tilted hexagonal symmetric media
        float[1024] CijklArr, CijArr
        # Voigt matrix for AA(azimuthally independent)/ETI(effective transversely isotropic) part
        float[1024] CijAA, CijETI
        float rmin    
        modparam.isomod isomod
    
        
        
    
    
    
    
    
