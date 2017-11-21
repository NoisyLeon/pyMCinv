# -*- coding: utf-8 -*-
"""
Module for handling parameterization of the model

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

from libcpp cimport bool

# import random from c++ random.h
cdef extern from "<random>" namespace "std":
    cdef cppclass mt19937 nogil:
        mt19937()    # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed)  # not worrying about matching the exact int type for seed

    cdef cppclass uniform_real_distribution[T] nogil:
        uniform_real_distribution()  
        uniform_real_distribution(T a, T b)  
        T operator()(mt19937 gen)  # ignore the possibility of using other classes for "gen"
        
    cdef cppclass normal_distribution[T] nogil:
        normal_distribution() 
        normal_distribution(T mu, T sigma) 
        T operator()(mt19937 gen)  # ignore the possibility of using other classes for "gen"

# import vector from c++ vector.h
cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T] nogil:
        cppclass iterator:
            T operator*() 
            iterator operator++() 
            bint operator==(iterator) 
            bint operator!=(iterator) 
        vector() 
        void push_back(T&) 
        T& operator[](int) 
        T& at(int) 
        iterator begin() 
        iterator end() 

ctypedef vector[vector[float]] FloatMatrix

cdef float random_gauss(float mu, float sigma) nogil

cdef float random_uniform(float a, float b) nogil

cdef class para1d:
    cdef public int npara, maxind
    cdef public bool isspace
    cdef public float[:, :] paraindex, space
    cdef public float[:] paraval
    
    cpdef init_arr(self, int npara)
    cdef bool new_paraval(self, int ptype) nogil
    cpdef copy(self)
    
cdef FloatMatrix bspl_basis(int nBs, int degBs, float zmin_Bs, float zmax_Bs, float disfacBs, int npts) nogil

cdef class isomod:
    cdef public int nmod
    cdef int maxlay, maxspl
    cdef public para1d para
    cdef public int[:] numbp, mtype, nlay, isspl
    cdef public float[:] thickness, vpvs
    cdef public float[:, :] cvel, ratio, vs, hArr
    cdef public float[:, :, :] spl
    
    cpdef init_arr(self, nmod)
    cdef bool bspline(self, Py_ssize_t i) nogil
    cdef bool update(self) nogil
    cdef void get_paraind(self) nogil
    cdef void mod2para(self) nogil
    cdef void para2mod(self) nogil
    cdef bool isgood(self, int m0, int m1, int g0, int g1) nogil
    cdef void get_vmodel(self, vector[float] &vs, vector[float] &vp, vector[float] &rho,\
                vector[float] &qs, vector[float] &qp, vector[float] &hArr) nogil
    cpdef copy(self)
#    
    
    
    
    
    
    
    
    
    
    


    