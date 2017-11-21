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
cimport modparam




cdef class model1d:
    cdef bool flat
    cdef modparam.isomod isomod
    
        
        
    
    
    
    
    
