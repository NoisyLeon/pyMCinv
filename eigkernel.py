# -*- coding: utf-8 -*-
"""
Module for handling output eigenfunction and sensitivity kernels of surface waves in tilted TI model

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
# Predefine the parameters for the eigkernel object
####################################################
spec_eigkernel = [
        # azimuthal anisotropic terms of the tilted TI model
        ('BcArr',   numba.float32[:]),
        ('BsArr',   numba.float32[:]),
        ('GcArr',   numba.float32[:]),
        ('GsArr',   numba.float32[:]),
        ('HcArr',   numba.float32[:]),
        ('HsArr',   numba.float32[:]),
        ('CcArr',   numba.float32[:]),
        ('CsArr',   numba.float32[:]),
        # PSV eigenfunctions
        ('uz',      numba.float32[:,:]),
        ('tuz',     numba.float32[:,:]),
        ('ur',      numba.float32[:,:]),
        ('tur',     numba.float32[:,:]),
        ('durdz',   numba.float32[:,:]),
        ('duzdz',   numba.float32[:,:]),
        # SH eigenfunctions
        ('ut',      numba.float32[:,:]),
        ('tut',     numba.float32[:,:]),
        ('dutdz',   numba.float32[:,:]),
        # velocity kernels
        ('dcdah',   numba.float32[:,:]),
        ('dcdav',   numba.float32[:,:]),
        ('dcdbh',   numba.float32[:,:]),
        ('dcdbv',   numba.float32[:,:]),
        ('dcdr',    numba.float32[:,:]),
        ('dcdn',    numba.float32[:,:]),
        # Love kernels
        ('dcdA',    numba.float32[:,:]),
        ('dcdC',    numba.float32[:,:]),
        ('dcdF',    numba.float32[:,:]),
        ('dcdL',    numba.float32[:,:]),
        ('dcdN',    numba.float32[:,:]),
        ('dcdrl',   numba.float32[:,:]),
        # number of freq
        ('nfreq',   numba.int32),
        # number of layers
        ('nlay',    numba.int32)
        
        ]

@numba.jitclass(spec_eigkernel)
class eigenkernel(object):
    """
    An object for handling parameter perturbations
    =====================================================================================================================
    ::: parameters :::
    :   eigenfunctions  :

    :   velocity kernels  :
    
    :   Love kernels  :
    
    =====================================================================================================================
    """
    def __init__(self):
        self.freq       = 0
        self.nlay       = -1.
        return
    
    