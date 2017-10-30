# -*- coding: utf-8 -*-
"""
Module for 1D profile inversion

Numba is used for speeding up of the code.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

import numpy as np
import vmodel, data, modparam
import fast_surf, theo



class vprofile1d(object):
    def __init__(self):
        self.model  = vmodel.model1d()
        self.indata = data.data1d()
        return
    
    def readdisp(self, infname, dtype='ph', wtype='ray'):
        dtype=dtype.lower()
        wtype=wtype.lower()
        if wtype=='ray' or wtype=='rayleigh' or wtype=='r':
            data.readdisptxt(infname=infname, indisp=self.indata.dispR, dtype=dtype)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            data.readdisptxt(infname=infname, indisp=self.indata.dispL, dtype=dtype)
        else:
            raise ValueError('Unexpected wave type: '+wtype)
        return
    
    def readrf(self, infname, dtype='R'):
        dtype=dtype.lower()
        if dtype=='r' or dtype == 'radial':
            data.readrftxt(infname=infname, inrf=self.indata.rfr)
        elif dtype=='t' or dtype == 'transverse':
            data.readrftxt(infname=infname, inrf=self.indata.rft)
        else:
            raise ValueError('Unexpected wave type: '+dtype)
        return
    
    def readmod(self, infname, dtype='iso'):
        dtype=dtype.lower()
        if dtype=='iso' or dtype == 'isotropic':
            modparam.readmodtxt(infname=infname, inmod=self.model.isomod)
        else:
            raise ValueError('Unexpected wave type: '+dtype)
        return
            
    
    