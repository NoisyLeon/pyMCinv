# -*- coding: utf-8 -*-
"""
Module for handling input data for joint inversion.

Numba is used for speeding up of the code.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import numpy as np
import numba



####################################################
# I/O functions
####################################################

def readdisptxt(infname, indisp, dtype='ph'):
    if dtype == 'ph' or dtype == 'phase':
        if indisp.isphase:
            print 'phase velocity data is already stored!'
            return
        inArr 		= np.loadtxt(infname, dtype=np.float32)
        indisp.pper = inArr[:,0]
        indisp.pvelo= inArr[:,1]
        indisp.npper= indisp.pper.size
        try:
            indisp.stdpvelo= inArr[:,2]
        except IndexError:
            pass
        indisp.isphase = True
    elif dtype == 'gr' or dtype == 'group':
        if indisp.isgroup:
            print 'group velocity data is already stored!'
            return
        inArr 		= np.loadtxt(infname, dtype=np.float32)
        indisp.gper = inArr[:,0]
        indisp.gvelo= inArr[:,1]
        indisp.ngper= indisp.gper.size
        try:
            indisp.stdgvelo= inArr[:,2]
        except IndexError:
            pass
        indisp.isgroup  = True
    else:
        raise ValueError('Unexpected dtype: '+dtype)
    return True

def readrftxt(infname, inrf):
    if inrf.npts > 0:
        print 'receiver function data is already stored!'
        return False
    inArr 		= np.loadtxt(infname, dtype=np.float32)
    inrf.to     = inArr[:,0]
    inrf.rfo    = inArr[:,1]
    inrf.npts   = inrf.rfo.size
    try:
        inrf.stdrfo = inArr[:,2]
    except IndexError:
        inrf.stdrfo = np.ones(self.npts, dtype=np.float32)*np.float32(0.1)
    inrf.fs     = 1./(inrf.to[1] - inrf.to[0])
    return True
    


####################################################
# Predefine the parameters for the disp object
####################################################
spec_disp = [
        # phase velocities
        ('npper', numba.int32),
        ('pper', numba.float32[:]),
        # observed 
        ('pvelo', numba.float32[:]),
        ('stdpvelo', numba.float32[:]),
        ('pphio', numba.float32[:]),
        ('pampo', numba.float32[:]),
        # predicted
        ('pvelp', numba.float32[:]),
        ('stdpvelp', numba.float32[:]),
        ('pphip', numba.float32[:]),
        ('pampp', numba.float32[:]),
        # 
        ('isphase', numba.boolean),
        ('pmisfit', numba.float32),
        ('pL', numba.float32),
        # group velocities
        ('ngper', numba.int32),
        ('gper', numba.float32[:]),
        # observed
        ('gvelo', numba.float32[:]),
        ('stdgvelo', numba.float32[:]),
        ('gphio', numba.float32[:]),
        ('gampo', numba.float32[:]),
        # predicted
        ('gvelp', numba.float32[:]),
        ('stdgvelp', numba.float32[:]),
        ('gphip', numba.float32[:]),
        ('gampp', numba.float32[:]),
        #
        ('isgroup', numba.boolean),
        ('gmisfit', numba.float32),
        ('gL', numba.float32),
        # total misfit/likelihood
        ('misfit', numba.float32),
        ('L', numba.float32),
        # common period for phase/group
        ('period', numba.float32[:]),
        ('nper', numba.int32)
        ]

@numba.jitclass(spec_disp)
class disp(object):
    def __init__(self):
        self.npper  = 0
        self.ngper  = 0
        self.nper   = 0
        self.isphase= False
        self.isgroup= False
        return
    
####################################################
# Predefine the parameters for the rf object
####################################################
spec_rf = [
        # sampling frequency/npts
        ('fs', numba.float32),
        ('npts', numba.int32),
        # observed receiver function
        ('rfo', numba.float32[:]),
        ('to', numba.float32[:]),
        ('stdrfo', numba.float32[:]),
        # predicted receiver function
        ('rfp', numba.float32[:]),
        ('tp', numba.float32[:]),
        # misfit/likelihood
        ('misfit', numba.float32),
        ('L', numba.float32)
        ]

@numba.jitclass(spec_rf)
class rf(object):
    def __init__(self):
        self.npts   = 0
        self.fs     = 0.
        return
    

        
    

    
# define type of disp object
disp_type   = numba.deferred_type()
disp_type.define(disp.class_type.instance_type)

# define type of rf object
rf_type     = numba.deferred_type()
rf_type.define(rf.class_type.instance_type)

####################################################
# Predefine the parameters for the data1d object
####################################################
spec_data=[# Rayleigh/Love dispersion data
            ('dispR',   disp_type),
            ('dispL',   disp_type),
            # radial/transverse receiver function data
            ('rfr',     rf_type),
            ('rft',     rf_type),
        ]

@numba.jitclass(spec_data)
class data1d(object):
    def __init__(self):
        self.dispR  = disp()
        self.dispL  = disp()
        self.rfr    = rf()
        self.rft    = rf()
        return
    
    
    
    
    