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
            indisp.stdpvelo= np.ones(indisp.npper, dtype=np.float32)
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
            indisp.stdgvelo= np.ones(indisp.ngper, dtype=np.float32)
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
    
    def get_pmisfit(self):
        temp    = 0.
        for i in xrange(self.npper):
            temp+= (self.pvelo[i] - self.pvelp[i])**2/self.stdpvelo[i]**2
            
        misfit  = np.sqrt(temp/self.npper)
        if temp > 50.:
            temp= np.sqrt(temp*50.)
        L       = np.exp(-0.5 * temp)
        
        self.pmisfit    = misfit
        self.pL         = L
        return
    
    def get_gmisfit(self):
        temp    = 0.
        for i in xrange(self.npper):
            temp+= (self.gvelo[i] - self.gvelp[i])**2/self.stdgvelo[i]**2
            
        misfit  = np.sqrt(temp/self.ngper)
        if temp > 50.:
            temp= np.sqrt(temp*50.)
        L       = np.exp(-0.5 * temp)
        
        self.gmisfit    = misfit
        self.gL         = L
        return
    
    def get_misfit(self):
        temp1   = 0.; temp2 = 0.
        # misfit for phase velocities
        if self.isphase:
            for i in xrange(self.npper):
                temp1   += (self.pvelo[i] - self.pvelp[i])**2/self.stdpvelo[i]**2
            tS          = temp1
            misfit      = np.sqrt(temp1/self.npper)
            if tS > 50.:
                tS      = np.sqrt(tS*50.)
            L           = np.exp(-0.5 * tS)
            
            self.pmisfit    = misfit
            self.pL         = L
        # misfit for group velocities
        if self.isgroup:
            for i in xrange(self.ngper):
                temp2   += (self.gvelo[i] - self.gvelp[i])**2/self.stdgvelo[i]**2
                
            tS          = temp2
            misfit      = np.sqrt(temp2/self.ngper)
            if tS > 50.:
                tS      = np.sqrt(tS*50.)
            L           = np.exp(-0.5 * tS)
            self.gmisfit    = misfit
            self.gL         = L
        # misfit for both
        temp    = temp1 + temp2
        self.misfit     = np.sqrt(temp/(self.npper+self.ngper))
        if temp > 50.:
            temp = np.sqrt(temp*50.)
        if temp > 50.:
            temp = np.sqrt(temp*50.)
        self.L          = np.exp(-0.5 * temp)
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
    
    def get_misfit_incompatible(self):
        temp    = 0.
        k       = 0
        factor  = 40.
        for i in xrange(self.npts):
            for j in xrange(self.tp.size):
                if self.to[i] == self.tp[j] and self.to[i] <= 10 and self.to[i] >= 0 :
                    temp    += ( (self.rfo[i] - self.rfp[j])**2 / (self.stdrfo[i]**2) )
                    k = k+1
                    break
        self.misfit = np.sqrt(temp/k)
        tS      = temp/factor
        if tS > 50.:
            tS      = np.sqrt(tS*50.)
        self.L      = np.exp(-0.5 * tS)
        return
    
    def get_misfit(self):
        temp    = 0.
        k       = 0
        factor  = 40.
        for i in xrange(self.npts):
            if self.to[i] != self.tp[i]:
                # # # raise ValueError('Incompatible time arrays!')
                print ('Incompatible time arrays!')
                self.get_misfit_incompatible()
                return
            if self.to[i] >= 0:
                temp    += ( (self.rfo[i] - self.rfp[i])**2 / (self.stdrfo[i]**2) )
                k       += 1
            if self.to[i] > 10:
                break
        self.misfit = np.sqrt(temp/k)
        tS      = temp/factor
        if tS > 50.:
            tS      = np.sqrt(tS*50.)
        self.L      = np.exp(-0.5 * tS)
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
            # misfit/likelihood
            ('misfit',  numba.float32),
            ('L',       numba.float32)
        ]

@numba.jitclass(spec_data)
class data1d(object):
    def __init__(self):
        self.dispR  = disp()
        self.dispL  = disp()
        self.rfr    = rf()
        self.rft    = rf()
        return
    
    
    
    
    
    
    