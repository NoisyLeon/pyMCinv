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
from __future__ import division
import numpy as np




class disp:
    """
    An object for handling dispersion data and computing misfit
    ==========================================================================
    ::: parameters :::
    --------------------------------------------------------------------------
    ::  phase   ::
    :   isotropic   :
    npper   - number of phase period
    pper    - phase period array
    pvelo   - observed phase velocities
    stdpvelo- uncertainties for observed phase velocities
    pvelp   - predicted phase velocities
    :   anisotropic :
    pphio   - observed phase velocity fast direction angle
    pampo   - observed phase velocity azimuthal anisotropic amplitude
    stdpphio- uncertainties for fast direction angle
    stdpampo- uncertainties for azimuthal anisotropic amplitude
    pphip   - predicted phase velocity fast direction angle
    pampp   - predicted phase velocity azimuthal anisotropic amplitude
    :   others  :
    isphase - phase dispersion data is stored or not
    pmisfit - phase dispersion misfit
    pL      - phase dispersion likelihood
    pS      - S function, L = exp(-0.5*S)
    --------------------------------------------------------------------------
    ::  group   ::
    ngper   - number of group period
    gper    - group period array
    gvelo   - observed group velocities
    stdgvelo- uncertainties for observed group velocities
    gvelp   - predicted group velocities
    :   others  :
    isgroup - group dispersion data is stored or not
    gmisfit - group dispersion misfit
    gL      - group dispersion likelihood
    --------------------------------------------------------------------------
    ::  others  ::
    misfit  - total misfit
    L       - total likelihood
    period  - common period array
    nper    - common number of periods
    ==========================================================================
    """

#    cdef float pvelo, stdpvelo, pvelp
    
    def __init__(self):
        self.npper  = 0
        self.ngper  = 0
        self.nper   = 0
        self.isphase= 0
        self.isgroup= 0
        self.pvelo  = np.array([])
        return
    
    def get_pmisfit(self):
        """
        Compute the misfit for phase velocities
        """
        if not self.isphase:
            print 'No phase velocity data stored'
            return False
        temp    = 0.
        for i in range(self.npper):
            temp+= (self.pvelo[i] - self.pvelp[i])**2/self.stdpvelo[i]**2
        self.pmisfit    = np.sqrt(temp/self.npper)
        self.pS         = temp
        if temp > 50.:
            temp        = np.sqrt(temp*50.)
        self.pL         = np.exp(-0.5 * temp)
        return True
#    
#    def get_gmisfit(self):
#        """
#        Compute the misfit for group velocities
#        """
#        if not self.isgroup:
#            print 'No group velocity data stored'
#            return False
#        temp    = 0.
#        for i in xrange(self.npper):
#            temp+= (self.gvelo[i] - self.gvelp[i])**2/self.stdgvelo[i]**2
#        self.gmisfit    = np.sqrt(temp/self.ngper)
#        self.gS         = temp
#        if temp > 50.:
#            temp= np.sqrt(temp*50.)
#        self.gL         = np.exp(-0.5 * temp)
#        return True
#    
#    def get_misfit(self):
#        """
#        Compute combined misfit
#        """
#        temp1   = 0.; temp2 = 0.
#        # misfit for phase velocities
#        if self.isphase:
#            for i in xrange(self.npper):
#                temp1   += (self.pvelo[i] - self.pvelp[i])**2/self.stdpvelo[i]**2
#            tS          = temp1
#            self.pS     = tS
#            misfit      = np.sqrt(temp1/self.npper)
#            if tS > 50.:
#                tS      = np.sqrt(tS*50.)
#            L           = np.exp(-0.5 * tS)
#            self.pmisfit    = misfit
#            self.pL         = L
#        # misfit for group velocities
#        if self.isgroup:
#            for i in xrange(self.ngper):
#                temp2   += (self.gvelo[i] - self.gvelp[i])**2/self.stdgvelo[i]**2
#            tS          = temp2
#            self.gS     = tS
#            misfit      = np.sqrt(temp2/self.ngper)
#            if tS > 50.:
#                tS      = np.sqrt(tS*50.)
#            L           = np.exp(-0.5 * tS)
#            self.gmisfit    = misfit
#            self.gL         = L
#        if (not self.isphase) and (not self.isgroup):
#            print 'No dispersion data stored!'
#            self.misfit = 0.
#            self.L      = 1.
#            return False
#        # misfit for both
#        temp    = temp1 + temp2
#        self.misfit     = np.sqrt(temp/(self.npper+self.ngper))
#        if temp > 50.:
#            temp = np.sqrt(temp*50.)
#        if temp > 50.:
#            temp = np.sqrt(temp*50.)
#        self.L          = np.exp(-0.5 * temp)
#        return True
#    
#    def get_misfit_tti(self):
#        """
#        compute misfit for inversion of tilted TI models, only applies to phase velocity dispersion
#        """
#        temp1   = 0.; temp2   = 0.; temp3   = 0.
#        for i in xrange(self.npper):
#            temp1   += (self.pvelo[i] - self.pvelp[i])**2/self.stdpvelo[i]**2
#            temp2   += (self.pampo[i] - self.pampp[i])**2/self.stdpampo[i]**2
#            phidiff = abs(self.pphio[i] - self.pphip[i])
#            if phidiff > 90.:
#                # # # phidiff -= 90.
#                phidiff = 180. - phidiff
#            temp3   += phidiff**2/self.stdpphio[i]**2
#        # # # temp2       *= 2.
#        # # # temp3       *= 2.
#        # # temp3       = 0. # debug !!!
#        self.pS     = temp1+temp2+temp3
#        tS          = temp1+temp2+temp3
#        self.pmisfit= np.sqrt(tS/3./self.npper)
#        if tS > 50.:
#            tS      = np.sqrt(tS*50.)
#        self.pL     = np.exp(-0.5 * tS)
#        return
#    
#    def get_res_tti(self):
#        r1  = []; r2 = []; r3 = []
#        for i in xrange(self.npper):
#            r1.append((self.pvelo[i] - self.pvelp[i])/self.stdpvelo[i])
#            r2.append((self.pampo[i] - self.pampp[i])/self.stdpampo[i])
#            phidiff = abs(self.pphio[i] - self.pphip[i])
#            if phidiff > 90.:
#                phidiff = 180. - phidiff
#            r3.append(phidiff/self.stdpphio[i])
#        r1  = np.array(r1, dtype = np.float32)
#        r2  = np.array(r2, dtype = np.float32)
#        r3  = np.array(r3, dtype = np.float32)
#        return r1, r2, r3
#    
#    def get_res_pvel(self):
#        r  = []
#        for i in xrange(self.npper):
#            r.append((self.pvelo[i] - self.pvelp[i])/self.stdpvelo[i])
#        r   = np.array(r, dtype = np.float32)
#        return r
        
