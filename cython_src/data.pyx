# -*- coding: utf-8 -*-
# distutils: language=c++
"""
Module for handling input data for joint inversion.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
from __future__ import division
import numpy as np
cimport numpy as np
from libcpp cimport bool
from libc.math cimport sqrt, exp, fabs
from libc.stdio cimport printf
cimport cython
#from cython.view cimport array as cvarray
#from cython.parallel import parallel, prange, threadid
#cimport openmp
#cimport data.pxd


        


cdef class rf:
    """
    An object for handling receiver function data and computing misfit
    ==========================================================================
    ::: parameters :::
    fs      - sampling rate
    npts    - number of data points
    rfo     - observed receiver function array
    to      - time array of observed receiver function
    stdrfo  - uncerntainties in observed receiver function
    rfp     - predicted receiver function array
    tp      - time array of predicted receiver function
    misfit  - misfit value
    L       - likelihood value
    ==========================================================================
    """

    def __init__(self):
        self.npts   = 0
        self.fs     = 0.
        return
    
    def readrftxt(self, str infname):
        """
        Read input txt file of receiver function
        ==========================================================================
        ::: input :::
        infname     - input file name
        ::: output :::
        receiver function data is stored in self
        ==========================================================================
        """
        cdef np.ndarray inArr  
        if self.npts > 0:
            print 'receiver function data is already stored!'
            return False
        inArr 		 = np.loadtxt(infname, dtype=np.float32)
        self.npts   = inArr.shape[0]        
        self.to     = inArr[:,0]
        self.rfo    = inArr[:,1]
        try:
            self.stdrfo = inArr[:,2]
        except IndexError:
            self.stdrfo = np.ones(self.npts, dtype=np.float32)*0.1
        self.fs     = 1./(self.to[1] - self.to[0])
        return True
#    
    def writerftxt(self, str outfname, float tf=10.):
        """
        Write receiver function data to a txt file
        ==========================================================================
        ::: input :::
        outfname    - output file name
        tf          - end time point for trim
        ::: output :::
        a txt file contains predicted and observed receiver function data
        ==========================================================================
        """
        cdef np.ndarray outArr
        cdef str header  
        cdef int nout 
        if self.npts == 0:
            print 'receiver function data is not stored!'
            return False
        nout    = int(self.fs*tf)+1
        nout    = min(nout, self.npts)
        outArr  = np.append(self.tp[:nout], self.rfp[:nout])
        outArr  = np.append(outArr, self.to[:nout])
        outArr  = np.append(outArr, self.rfo[:nout])
        outArr  = np.append(outArr, self.stdrfo[:nout])
        outArr  = outArr.reshape((5, nout))    
        outArr  = outArr.T
        header  = 'tp rfp to rfo stdrfo'
        np.savetxt(outfname, outArr, fmt='%g', header = header)
        return True
    
    @cython.boundscheck(False)
    cdef bool get_misfit_incompatible(self, float rffactor=40.) nogil:
        """
        compute misfit when the time array of predicted and observed data is incompatible, quite slow!
        ==============================================================================
        ::: input :::
        rffactor    - factor for downweighting the misfit for likelihood computation
        ==============================================================================
        """
        cdef float temp = 0.
        cdef int k      = 0
        cdef Py_ssize_t i, j
        cdef float tS
        if self.npts == 0:
            self.misfit = 0.
            self.L      = 1.
            return False
        j = 0
        for i in range(self.npts):
            while (self.tp[j] < self.to[i]):
                if self.to[i] == self.tp[j] and self.to[i] <= 10 and self.to[i] >= 0 :
                    temp    += ( (self.rfo[i] - self.rfp[j])**2 / (self.stdrfo[i]**2) )
                    k       += 1
                    break
        self.misfit = sqrt(temp/k)
        tS          = temp/rffactor
        if tS > 50.:
            tS      = sqrt(tS*50.)
        self.L      = exp(-0.5 * tS)
        return True
#    
    @cython.boundscheck(False)
    cdef bool get_misfit(self, float rffactor=40.) nogil:
        """
        Compute misfit for receiver function
        ==============================================================================
        ::: input :::
        rffactor    - factor for downweighting the misfit for likelihood computation
        ==============================================================================
        """
        cdef float temp = 0.
        cdef Py_ssize_t k      = 0
        cdef Py_ssize_t i
        cdef float tS
        if self.npts == 0:
            self.misfit = 0.
            self.L      = 1.
            return False
        for i in range(self.npts):
            if self.to[i] != self.tp[i]:
                printf('Incompatible time arrays!')
                return self.get_misfit_incompatible(rffactor=rffactor)
            if self.to[i] >= 0:
                temp    += ( (self.rfo[i] - self.rfp[i])**2 / (self.stdrfo[i]**2) )
                k       += 1
            if self.to[i] > 10:
                break
        self.misfit = sqrt(temp/(<float>k))
        tS          = temp/rffactor
        if tS > 50.:
            tS      = sqrt(tS*50.)
        self.L      = exp(-0.5 * tS)
        return True
    
cdef class disp:
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

    
    def __init__(self):
        self.npper  = 0
        self.ngper  = 0
        self.nper   = 0
        self.isphase= False
        self.isgroup= False
        return
    
    #----------------------------------------------------
    # I/O functions
    #----------------------------------------------------
    
    def readdisptxt(self, str infname, str dtype='ph'):
        """
        Read input txt file of dispersion curve
        ==========================================================================
        ::: input :::
        infname     - input file name
        dtype       - data type (phase/group)
        ::: output :::
        dispersion curve is stored
        ==========================================================================
        """
        cdef np.ndarray inArr
        dtype   = dtype.lower()
        if dtype == 'ph' or dtype == 'phase':
            if self.isphase:
                print 'phase velocity data is already stored!'
                return False
            inArr 		 = np.loadtxt(infname, dtype=np.float32)
            self.pper  = inArr[:,0]
            self.pvelo = inArr[:,1]
            self.npper = self.pper.size
            try:
                self.stdpvelo= inArr[:,2]
            except IndexError:
                self.stdpvelo= np.ones(self.npper, dtype=np.float32)
            self.isphase = True
        elif dtype == 'gr' or dtype == 'group':
            if self.isgroup:
                print 'group velocity data is already stored!'
                return False
            inArr 	  = np.loadtxt(infname, dtype=np.float32)
            self.gper = inArr[:,0]
            self.gvelo= inArr[:,1]
            self.ngper= self.gper.size
            try:
                self.stdgvelo= inArr[:,2]
            except IndexError:
                self.stdgvelo= np.ones(self.ngper, dtype=np.float32)
            self.isgroup  = True
        else:
            raise ValueError('Unexpected dtype = '+dtype)
        return True
    
    def writedisptxt(self, str outfname, str dtype='ph'):
        """
        Write dispersion curve to a txt file
        ==========================================================================
        ::: input :::
        outfname    - output file name
        dtype       - data type (phase/group)
        ::: output :::
        a txt file contains predicted and observed dispersion data
        ==========================================================================
        """
        cdef np.ndarray outArr
        cdef str header  
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            outArr  = np.append(self.pper, self.pvelp)
            outArr  = np.append(outArr, self.pvelo)
            outArr  = np.append(outArr, self.stdpvelo)
            outArr  = outArr.reshape((4, self.npper))
            outArr  = outArr.T
            header  = 'pper pvelp pvelo stdpvelo'
            np.savetxt(outfname, outArr, fmt='%g', header=header)
        elif dtype == 'gr' or dtype == 'group':
            if not self.isgroup:
                print 'group velocity data is not stored!'
                return False
            outArr  = np.append(self.gper, self.gvelp)
            outArr  = np.append(outArr, self.gvelo)
            outArr  = np.append(outArr, self.stdgvelo)
            outArr  = outArr.reshape((4, self.ngper))
            outArr  = outArr.T 
            header  = 'gper gvelp gvelo stdgvelo'
            np.savetxt(outfname, outArr, fmt='%g', header=header)
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    
    def readaziamptxt(self, str infname, str dtype='ph'):
        """
        Read input txt file of azimuthal amplitude
        ==========================================================================
        ::: input :::
        infname     - input file name
        dtype       - data type (phase/group)
        ::: output :::
        azimuthal amplitude is stored
        ==========================================================================
        """
        cdef np.ndarray inArr
        dtype   = dtype.lower()
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            inArr 		= np.loadtxt(infname, dtype=np.float32)
            if not np.allclose(self.pper , inArr[:,0]):
                print 'inconsistent period array !'
                return False
            self.pampo= inArr[:,1]
            self.npper= self.pper.size
            try:
                self.stdpampo= inArr[:,2]
            except IndexError:
                self.stdpampo= np.ones(self.npper, dtype=np.float32)
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    def writeaziamptxt(self, str outfname, str dtype='ph'):
        """
        Write azimuthal amplitude to a txt file
        ==========================================================================
        ::: input :::
        outfname    - output file name
        dtype       - data type (phase/group)
        ::: output :::
        a txt file contains predicted and observed dispersion data
        ==========================================================================
        """
        cdef np.ndarray outArr
        cdef str header  
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            outArr  = np.append(self.pper, self.pampp)
            outArr  = np.append(outArr, self.pampo)
            outArr  = np.append(outArr, self.stdpampo)
            outArr  = outArr.reshape((4, self.npper))
            outArr  = outArr.T
            header  = 'pper pampp pampo stdpampo'
            np.savetxt(outfname, outArr, fmt='%g')
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    def readaziphitxt(self, str infname, str dtype='ph'):
        """
        Read input txt file of fast direction azimuth
        ==========================================================================
        ::: input :::
        infname     - input file name
        dtype       - data type (phase/group)
        ::: output :::
        fast direction azimuth is stored 
        ==========================================================================
        """
        cdef np.ndarray inArr
        dtype   = dtype.lower()
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            inArr 		= np.loadtxt(infname, dtype=np.float32)
            if not np.allclose(self.pper , inArr[:, 0]):
                print 'inconsistent period array !'
                return False
            self.pphio= inArr[:,1]
            self.npper= self.pper.size
            try:
                self.stdpphio= inArr[:,2]
            except IndexError:
                self.stdpphio= np.ones(self.npper, dtype=np.float32)
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    def writeaziphitxt(self, str outfname, str dtype='ph'):
        """
        Write fast direction azimuth to a txt file
        ==========================================================================
        ::: input :::
        outfname    - output file name
        dtype       - data type (phase/group)
        ::: output :::
        a txt file contains predicted and observed dispersion data
        ==========================================================================
        """
        cdef np.ndarray outArr
        cdef str header  
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            outArr  = np.append(self.pper, self.pphip)
            outArr  = np.append(outArr, self.pphio)
            outArr  = np.append(outArr, self.stdpphio)
            outArr  = outArr.reshape((4, self.npper))
            outArr  = outArr.T
            header  = 'pper pphip pphio stdpphio'
            np.savetxt(outfname, outArr, fmt='%g', header=header)
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
        
    def writedispttitxt(self, str outfname, str dtype='ph'):
        """
        Write dispersion curve to a txt file
        ==========================================================================
        ::: input :::
        outfname    - output file name
        dtype       - data type (phase/group)
        ::: output :::
        a txt file contains predicted and observed dispersion data
        ==========================================================================
        """
        cdef np.ndarray outArr
        cdef str header  
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            outArr  = np.append(self.pper, self.pvelp)
            outArr  = np.append(outArr, self.pvelo)
            outArr  = np.append(outArr, self.stdpvelo)
            # azimuthal amplitude
            outArr  = np.append(outArr, self.pampp)
            outArr  = np.append(outArr, self.pampo)
            outArr  = np.append(outArr, self.stdpampo)
            # fast-direction azimuth
            outArr  = np.append(outArr, self.pphip)
            outArr  = np.append(outArr, self.pphio)
            outArr  = np.append(outArr, self.stdpphio)
            outArr  = outArr.reshape((10, self.npper))
            outArr  = outArr.T
            header  = 'pper pvelp pvelo stdpvelo pampp pampo stdpampo pphip pphio stdpphio'
            np.savetxt(outfname, outArr, fmt='%g', header=header)
        return True
    
    #----------------------------------------------------
    # functions computing misfit
    #----------------------------------------------------
    @cython.boundscheck(False)
    cdef bool get_pmisfit(self) nogil:
        """
        Compute the misfit for phase velocities
        """
        cdef float temp = 0.
        cdef Py_ssize_t i
        if not self.isphase :
            printf('No phase velocity data stored')
            return False
        for i in range(self.npper):
            temp = temp + (self.pvelo[i] - self.pvelp[i])**2/self.stdpvelo[i]**2
        self.pmisfit    = sqrt(temp/self.npper)
        self.pS         = temp
        if temp > 50.:
            temp        = sqrt(temp*50.)
        self.pL         = exp(-0.5 * temp)
        return True
    
    @cython.boundscheck(False)
    cdef bool get_gmisfit(self) nogil:
        """
        Compute the misfit for group velocities
        """
        cdef float temp = 0.
        cdef Py_ssize_t i
        if not self.isgroup:
            printf('No group velocity data stored')
            return False
        for i in range(self.npper):
            temp+= (self.gvelo[i] - self.gvelp[i])**2/self.stdgvelo[i]**2
        self.gmisfit    = sqrt(temp/self.ngper)
        self.gS         = temp
        if temp > 50.:
            temp= sqrt(temp*50.)
        self.gL         = exp(-0.5 * temp)
        return True
    
    @cython.boundscheck(False)
    cdef bool get_misfit(self) nogil:
        """
        Compute combined misfit
        """
        cdef float temp1    = 0. 
        cdef float temp2    = 0.
        cdef float tS, L, misfit, temp
        cdef Py_ssize_t i
        # misfit for phase velocities
        if self.isphase:
            for i in range(self.npper):
                temp1   += (self.pvelo[i] - self.pvelp[i])**2/self.stdpvelo[i]**2
            tS          = temp1
            self.pS     = tS
            misfit      = sqrt(temp1/self.npper)
            if tS > 50.:
                tS      = sqrt(tS*50.)
            L           = exp(-0.5 * tS)
            self.pmisfit    = misfit
            self.pL         = L
        # misfit for group velocities
        if self.isgroup:
            for i in range(self.ngper):
                temp2   += (self.gvelo[i] - self.gvelp[i])**2/self.stdgvelo[i]**2
            tS          = temp2
            self.gS     = tS
            misfit      = sqrt(temp2/self.ngper)
            if tS > 50.:
                tS      = sqrt(tS*50.)
            L           = exp(-0.5 * tS)
            self.gmisfit    = misfit
            self.gL         = L
        if (not self.isphase) and (not self.isgroup):
            printf('No dispersion data stored!')
            self.misfit = 0.
            self.L      = 1.
            return False
        # misfit for both
        temp    = temp1 + temp2
        self.S          = temp
        self.misfit     = sqrt(temp/(self.npper+self.ngper))
        if temp > 50.:
            temp = sqrt(temp*50.)
        if temp > 50.:
            temp = sqrt(temp*50.)
        self.L          = exp(-0.5 * temp)
        return True
    
    @cython.boundscheck(False)
    cdef void get_misfit_tti(self) nogil:
        """
        compute misfit for inversion of tilted TI models, only applies to phase velocity dispersion
        """
        cdef float temp1   = 0.
        cdef float temp2   = 0.
        cdef float temp3   = 0.
        cdef Py_ssize_t i
        cdef float phidiff, tS
        for i in range(self.npper):
            temp1   += (self.pvelo[i] - self.pvelp[i])**2/self.stdpvelo[i]**2
            temp2   += (self.pampo[i] - self.pampp[i])**2/self.stdpampo[i]**2
            phidiff = fabs(self.pphio[i] - self.pphip[i])
            if phidiff > 90.:
                phidiff = 180. - phidiff
            temp3   += phidiff**2/self.stdpphio[i]**2
        # # # temp2       *= 2.
        # # # temp3       *= 2.
        # # temp3       = 0. # debug !!!
        self.pS     = temp1+temp2+temp3
        tS          = temp1+temp2+temp3
        self.pmisfit= sqrt(tS/3./self.npper)
        if tS > 50.:
            tS      = sqrt(tS*50.)
        self.pL     = exp(-0.5 * tS)
        return
#    
    cpdef get_res_tti(self):
        cdef float[:] r1, r2, r3    
        cdef Py_ssize_t i
        cdef float phidiff
        
        r1   = np.zeros(self.npper, dtype=np.float32)
        r2   = np.zeros(self.npper, dtype=np.float32)
        r3   = np.zeros(self.npper, dtype=np.float32)
        for i in range(self.npper):
            r1[i]   = (self.pvelo[i] - self.pvelp[i])/self.stdpvelo[i]
            print r1[i]
            r2[i]   = (self.pampo[i] - self.pampp[i])/self.stdpampo[i]
            print r2[i]
            phidiff = abs(self.pphio[i] - self.pphip[i])
            if phidiff > 90.:
                phidiff = 180. - phidiff
            r3[i]   = phidiff/self.stdpphio[i]
            print r3[i]
        return r1, r2, r3
    
    cpdef get_res_pvel(self):
        cdef float[:] r  
        cdef Py_ssize_t i
        
        r           = np.zeros(self.npper, dtype=np.float32)
        for i in range(self.npper):
            r[i]    = (self.pvelo[i] - self.pvelp[i])/self.stdpvelo[i]
        return r
# 
        
cdef class data1d:
    """
    An object for handling input data for inversion
    ==========================================================================
    ::: parameters :::
    dispR   - Rayleigh wave dispersion data
    dispL   - Love wave dispersion data
    rfr     - radial receiver function data
    rft     - transverse receiver function data
    misfit  - misfit value
    L       - likelihood value
    ==========================================================================
    """

    
    def __init__(self):
        self.dispR  = disp()
        self.dispL  = disp()
        self.rfr    = rf()
        self.rft    = rf()
        return
    
    @cython.boundscheck(False)
    cdef public void get_misfit(self, float wdisp, float rffactor) nogil:
        """
        Compute combined misfit
        ==========================================================================================
        ::: input :::
        wdisp       - relative weigh for dispersion data ( 0.~1. )
        rffactor    - factor for downweighting the misfit for likelihood computation of rf
        ==========================================================================================
        """
        self.dispR.get_misfit()
        self.rfr.get_misfit(rffactor = rffactor)
        # compute combined misfit and likelihood
        self.misfit = wdisp*self.dispR.misfit + (1.-wdisp)*self.rfr.misfit
        self.L      = ((self.dispR.L)**wdisp)*((self.rfr.L)**(1.-wdisp))
        return
#    
#    def get_misfit_tti(self):
#        """
#        compute misfit for inversion of tilted TI models, only applies to phase velocity dispersion
#        """
#        self.dispR.get_misfit_tti()
#        self.dispL.get_pmisfit()
#        self.misfit = sqrt((self.dispR.pS + self.dispL.pS)/(3.*self.dispR.npper + self.dispL.npper) )
#        tS          = 0.5*(self.dispR.pS + self.dispL.pS)
#        if tS > 50.:
#            tS      = sqrt(tS*50.)
#        if tS > 50.:
#            tS      = sqrt(tS*50.)
#        if tS > 50.:
#            tS      = sqrt(tS*50.)
#        # if tS > 50.:
#        #     tS      = sqrt(tS*50.)
#        self.L      = exp(-0.5 * tS)
#        return
#
#    def printtest(self):
#        i=np.int32(3)
#        print 'accept a model', (i, self.L)
#    
#    def get_res_tti(self):
#        r1, r2, r3  = self.dispR.get_res_tti()
#        r4          = self.dispL.get_res_pvel()
        return r1, r2, r3, r4