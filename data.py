# -*- coding: utf-8 -*-
# distutils: language=c++
"""
Module for handling input data for Bayesian Monte Carlo inversion.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import numpy as np
import matplotlib.pyplot as plt

class rf(object):
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
    
    def readrftxt(self, infname):
        """
        Read input txt file of receiver function
        ==========================================================================
        ::: input :::
        infname     - input file name
        ::: output :::
        receiver function data is stored in self
        ==========================================================================
        """
        if self.npts > 0:
            print 'receiver function data is already stored!'
            return False
        inArr 		= np.loadtxt(infname, dtype=np.float64)
        self.npts   = inArr.shape[0]        
        self.to     = inArr[:,0]
        self.rfo    = inArr[:,1]
        try:
            self.stdrfo = inArr[:,2]
        except IndexError:
            self.stdrfo = np.ones(self.npts, dtype=np.float64)*0.1
        self.fs     = 1./(self.to[1] - self.to[0])
        return True
    
    def get_rf(self, indata):
        """
        get input receiver function data
        ==========================================================================
        ::: input :::
        indata      - input data array (3, N)
        ::: output :::
        receiver function data is stored in self
        ==========================================================================
        """
        if self.npts > 0:
            print 'receiver function data is already stored!'
            return False
        self.npts   = indata.shape[1]        
        self.to     = indata[0, :]
        self.rfo    = indata[1, :]
        try:
            self.stdrfo = indata[2, :]
        except IndexError:
            self.stdrfo = np.ones(self.npts, dtype=np.float64)*0.1
        self.fs     = 1./(self.to[1] - self.to[0])
        return True
  
    def writerftxt(self, outfname, tf=10.):
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
    
    # def get_misfit_incompatible(self, rffactor=40.):
    #     """
    #     compute misfit when the time array of predicted and observed data is incompatible, quite slow!
    #     ==============================================================================
    #     ::: input :::
    #     rffactor    - factor for downweighting the misfit for likelihood computation
    #     ==============================================================================
    #     """
    #     if self.npts == 0:
    #         self.misfit = 0.
    #         self.L      = 1.
    #         return 0
    #     j = 0
    #     for i in range(self.npts):
    #         while (self.tp[j] < self.to[i]):
    #             if self.to[i] == self.tp[j] and self.to[i] <= 10 and self.to[i] >= 0 :
    #                 temp    += ( (self.rfo[i] - self.rfp[j])**2 / (self.stdrfo[i]**2) )
    #                 k       += 1
    #                 break
    #             j   += 1
    #     self.misfit = sqrt(temp/k)
    #     tS          = temp/rffactor
    #     if tS > 50.:
    #         tS      = sqrt(tS*50.)
    #     self.L      = exp(-0.5 * tS)
    #     return 1

    def get_misfit(self, rffactor=40.):
        """
        Compute misfit for receiver function
        ==============================================================================
        ::: input :::
        rffactor    - factor for downweighting the misfit for likelihood computation
        ==============================================================================
        """
        if self.npts == 0:
            self.misfit = 0.
            self.L      = 1.
            return False
        if not np.allclose(self.to, self.tp):
            raise ValueError('Incompatable time arrays for predicted and observed rf!')
        ind         = (self.to<10.)*(self.to>=0.)
        temp        = ((self.rfo[ind] - self.rfp[ind])**2 / (self.stdrfo[ind]**2)).sum()
        k           = (self.rfo[ind]).size
        self.misfit = np.sqrt(temp/k)
        tS          = temp/rffactor
        if tS > 50.:
            tS      = np.sqrt(tS*50.)
        self.L      = np.exp(-0.5 * tS)
        return True
    
    def plot(self, showfig=True, prediction=False):
        if self.npts == 0:
            print 'No data for plotting!'
            return
        # First illustrate basic pyplot interface, using defaults where possible.
        plt.figure()
        ax  = plt.subplot()
        plt.errorbar(self.to, self.rfo, yerr=self.stdrfo)
        if prediction:
            plt.plot(self.to, self.rfp, 'r--', lw=3)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlabel('time (sec)', fontsize=30)
        plt.ylabel('ampltude', fontsize=30)
        plt.title('receiver function', fontsize=30)
        if showfig:
            plt.show()
    
    
class disp(object):
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
    
    def readdisptxt(self, infname, dtype='ph'):
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
        dtype   = dtype.lower()
        if dtype == 'ph' or dtype == 'phase':
            if self.isphase:
                print 'phase velocity data is already stored!'
                return False
            inArr 	    = np.loadtxt(infname, dtype=np.float64)
            self.pper   = inArr[:,0]
            self.pvelo  = inArr[:,1]
            self.npper  = self.pper.size
            try:
                self.stdpvelo= inArr[:,2]
            except IndexError:
                self.stdpvelo= np.ones(self.npper, dtype=np.float64)
            self.isphase = True
        elif dtype == 'gr' or dtype == 'group':
            if self.isgroup:
                print 'group velocity data is already stored!'
                return False
            inArr 	  = np.loadtxt(infname, dtype=np.float64)
            self.gper = inArr[:,0]
            self.gvelo= inArr[:,1]
            self.ngper= self.gper.size
            try:
                self.stdgvelo= inArr[:,2]
            except IndexError:
                self.stdgvelo= np.ones(self.ngper, dtype=np.float64)
            self.isgroup  = True
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    def get_disp(self, indata, dtype='ph'):
        """
        get dispersion curve data from a input numpy array
        ==========================================================================
        ::: input :::
        indata      - input array (3, N)
        dtype       - data type (phase/group)
        ::: output :::
        dispersion curve is stored
        ==========================================================================
        """
        dtype   = dtype.lower()
        if dtype == 'ph' or dtype == 'phase':
            if self.isphase:
                print 'phase velocity data is already stored!'
                return False
            self.pper   = indata[0, :]
            self.pvelo  = indata[1, :]
            self.npper  = self.pper.size
            try:
                self.stdpvelo= indata[2, :]
            except IndexError:
                self.stdpvelo= np.ones(self.npper, dtype=np.float64)
            self.isphase = True
        elif dtype == 'gr' or dtype == 'group':
            if self.isgroup:
                print 'group velocity data is already stored!'
                return False
            self.gper = indata[0, :]
            self.gvelo= indata[1, :]
            self.ngper= self.gper.size
            try:
                self.stdgvelo= indata[2, :]
            except IndexError:
                self.stdgvelo= np.ones(self.ngper, dtype=np.float64)
            self.isgroup  = True
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    def writedisptxt(self, outfname, dtype='ph'):
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
    
    
    def readaziamptxt(self, infname, dtype='ph'):
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
        dtype   = dtype.lower()
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            inArr 	    = np.loadtxt(infname, dtype=np.float64)
            if not np.allclose(self.pper , inArr[:,0]):
                print 'inconsistent period array !'
                return False
            self.pampo  = inArr[:,1]
            self.npper  = self.pper.size
            try:
                self.stdpampo   = inArr[:,2]
            except IndexError:
                self.stdpampo   = np.ones(self.npper, dtype=np.float64)
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    def writeaziamptxt(self, outfname, dtype='ph'):
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
    
    def readaziphitxt(self, infname, dtype='ph'):
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
        dtype   = dtype.lower()
        if dtype == 'ph' or dtype == 'phase':
            if not self.isphase:
                print 'phase velocity data is not stored!'
                return False
            inArr 		= np.loadtxt(infname, dtype=np.float64)
            if not np.allclose(self.pper , inArr[:, 0]):
                print 'inconsistent period array !'
                return False
            self.pphio  = inArr[:,1]
            self.npper  = self.pper.size
            try:
                self.stdpphio   = inArr[:,2]
            except IndexError:
                self.stdpphio   = np.ones(self.npper, dtype=np.float64)
        else:
            raise ValueError('Unexpected dtype: '+dtype)
        return True
    
    def writeaziphitxt(self, outfname, dtype='ph'):
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
        
    def writedispttitxt(self, outfname, dtype='ph'):
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
    def get_pmisfit(self):
        """
        Compute the misfit for phase velocities
        """
        if not self.isphase :
            print('No phase velocity data stored')
            return False
        temp            = ((self.pvelo - self.pvelp)**2/self.stdpvelo**2).sum()
        self.pmisfit    = np.sqrt(temp/self.npper)
        self.pS         = temp
        if temp > 50.:
            temp        = np.sqrt(temp*50.)
        self.pL         = np.exp(-0.5 * temp)
        return True
    
    def get_gmisfit(self):
        """
        Compute the misfit for group velocities
        """
        if not self.isgroup:
            print('No group velocity data stored')
            return False
        temp            = ((self.gvelo - self.gvelp)**2/self.stdgvelo**2).sum()
        self.gmisfit    = np.sqrt(temp/self.ngper)
        self.gS         = temp
        if temp > 50.:
            temp        = np.sqrt(temp*50.)
        self.gL         = np.exp(-0.5 * temp)
        return True
    
    def get_misfit(self):
        """
        Compute combined misfit
        """
        # misfit for phase velocities
        temp1           = 0.
        temp2           = 0.
        if self.isphase:
            temp1       += ((self.pvelo - self.pvelp)**2/self.stdpvelo**2).sum()
            tS          = temp1
            self.pS     = tS
            misfit      = np.sqrt(temp1/self.npper)
            if tS > 50.:
                tS      = np.sqrt(tS*50.)
            L           = np.exp(-0.5 * tS)
            self.pmisfit    = misfit
            self.pL         = L
        # misfit for group velocities
        if self.isgroup:
            temp2       += ((self.gvelo - self.gvelp)**2/self.stdgvelo**2).sum()
            tS          = temp2
            self.gS     = tS
            misfit      = np.sqrt(temp2/self.ngper)
            if tS > 50.:
                tS      = np.sqrt(tS*50.)
            L           = np.exp(-0.5 * tS)
            self.gmisfit    = misfit
            self.gL         = L
        if (not self.isphase) and (not self.isgroup):
            printf('No dispersion data stored!')
            self.misfit = 0.
            self.L      = 1.
            return False
        # misfit for both
        temp            = temp1 + temp2
        self.S          = temp
        self.misfit     = np.sqrt(temp/(self.npper+self.ngper))
        if temp > 50.:
            temp        = np.sqrt(temp*50.)
        if temp > 50.:
            temp        = np.sqrt(temp*50.)
        self.L          = np.exp(-0.5 * temp)
        return True
    
    def get_misfit_tti(self):
        """
        compute misfit for inversion of tilted TI models, only applies to phase velocity dispersion
        """
        temp1                   = ((self.pvelo - self.pvelp)**2/self.stdpvelo**2).sum()
        temp2                   = ((self.pampo - self.pampp)**2/self.stdpampo**2).sum()
        phidiff                 = abs(self.pphio - self.pphip)
        phidiff[phidiff>90.]    = 180. - phidiff[phidiff>90.]
        temp3                   = (phidiff**2/self.stdpphio**2).sum()
        self.pS                 = temp1+temp2+temp3
        tS                      = temp1+temp2+temp3
        self.pmisfit            = np.sqrt(tS/3./self.npper)
        if tS > 50.:
            tS                  = np.sqrt(tS*50.)
        self.pL                 = np.exp(-0.5 * tS)
        return
# #    
#     cpdef get_res_tti(self):
#         cdef float[:] r1, r2, r3    
#         cdef Py_ssize_t i
#         cdef float phidiff
#         
#         r1   = np.zeros(self.npper, dtype=np.float64)
#         r2   = np.zeros(self.npper, dtype=np.float64)
#         r3   = np.zeros(self.npper, dtype=np.float64)
#         for i in range(self.npper):
#             r1[i]   = (self.pvelo[i] - self.pvelp[i])/self.stdpvelo[i]
#             print r1[i]
#             r2[i]   = (self.pampo[i] - self.pampp[i])/self.stdpampo[i]
#             print r2[i]
#             phidiff = abs(self.pphio[i] - self.pphip[i])
#             if phidiff > 90.:
#                 phidiff = 180. - phidiff
#             r3[i]   = phidiff/self.stdpphio[i]
#             print r3[i]
#         return r1, r2, r3
#     
#     cpdef get_res_pvel(self):
#         cdef float[:] r  
#         cdef Py_ssize_t i
#         
#         r           = np.zeros(self.npper, dtype=np.float64)
#         for i in range(self.npper):
#             r[i]    = (self.pvelo[i] - self.pvelp[i])/self.stdpvelo[i]
#         return r
# # 
        
class data1d(object):
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
    
    def get_misfit(self, wdisp=0.2, rffactor=40.):
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
# #    
# #    def get_misfit_tti(self):
# #        """
# #        compute misfit for inversion of tilted TI models, only applies to phase velocity dispersion
# #        """
# #        self.dispR.get_misfit_tti()
# #        self.dispL.get_pmisfit()
# #        self.misfit = sqrt((self.dispR.pS + self.dispL.pS)/(3.*self.dispR.npper + self.dispL.npper) )
# #        tS          = 0.5*(self.dispR.pS + self.dispL.pS)
# #        if tS > 50.:
# #            tS      = sqrt(tS*50.)
# #        if tS > 50.:
# #            tS      = sqrt(tS*50.)
# #        if tS > 50.:
# #            tS      = sqrt(tS*50.)
# #        # if tS > 50.:
# #        #     tS      = sqrt(tS*50.)
# #        self.L      = exp(-0.5 * tS)
# #        return
# #
# #    def printtest(self):
# #        i=np.int32(3)
# #        print 'accept a model', (i, self.L)
# #    
# #    def get_res_tti(self):
# #        r1, r2, r3  = self.dispR.get_res_tti()
# #        r4          = self.dispL.get_res_pvel()
#         return r1, r2, r3, r4