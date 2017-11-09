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
    """
    Read input txt file of dispersion curve
    ==========================================================================
    ::: input :::
    infname     - input file name
    indisp      - disp object to store dispersion data
    dtype       - data type (phase/group)
    ::: output :::
    dispersion curve is stored in indisp
    ==========================================================================
    """
    if not isinstance(indisp, disp):
        raise ValueError('indisp should be type of disp!')
    dtype   = dtype.lower()
    if dtype == 'ph' or dtype == 'phase':
        if indisp.isphase:
            print 'phase velocity data is already stored!'
            return False
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
            return False
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

def writedisptxt(outfname, outdisp, dtype='ph'):
    """
    Write dispersion curve to a txt file
    ==========================================================================
    ::: input :::
    outfname    - output file name
    outdisp     - disp object storing dispersion data
    dtype       - data type (phase/group)
    ::: output :::
    a txt file contains predicted and observed dispersion data
    ==========================================================================
    """
    if not isinstance(outdisp, disp):
        raise ValueError('outdisp should be type of disp!')
    if dtype == 'ph' or dtype == 'phase':
        if not outdisp.isphase:
            print 'phase velocity data is not stored!'
            return False
        outArr  = np.append(outdisp.pper, outdisp.pvelp)
        outArr  = np.append(outArr, outdisp.pvelo)
        outArr  = np.append(outArr, outdisp.stdpvelo)
        outArr  = outArr.reshape((4, outdisp.npper))
        outArr  = outArr.T
        np.savetxt(outfname, outArr, fmt='%g')
    
    elif dtype == 'gr' or dtype == 'group':
        if not outdisp.isgroup:
            print 'group velocity data is not stored!'
            return False
        outArr  = np.append(outdisp.gper, outdisp.gvelp)
        outArr  = np.append(outArr, outdisp.gvelo)
        outArr  = np.append(outArr, outdisp.stdgvelo)
        outArr  = outArr.reshape((4, outdisp.ngper))
        outArr  = outArr.T
        np.savetxt(outfname, outArr, fmt='%g')
    else:
        raise ValueError('Unexpected dtype: '+dtype)
    return True

def readaziamptxt(infname, indisp, dtype='ph'):
    """
    Read input txt file of azimuthal amplitude
    ==========================================================================
    ::: input :::
    infname     - input file name
    indisp      - disp object to store azimuthal amplitude data
    dtype       - data type (phase/group)
    ::: output :::
    azimuthal amplitude is stored in indisp
    ==========================================================================
    """
    if not isinstance(indisp, disp):
        raise ValueError('indisp should be type of disp!')
    dtype   = dtype.lower()
    if dtype == 'ph' or dtype == 'phase':
        if not indisp.isphase:
            print 'phase velocity data is not stored!'
            return False
        inArr 		= np.loadtxt(infname, dtype=np.float32)
        if not np.allclose(indisp.pper , inArr[:,0]):
            print 'inconsistent period array !'
            return False
        indisp.pampo= inArr[:,1]
        indisp.npper= indisp.pper.size
        try:
            indisp.stdpampo= inArr[:,2]
        except IndexError:
            indisp.stdpampo= np.ones(indisp.npper, dtype=np.float32)
    # # # elif dtype == 'gr' or dtype == 'group':
    # # #     if indisp.isgroup:
    # # #         print 'group velocity data is already stored!'
    # # #         return False
    # # #     inArr 		= np.loadtxt(infname, dtype=np.float32)
    # # #     indisp.gper = inArr[:,0]
    # # #     indisp.gvelo= inArr[:,1]
    # # #     indisp.ngper= indisp.gper.size
    # # #     try:
    # # #         indisp.stdgvelo= inArr[:,2]
    # # #     except IndexError:
    # # #         indisp.stdgvelo= np.ones(indisp.ngper, dtype=np.float32)
    # # #     indisp.isgroup  = True
    else:
        raise ValueError('Unexpected dtype: '+dtype)
    return True

def writeaziamptxt(outfname, outdisp, dtype='ph'):
    """
    Write azimuthal amplitude to a txt file
    ==========================================================================
    ::: input :::
    outfname    - output file name
    outdisp     - disp object storing azimuthal amplitude data
    dtype       - data type (phase/group)
    ::: output :::
    a txt file contains predicted and observed dispersion data
    ==========================================================================
    """
    if not isinstance(outdisp, disp):
        raise ValueError('outdisp should be type of disp!')
    if dtype == 'ph' or dtype == 'phase':
        if not outdisp.isphase:
            print 'phase velocity data is not stored!'
            return False
        outArr  = np.append(outdisp.pper, outdisp.pampp)
        outArr  = np.append(outArr, outdisp.pampo)
        outArr  = np.append(outArr, outdisp.stdpampo)
        outArr  = outArr.reshape((4, outdisp.npper))
        outArr  = outArr.T
        np.savetxt(outfname, outArr, fmt='%g')
    
    # # # elif dtype == 'gr' or dtype == 'group':
    # # #     if not outdisp.isgroup:
    # # #         print 'group velocity data is not stored!'
    # # #         return False
    # # #     outArr  = np.append(outdisp.gper, outdisp.gvelp)
    # # #     outArr  = np.append(outArr, outdisp.gvelo)
    # # #     outArr  = np.append(outArr, outdisp.stdgvelo)
    # # #     outArr  = outArr.reshape((4, outdisp.ngper))
    # # #     outArr  = outArr.T
    # # #     np.savetxt(outfname, outArr, fmt='%g')
    else:
        raise ValueError('Unexpected dtype: '+dtype)
    return True

def readaziphitxt(infname, indisp, dtype='ph'):
    """
    Read input txt file of fast direction azimuth
    ==========================================================================
    ::: input :::
    infname     - input file name
    indisp      - disp object to store fast direction azimuth data
    dtype       - data type (phase/group)
    ::: output :::
    fast direction azimuth is stored in indisp
    ==========================================================================
    """
    if not isinstance(indisp, disp):
        raise ValueError('indisp should be type of disp!')
    dtype   = dtype.lower()
    if dtype == 'ph' or dtype == 'phase':
        if not indisp.isphase:
            print 'phase velocity data is not stored!'
            return False
        inArr 		= np.loadtxt(infname, dtype=np.float32)
        if not np.allclose(indisp.pper , inArr[:,0]):
            print 'inconsistent period array !'
            return False
        indisp.pphio= inArr[:,1]
        indisp.npper= indisp.pper.size
        try:
            indisp.stdpphio= inArr[:,2]
        except IndexError:
            indisp.stdpphio= np.ones(indisp.npper, dtype=np.float32)
    # # # elif dtype == 'gr' or dtype == 'group':
    # # #     if indisp.isgroup:
    # # #         print 'group velocity data is already stored!'
    # # #         return False
    # # #     inArr 		= np.loadtxt(infname, dtype=np.float32)
    # # #     indisp.gper = inArr[:,0]
    # # #     indisp.gvelo= inArr[:,1]
    # # #     indisp.ngper= indisp.gper.size
    # # #     try:
    # # #         indisp.stdgvelo= inArr[:,2]
    # # #     except IndexError:
    # # #         indisp.stdgvelo= np.ones(indisp.ngper, dtype=np.float32)
    # # #     indisp.isgroup  = True
    else:
        raise ValueError('Unexpected dtype: '+dtype)
    return True

def writeaziphitxt(outfname, outdisp, dtype='ph'):
    """
    Write fast direction azimuth to a txt file
    ==========================================================================
    ::: input :::
    outfname    - output file name
    outdisp     - disp object storing fast direction azimuth data
    dtype       - data type (phase/group)
    ::: output :::
    a txt file contains predicted and observed dispersion data
    ==========================================================================
    """
    if not isinstance(outdisp, disp):
        raise ValueError('outdisp should be type of disp!')
    if dtype == 'ph' or dtype == 'phase':
        if not outdisp.isphase:
            print 'phase velocity data is not stored!'
            return False
        outArr  = np.append(outdisp.pper, outdisp.pphip)
        outArr  = np.append(outArr, outdisp.pphio)
        outArr  = np.append(outArr, outdisp.stdpphio)
        outArr  = outArr.reshape((4, outdisp.npper))
        outArr  = outArr.T
        np.savetxt(outfname, outArr, fmt='%g')
    
    # # # elif dtype == 'gr' or dtype == 'group':
    # # #     if not outdisp.isgroup:
    # # #         print 'group velocity data is not stored!'
    # # #         return False
    # # #     outArr  = np.append(outdisp.gper, outdisp.gvelp)
    # # #     outArr  = np.append(outArr, outdisp.gvelo)
    # # #     outArr  = np.append(outArr, outdisp.stdgvelo)
    # # #     outArr  = outArr.reshape((4, outdisp.ngper))
    # # #     outArr  = outArr.T
    # # #     np.savetxt(outfname, outArr, fmt='%g')
    else:
        raise ValueError('Unexpected dtype: '+dtype)
    return True
    

def readrftxt(infname, inrf):
    """
    Read input txt file of receiver function
    ==========================================================================
    ::: input :::
    infname     - input file name
    inrf        - disp object to store dispersion data
    ::: output :::
    receiver function data is stored in inrf
    ==========================================================================
    """
    if not isinstance(inrf, rf):
        raise ValueError('inrf should be type of rf!')
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

def writerftxt(outfname, outrf, tf=10.):
    """
    Write receiver function data to a txt file
    ==========================================================================
    ::: input :::
    outfname    - output file name
    outdisp     - rf object storing dispersion data
    tf          - end time point for trim
    ::: output :::
    a txt file contains predicted and observed receiver function data
    ==========================================================================
    """
    if not isinstance(outrf, rf):
        raise ValueError('outrf should be type of rf!')
    if outrf.npts == 0:
        print 'receiver function data is not stored!'
        return False
    nout    = int(outrf.fs*tf)+1
    nout    = min(nout, outrf.npts)
    outArr  = np.append(outrf.tp[:nout], outrf.rfp[:nout])
    outArr  = np.append(outArr, outrf.to[:nout])
    outArr  = np.append(outArr, outrf.rfo[:nout])
    outArr  = np.append(outArr, outrf.stdrfo[:nout])
    outArr  = outArr.reshape((5, nout))    
    outArr  = outArr.T
    np.savetxt(outfname, outArr, fmt='%g')
    return True

####################################################
# Predefine the parameters for the disp object
####################################################
spec_disp = [
        #########################
        # phase velocities
        #########################
        ('npper',   numba.int32),
        ('pper',    numba.float32[:]),
        # observed 
        ('pvelo',   numba.float32[:]),
        ('stdpvelo',numba.float32[:]),
        ('pphio',   numba.float32[:]),
        ('stdpphio',numba.float32[:]),
        ('pampo',   numba.float32[:]),
        ('stdpampo',numba.float32[:]),
        # reference
        ('pvelref', numba.float32[:]),
        # predicted
        ('pvelp',   numba.float32[:]),
        ('pphip',   numba.float32[:]),
        ('pampp',   numba.float32[:]),
        # 
        ('isphase', numba.boolean),
        ('pmisfit', numba.float32),
        ('pL',      numba.float32),
        #########################
        # group velocities
        #########################
        ('ngper',   numba.int32),
        ('gper',    numba.float32[:]),
        # observed
        ('gvelo',   numba.float32[:]),
        ('stdgvelo',numba.float32[:]),
        # ('gphio', numba.float32[:]),
        # ('gampo', numba.float32[:]),
        # predicted
        ('gvelp',   numba.float32[:]),
        # ('gphip', numba.float32[:]),
        # ('gampp', numba.float32[:]),
        ('isgroup', numba.boolean),
        ('gmisfit', numba.float32),
        ('gL',      numba.float32),
        # total misfit/likelihood
        ('misfit',  numba.float32),
        ('L',       numba.float32),
        # common period for phase/group
        ('period',  numba.float32[:]),
        ('nper',    numba.int32)
        ]

@numba.jitclass(spec_disp)
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
    pphip   - predicted phase velocity fast direction angle
    pampp   - predicted phase velocity azimuthal anisotropic amplitude
    :   others  :
    isphase - phase dispersion data is stored or not
    pmisfit - phase dispersion misfit
    pL      - phase dispersion likelihood 
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
    
    def get_pmisfit(self):
        """
        Compute the misfit for phase velocities
        """
        if not self.isphase:
            print 'No phase velocity data stored'
            return False
        temp    = 0.
        for i in xrange(self.npper):
            temp+= (self.pvelo[i] - self.pvelp[i])**2/self.stdpvelo[i]**2
        misfit  = np.sqrt(temp/self.npper)
        if temp > 50.:
            temp= np.sqrt(temp*50.)
        L       = np.exp(-0.5 * temp)
        self.pmisfit    = misfit
        self.pL         = L
        return True
    
    def get_gmisfit(self):
        """
        Compute the misfit for group velocities
        """
        if not self.isgroup:
            print 'No group velocity data stored'
            return False
        temp    = 0.
        for i in xrange(self.npper):
            temp+= (self.gvelo[i] - self.gvelp[i])**2/self.stdgvelo[i]**2
        misfit  = np.sqrt(temp/self.ngper)
        if temp > 50.:
            temp= np.sqrt(temp*50.)
        L       = np.exp(-0.5 * temp)
        self.gmisfit    = misfit
        self.gL         = L
        return True
    
    def get_misfit(self):
        """
        Compute combined misfit
        """
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
        if (not self.isphase) and (not self.isgroup):
            print 'No dispersion data stored!'
            self.misfit = 0.
            self.L      = 1.
            return False
        # misfit for both
        temp    = temp1 + temp2
        self.misfit     = np.sqrt(temp/(self.npper+self.ngper))
        if temp > 50.:
            temp = np.sqrt(temp*50.)
        if temp > 50.:
            temp = np.sqrt(temp*50.)
        self.L          = np.exp(-0.5 * temp)
        return True
    
    # def get_misfit_tti(self):
        
    
####################################################
# Predefine the parameters for the rf object
####################################################
spec_rf = [
        # sampling frequency/npts
        ('fs',      numba.float32),
        ('npts',    numba.int32),
        # observed receiver function
        ('rfo',     numba.float32[:]),
        ('to',      numba.float32[:]),
        ('stdrfo',  numba.float32[:]),
        # predicted receiver function
        ('rfp',     numba.float32[:]),
        ('tp',      numba.float32[:]),
        # misfit/likelihood
        ('misfit',  numba.float32),
        ('L',       numba.float32)
        ]

@numba.jitclass(spec_rf)
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
    
    def get_misfit_incompatible(self, rffactor):
        """
        compute misfit when the time array of predicted and observed data is incompatible, quite slow!
        ==============================================================================
        ::: input :::
        rffactor    - factor for downweighting the misfit for likelihood computation
        ==============================================================================
        """
        if self.npts == 0:
            self.misfit = 0.
            self.L      = 1.
            return False
        temp    = 0.
        k       = 0
        for i in xrange(self.npts):
            for j in xrange(self.tp.size):
                if self.to[i] == self.tp[j] and self.to[i] <= 10 and self.to[i] >= 0 :
                    temp    += ( (self.rfo[i] - self.rfp[j])**2 / (self.stdrfo[i]**2) )
                    k       += 1
                    break
        self.misfit = np.sqrt(temp/k)
        tS          = temp/rffactor
        if tS > 50.:
            tS      = np.sqrt(tS*50.)
        self.L      = np.exp(-0.5 * tS)
        return True
    
    def get_misfit(self, rffactor):
        """
        Compute misfit for receiver function
        ==============================================================================
        ::: input :::
        rffactor    - factor for downweighting the misfit for likelihood computation
        ==============================================================================
        """
        temp    = 0.
        k       = 0
        if self.npts == 0:
            self.misfit = 0.
            self.L      = 1.
            return False
        for i in xrange(self.npts):
            if self.to[i] != self.tp[i]:
                print ('Incompatible time arrays!')
                return self.get_misfit_incompatible(rffactor)
            if self.to[i] >= 0:
                temp    += ( (self.rfo[i] - self.rfp[i])**2 / (self.stdrfo[i]**2) )
                k       += 1
            if self.to[i] > 10:
                break
        self.misfit = np.sqrt(temp/k)
        tS          = temp/rffactor
        if tS > 50.:
            tS      = np.sqrt(tS*50.)
        self.L      = np.exp(-0.5 * tS)
        return True
    
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
    
    def get_misfit(self, wdisp, rffactor):
        """
        Compute combined misfit
        ==========================================================================================
        ::: input :::
        wdisp       - relative weigh for dispersion data ( 0.~1. )
        rffactor    - factor for downweighting the misfit for likelihood computation of rf
        ==========================================================================================
        """
        self.dispR.get_misfit()
        self.rfr.get_misfit(rffactor)
        # compute combined misfit and likelihood
        self.misfit = wdisp*self.dispR.misfit + (1.-wdisp)*self.rfr.misfit
        self.L      = ((self.dispR.L)**wdisp)*((self.rfr.L)**(1.-wdisp))
        return
