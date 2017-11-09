# -*- coding: utf-8 -*-
"""
Module for handling parameterization of the model

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
import random

####################################################
# I/O functions
####################################################
def readmodtxt(infname, inmod):
    """
    Read model parameterization from a txt file
    column 1: id
    column 2: flag  - layer(1)/B-splines(2/3)/gradient layer(4)/water(5)
    column 3: thickness
    column 4: number of control points for the group
    column 5 - (4+tnp): value
    column 4+tnp - 4+2*tnp: ratio
    column -1: vpvs ratio
    ==========================================================================
    ::: input :::
    infname - input file name
    inmod   - isomod object to store model parameterization
    ==========================================================================
    """
    if not isinstance(inmod, isomod):
        raise ValueError('inmod should be type of isomod!')
    nmod   = 0
    for l1 in open(infname,"r"):
        nmod    += 1
    print "Number of model parameter groups: %d " % nmod
    inmod.init_arr(nmod)
    
    for l1 in open(infname,"r"):
        l1 			        = l1.rstrip()
        l2 			        = l1.split()
        iid 		        = int(l2[0])
        flag		        = int(l2[1])
        thickness	        = float(l2[2])
        tnp 		        = int(l2[3]) # number of parameters
        inmod.mtype[iid]	= flag
        inmod.thickness[iid]= thickness
        inmod.numbp[iid] 	= tnp 
        if (int(l2[1]) == 5):  # water layer			
            if (tnp != 1):
                print " Water layer! Only one value for Vp"
                return False
        if (int(l2[1]) == 4):
            if (tnp != 2):
                print "Error: only two values needed for gradient type, and one value for vpvs"
                print tnp
                return False
        if ( (int(l2[1])==1 and len(l2) != 4+2*tnp + 1) or (int(l2[1]) == 2 and len(l2) != 4+tnp + 1) ): # tnp parameters (+ tnp ratio for layered model) + 1 vpvs parameter
            print "wrong input !!!"
            return False
        cvel        = []
        ratio       = []
        nr          = 0
        for i in xrange(tnp):
            cvel.append(float(l2[4+i]))
            if (int(l2[1]) ==1):  # type 1 layer
                nr  += 1
                ratio.append(float(l2[4+tnp+i]))
        inmod.vpvs[iid]         = (float(l2[-1]))-0.
        cvel                    = np.array(cvel, dtype=np.float32)
        ratio                   = np.array(ratio, dtype=np.float32)
        inmod.cvel[:tnp, iid]   = cvel
        inmod.ratio[:nr, iid]   = ratio
    return True

def readtimodtxt(infname, inmod):
    """
    Read model parameterization from a txt file
    column 1: id
    column 2: flag  - layer(1)/B-splines(2/3)/gradient layer(4)/water(5)
    column 3: thickness
    column 4: number of control points for the group
    column 5 - (4+tnp): value
    column 4+tnp - 4+2*tnp: ratio
    column -1: vpvs ratio
    ==========================================================================
    ::: input :::
    infname - input file name
    inmod   - isomod object to store model parameterization
    ==========================================================================
    """
    if not isinstance(inmod, ttimod):
        raise ValueError('inmod should be type of isomod!')
    nmod   = 0
    for l1 in open(infname,"r"):
        nmod    += 1
    print "Number of model parameter groups: %d " % nmod
    inmod.init_arr(nmod)
    
    for l1 in open(infname,"r"):
        l1 			        = l1.rstrip()
        l2 			        = l1.split()
        iid 		        = int(l2[0])
        flag		        = int(l2[1])
        nmodp               = int(l2[2])
        flag2               = int(l2[3]) # computation type flag, deprecated!!!
        thickness	        = float(l2[4])
        tnp 		        = int(l2[5]) # number of parameters
        inmod.mtype[iid]	= flag
        inmod.thickness[iid]= thickness
        inmod.numbp[iid] 	= tnp
        
        if (flag == 5):  # water layer			
            if (tnp != 1):
                print " Water layer! Only one value for Vp"
                return False
        if (flag == 4):
            if (tnp != 2):
                print "Error: only two values needed for gradient type, and one value for vpvs"
                print tnp
                return False
        if ( (flag ==1 and len(l2) != 6+2*tnp*nmodp + 1) or (flag  == 2 and len(l2) != 6+tnp*nmodp + 1) ): # tnp parameters (+ tnp ratio for layered model) + 1 vpvs parameter
            print "wrong input !!!"
            return False
        if nmodp !=5 and nmodp !=7  and nmodp !=1:
            print "wrong input, nmodp=", nmodp
            return False
        
        cvph        = []
        cvpv        = []
        cvsh        = []
        cvsv        = []
        ceta        = []
        if nmodp == 7:
            cdip    = []
            cstrike = []
        ratio       = []
        nr          = 0
        for i in xrange(tnp):
            cvsv.append(float(l2[6+i*nmodp]))
            if nmodp > 1:
                cvsh.append(float(l2[6+i*nmodp+1]))
                cvpv.append(float(l2[6+i*nmodp+2]))
                cvph.append(float(l2[6+i*nmodp+3]))
                ceta.append(float(l2[6+i*nmodp+4]))
            if nmodp == 7:
                cdip.append(float(l2[6+i*nmodp+5]))
                cstrike.append(float(l2[6+i*nmodp+6]))
            if (flag ==1):  # type 1 layer
                nr  += 1
                ratio.append(float(l2[6+tnp*nmodp+i]))
        inmod.vpvs[iid]         = (float(l2[-1]))-0.
        
        cvsv                    = np.array(cvsv, dtype=np.float32)
        if nmodp == 1:
            cvph                    = cvsv * inmod.vpvs[iid]
            cvpv                    = cvsv * inmod.vpvs[iid]
            cvsh                    = cvsv
            ceta                    = np.ones(cvsv.size, dtype=np.float32)
        else:
            cvph                    = np.array(cvph, dtype=np.float32) 
            cvpv                    = np.array(cvpv, dtype=np.float32) 
            cvsh                    = np.array(cvsh, dtype=np.float32)
            ceta                    = np.array(ceta, dtype=np.float32)
        if nmodp == 7:
            cdip                    = np.array(cdip, dtype=np.float32)
            cstrike                 = np.array(cstrike, dtype=np.float32)
        ratio                   = np.array(ratio, dtype=np.float32)
        inmod.cvph[:tnp, iid]   = cvph
        inmod.cvpv[:tnp, iid]   = cvpv
        inmod.cvsh[:tnp, iid]   = cvsh
        inmod.cvsv[:tnp, iid]   = cvsv
        inmod.ceta[:tnp, iid]   = ceta
        if nmodp == 7:
            inmod.cdip[:tnp, iid]   = cdip
            inmod.cstrike[:tnp, iid]= cstrike
        inmod.ratio[:nr, iid]   = ratio
    return True

def readparatxt(infname, inpara):
    """
    read txt perturbation parameter file
    ==========================================================================
    ::: input :::
    infname - input file name
    inpara  - para object
    ==========================================================================
    """
    npara   = 0
    for l1 in open(infname,"r"):
        npara   += 1
    print "Number of parameters for perturbation: %d " % npara
    inpara.init_arr(npara)
    i   = 0
    with open(infname, 'r') as fid:
        for line in fid.readlines():
            temp                        = np.array(line.split(), dtype=np.float32)
            ne                          = temp.size
            # # # inpara.numbind[i]           = ne
            inpara.paraindex[:ne, i]    = temp
            i                           += 1
    # print "read para over!"
    return
    
####################################################
# auxiliary functions
####################################################

@numba.jit(numba.types.Tuple((numba.float32[:], numba.float32[:,:]))(\
        numba.int32, numba.int32, numba.float32, numba.float32, numba.int32, numba.int32))
def bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts):
    #-------------------------------- 
    # defining the knot vector
    #--------------------------------
    m           = nBs-1+degBs
    t           = np.zeros(m+1, dtype=np.float32)
    for i in xrange(degBs):
        t[i] = zmin_Bs + i*(zmax_Bs-zmin_Bs)/10000.
    for i in range(degBs,m+1-degBs):
        n_temp  = m+1-degBs-degBs+1
        if (disfacBs !=1):
            temp= (zmax_Bs-zmin_Bs)*(disfacBs-1)/(math.pow(disfacBs,n_temp)-1)
        else:
            temp= (zmax_Bs-zmin_Bs)/n_temp
        t[i] = temp*math.pow(disfacBs,(i-degBs)) + zmin_Bs
    for i in range(m+1-degBs,m+1):
        t[i] = zmax_Bs-(zmax_Bs-zmin_Bs)/10000.*(m-i)
    # depth array
    step    = (zmax_Bs-zmin_Bs)/(npts-1)
    depth   = np.zeros(npts, dtype=np.float32)
    for i in xrange(npts):
        depth[i]    = np.float32(i) * np.float32(step) + np.float32(zmin_Bs)
    # arrays for storing B spline basis
    obasis  = np.zeros((np.int64(m), np.int64(npts)), dtype = np.float32)
    nbasis  = np.zeros((np.int64(m), np.int64(npts)), dtype = np.float32)
    #-------------------------------- 
    # computing B spline basis
    #--------------------------------
    for i in xrange (m):
        for j in xrange (npts):
            if (depth[j] >=t[i] and depth[j]<t[i+1]):
                obasis[i][j] = 1
            else:
                obasis[i][j] = 0
    for pp in range (1,degBs):
        for i in xrange (m-pp):
            for j in xrange (npts):
                nbasis[i][j] = (depth[j]-t[i])/(t[i+pp]-t[i])*obasis[i][j] + \
                        (t[i+pp+1]-depth[j])/(t[i+pp+1]-t[i+1])*obasis[i+1][j]
        for i in xrange (m-pp):
            for j in xrange (npts):
                obasis[i][j] = nbasis[i][j]
    nbasis[0][0]            = 1
    nbasis[nBs-1][npts-1]   = 1
    return t, nbasis

####################################################
# Predefine the parameters for the para1d object
####################################################
spec_para1d = [
        ('npara',       numba.int32),
        ('maxind',      numba.int32),
        ('paraindex',   numba.float32[:,:]),
        # # # ('numbind',     numba.int32[:]),
        ('paraval',     numba.float32[:]),
        ('isspace',     numba.boolean),
        ('space',       numba.float32[:,:]),
        # total misfit/likelihood
        ('misfit',      numba.float32),
        ('L',           numba.float32)
        ]
@numba.jitclass(spec_para1d)
class para1d(object):
    """
    An object for handling parameter perturbations
    =====================================================================================================================
    ::: parameters :::
    :   values  :
    npara       - number of parameters for perturbations
    misfit      - misfit
    L           - likelihood
    maxind      - maximum number of index for each parameters
    isspace     - if space array is computed or not
    :   arrays  :
    numbind     - number of index for each parameters (currently deprecated, may be reused later...)
    paraval     - parameter array for perturbation
    paraindex   - index array indicating numerical setup for each parameter
                1.  isomod
                    paraindex[0, :] - type of parameters
                                        0   - velocity coefficient for splines
                                        1   - thickness
                                        -1  - vp/vs ratio
                                        others to be added for tilted TI model
                    paraindex[1, :] - index for type of amplitude for parameter perturbation
                                        1   - absolute
                                        else- relative
                    paraindex[2, :] - amplitude for parameter perturbation (absolute/relative)
                    paraindex[3, :] - step for parameter space 
                    paraindex[4, :] - index for the parameter in the model group   
                    paraindex[5, :] - index for spline basis/grid point, ONLY works when paraindex[0, :] == 0
                2.  ttimod
                    paraindex[0, :] - type of parameters
                                        0   - vph coefficient for splines
                                        1   - vpv coefficient for splines
                                        2   - vsh coefficient for splines
                                        3   - vsv coefficient for splines
                                        4   - eta coefficient for splines
                                        5   - dip
                                        6   - strike
                                        
                                        7   - rho coefficient for splines
                                        8   - thickness
                                        -1  - vp/vs ratio
                                        
                    paraindex[1, :] - index for type of amplitude for parameter perturbation
                                        1   - absolute
                                        else- relative
                    paraindex[2, :] - amplitude for parameter perturbation (absolute/relative)
                    paraindex[3, :] - step for parameter space 
                    paraindex[4, :] - index for the parameter in the model group   
                    paraindex[5, :] - index for spline basis/grid point, ONLY works when paraindex[0, :] == 0
    space       - space array for defining perturbation range
                    space[0, :]     - min value
                    space[1, :]     - max value
                    space[2, :]     - step, used as sigma in Gaussian random generator
    =====================================================================================================================
    """
    def __init__(self):
        self.npara          = 0
        self.misfit         = -1.
        self.L              = 1.
        self.maxind         = 6
        self.isspace        = False
        return
    
    def init_arr(self, npara):
        self.npara          = npara
        # # # self.numbind        = np.zeros(np.int64(self.npara), dtype=np.int32)
        self.paraval        = np.zeros(np.int64(self.npara), dtype=np.float32)
        self.paraindex      = np.zeros((np.int64(self.maxind), np.int64(self.npara)), dtype = np.float32)
        self.space          = np.zeros((3, np.int64(self.npara)), dtype = np.float32)
        return
    
    def space2true(self): self.isspace = True
    
    def new_paraval(self, ptype):
        """
        peturb parameters in paraval array
        ===============================================================================
        ::: input :::
        ptype   - perturbation type
                    0   - uniform random value generated from parameter space
                    1   - Gauss random number generator given mu = oldval, sigma=step
        ===============================================================================
        """
        paralst = []
        # if ptype == 0:
        #     for i in xrange(self.npara):
        #         newval  = np.random.uniform(self.space[0, i], self.space[1, i])
        #         paralst.append(newval)
        # else:
        #     for i in xrange(self.npara):
        #         oldval 	= self.paraval[i]
        #         step 	= self.space[2, i]
        #         run 	= True
        #         j		= 0
        #         while (run and j<10000): 
        #             newval  = random.gauss(oldval, step)
        #             if (newval >= self.space[0, i] and newval <= self.space[1, i]):
        #                 run = False
        #             j   +=1
        #         paralst.append(newval)
        
        for i in xrange(self.npara):
            if ptype  == 1 and self.space[2, i] > 0.:
                oldval 	= self.paraval[i]
                step 	= self.space[2, i]
                run 	= True
                j		= 0
                while (run and j<10000): 
                    newval  = random.gauss(oldval, step)
                    if (newval >= self.space[0, i] and newval <= self.space[1, i]):
                        run = False
                    j   +=1
            else: 
                newval  = np.random.uniform(self.space[0, i], self.space[1, i])
            paralst.append(newval)
        self.paraval    = np.array(paralst, dtype=np.float32)
        return
        
    def copy(self):
        """
        return a copy of the object
        """
        outpara             = para1d()
        outpara.init_arr(self.npara)
        outpara.paraindex   = self.paraindex.copy()
        # # # outpara.numbind     = self.numbind.copy()
        outpara.paraval     = self.paraval.copy()
        outpara.isspace     = self.isspace
        outpara.space       = self.space.copy()
        outpara.misfit      = self.misfit
        outpara.L           = self.L
        return outpara

# define type of disp object
para1d_type   = numba.deferred_type()
para1d_type.define(para1d.class_type.instance_type)

####################################################
# Predefine the parameters for the isomod object
####################################################
spec_isomod = [
        ('nmod',        numba.int32),
        ('maxlay',      numba.int32),
        ('maxspl',      numba.int32),
        # number of control points
        ('numbp',       numba.int32[:]),
        ('mtype',       numba.int32[:]),
        ('thickness',   numba.float32[:]),
        ('nlay',        numba.int32[:]),
        ('vpvs',        numba.float32[:]),
        ('isspl',       numba.int32[:]),
        # arrays
        ('spl',         numba.float32[:, :, :]),
        ('ratio',       numba.float32[:, :]),
        ('cvel',        numba.float32[:, :]),
        ('vs',          numba.float32[:, :]),
        ('hArr',        numba.float32[:, :]),
        ('t',           numba.float32[:, :]),
        # para1d object
        ('para',        para1d_type)
        ]

@numba.jitclass(spec_isomod)
class isomod(object):
    """
    An object for handling parameterization of 1d isotropic model for the inversion
    =====================================================================================================================
    ::: parameters :::
    :   numbers     :
    nmod        - number of model groups
    maxlay      - maximum layers for each group (default - 100)
    maxspl      - maximum spline coefficients for each group (default - 20)
    :   1D arrays   :
    numbp       - number of control points/basis (1D int array with length nmod)
    mtype       - model parameterization types (1D int array with length nmod)
                    1   - layer         - nlay  = numbp, hArr = ratio*thickness, vs = cvel
                    2   - B-splines     - hArr  = thickness/nlay, vs    = (cvel*spl)_sum over numbp
                    4   - gradient layer- nlay is defined depends on thickness
                                            hArr  = thickness/nlay, vs  = from cvel[0, i] to cvel[1, i]
                    5   - water         - nlay  = 1, vs = 0., hArr = thickness
    thickness   - thickness of each group (1D float array with length nmod)
    nlay        - number of layres in each group (1D int array with length nmod)
    vpvs        - vp/vs ratio in each group (1D float array with length nmod)
    isspl       - flag array indicating the existence of basis B spline (1D int array with length nmod)
                    0 - spline basis has NOT been computed
                    1 - spline basis has been computed
    :   multi-dim arrays    :
    t           - knot vectors for B splines (2D array - [:(self.numb[i]+degBs), i]; i indicating group id)
    spl         - B spline basis array (3D array - [:self.numb[i], :self.nlay[i], i]; i indicating group id)
                    ONLY used for mtype == 2
    ratio       - array for the ratio of each layer (2D array - [:self.nlay[i], i]; i indicating group id)
                    ONLY used for mtype == 1
    cvel        - velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
                    layer mod   - input velocities for each layer
                    spline mod  - coefficients for B spline
                    gradient mod- top/bottom layer velocity
    :   model arrays        :
    vs          - vs velocity arrays (2D array - [:self.nlay[i], i]; i indicating group id)
    hArr        - layer arrays (2D array - [:self.nlay[i], i]; i indicating group id)
    :   para1d  :
    para        - object storing parameters for perturbation
    =====================================================================================================================
    """
    
    def __init__(self):

        self.nmod       = 0
        self.maxlay     = 100
        self.maxspl     = 20
        self.para       = para1d()
        return
    
    def init_arr(self, nmod):
        """
        initialization of arrays
        """
        self.nmod       = nmod
        # arrays of size nmod
        self.numbp      = np.zeros(np.int64(self.nmod), dtype=np.int32)
        self.mtype      = np.zeros(np.int64(self.nmod), dtype=np.int32)
        self.thickness  = np.zeros(np.int64(self.nmod), dtype=np.float32)
        self.nlay       = np.ones(np.int64(self.nmod), dtype=np.int32)*np.int32(20) 
        self.vpvs       = np.ones(np.int64(self.nmod), dtype=np.float32)*np.float32(1.75)
        self.isspl      = np.zeros(np.int64(self.nmod), dtype=np.int32)
        # arrays of size maxspl, nmod
        self.cvel       = np.zeros((np.int64(self.maxspl), np.int64(self.nmod)), dtype = np.float32)
        self.t          = np.zeros((np.int64(self.maxspl), np.int64(self.nmod)), dtype = np.float32)
        # arrays of size maxlay, nmod
        self.ratio      = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.vs         = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.hArr       = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        # arrays of size maxspl, maxlay, nmod
        self.spl        = np.zeros((np.int64(self.maxspl), np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        return
    
    def bspline(self, i):
        """
        Compute B-spline basis
        The actual degree is k = degBs - 1
        e.g. nBs = 5, degBs = 4, k = 3, cubic B spline
        ::: output :::
        self.spl    - (nBs+k, npts)
                        [:nBs, :] B spline basis for nBs control points
                        [nBs:,:] can be ignored
        """
        if self.thickness[i] >= 150:
            self.nlay[i]    = 60
        elif self.thickness[i] < 10:
            self.nlay[i]    = 5
        elif self.thickness[i] < 20:
            self.nlay[i]    = 10
        else:
            self.nlay[i]    = 30
            
        if self.isspl[i] == 1:
            print("spline basis already exists!")
            return
        if self.mtype[i] != 2:
            print('Not spline parameterization!')
            return 
        # initialize
        if i >= self.nmod:
            raise ValueError('index for spline group out of range!')
            return
        nBs         = self.numbp[i]
        if nBs < 4:
            degBs   = 3
        else:
            degBs   = 4
        zmin_Bs     = 0.
        zmax_Bs     = self.thickness[i]
        disfacBs    = 2.
        npts        = self.nlay[i]
        t, nbasis   = bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts)
        m           = nBs-1+degBs
        if m > self.maxspl:
            raise ValueError('number of splines is too large, change default maxspl!')
        # # # self.spl[:m, :npts, i]  = nbasis[:m, :]
        self.spl[:nBs, :npts, i]= nbasis[:nBs, :]
        self.isspl[i]           = 1
        self.t[:m+1,i]          = t
        return 

    def update(self):
        """
        Update model (vs and hArr arrays)
        """
        for i in xrange(self.nmod):
            if self.nlay[i] > self.maxlay:
                raise ValueError('number of layers is too large, need change default maxlay!')
            # layered model
            if self.mtype[i] == 1:
                self.nlay[i]    = self.numbp[i]
                self.hArr[:, i] = self.ratio[:, i] * self.thickness[i]
                self.vs[:, i]   = self.cvel[:, i]
            # B spline model
            elif self.mtype[i] == 2:
                self.isspl[i]   = 0
                self.bspline(i)
                # # if self.isspl[i] != 1:
                # #     self.bspline(i)
                for ilay in xrange(self.nlay[i]):
                    tvalue 	= 0.
                    for ibs in xrange(self.numbp[i]):
                        tvalue = tvalue + self.spl[ibs, ilay, i] * self.cvel[ibs, i]
                    self.vs[ilay, i]    = tvalue
                    self.hArr[ilay, i]  = self.thickness[i]/self.nlay[i]
            # gradient layer
            elif self.mtype[i] == 4:
                nlay 	    = 4
                if self.thickness[i] >= 20.:
                    nlay    = 20
                if self.thickness[i] > 10. and self.thickness[i] < 20.:
                    nlay    = int(self.thickness[i]/1.)
                if self.thickness[i] > 2. and self.thickness[i] <= 10.:
                    nlay    = int(self.thickness[i]/0.5)
                if self.thickness[i] < 0.5:
                    nlay    = 2
                dh 	        = self.thickness[i]/float(nlay)
                dcvel 		= (self.cvel[1, i] - self.cvel[0, i])/(nlay - 1.)
                vs          = np.zeros(np.int64(nlay), dtype=np.float32)
                for ilay in xrange(nlay):
                    vs[ilay]= self.cvel[0, i] + dcvel*np.float32(ilay)
                hArr 	    = np.ones(nlay, dtype=np.float32)*np.float32(dh)
                self.vs[:nlay, i]   = vs
                self.hArr[:nlay, i] = hArr
                self.nlay[i]        = nlay
            # water layer
            elif self.mtype[i] == 5:
                nlay    = 1
                self.vs[0, i]       = 0.
                self.hArr[0, i]     = self.thickness[i]
                self.nlay[i]        = 1
        return
    
    def get_vmodel(self):
        """
        get velocity models
        ==========================================================================
        ::: output :::
        hArr, vs, vp, rho, qs, qp
        ==========================================================================
        """
        vs      = []
        vp      = []
        rho     = []
        qs      = []
        qp      = []
        hArr    = []
        depth   = 0.
        for i in xrange(self.nmod):
            for j in xrange(self.nlay[i]):
                hArr.append(self.hArr[j, i])
                depth += self.hArr[j, i]
                if self.mtype[i] == 5:
                    vs.append(0.)
                    vp.append(self.cvel[0, i])
                    rho.append(1.02)
                    qs.append(10000.)
                    qp.append(57822.)
                elif (i == 0 and self.mtype[i] != 5) or (i == 1 and self.mtype[0] == 5):
                    vs.append(self.vs[j, i])
                    vp.append(self.vs[j, i]*self.vpvs[i])
                    rho.append(0.541 + 0.3601*self.vs[j, i]*self.vpvs[i])
                    qs.append(80.)
                    qp.append(160.)
                else:
                    vs.append(self.vs[j, i])
                    vp.append(self.vs[j, i]*self.vpvs[i])
                    # if depth < 18.:
                    qs.append(600.)
                    qp.append(1400.)
                    if (self.vs[j, i]*self.vpvs[i]) < 7.5:
                        rho.append(0.541 + 0.3601*self.vs[j, i]*self.vpvs[i])
                    else:
                        rho.append(3.35) # Kaban, M. K et al. (2003), Density of the continental roots: Compositional and thermal contributions
        vs      = np.array(vs, dtype=np.float32)
        vp      = np.array(vp, dtype=np.float32)
        rho     = np.array(rho, dtype=np.float32)
        qs      = np.array(qs, dtype=np.float32)
        qp      = np.array(qp, dtype=np.float32)
        hArr    = np.array(hArr, dtype=np.float32)
        return hArr, vs, vp, rho, qs, qp
    
    def get_paraind(self):
        """
        get parameter index arrays for para
        Table 1 and 2 in Shen et al. 2012
        
        references:
        Shen, W., Ritzwoller, M.H., Schulte-Pelkum, V. and Lin, F.C., 2012.
            Joint inversion of surface wave dispersion and receiver functions: a Bayesian Monte-Carlo approach.
                Geophysical Journal International, 192(2), pp.807-836.
        """
        npara   = self.numbp.sum()  + self.nmod - 1
        self.para.init_arr(npara)
        ipara   = 0
        for i in xrange(self.nmod):
            for j in xrange(self.numbp[i]):
                self.para.paraindex[0, ipara]   = 0
                if i == 0:
                    # sediment, cvel space is +- 1 km/s, different from Shen et al. 2012
                    self.para.paraindex[1, ipara]   = 1
                    self.para.paraindex[2, ipara]   = 1.
                else:
                    # +- 20 %
                    self.para.paraindex[1, ipara]   = -1
                    self.para.paraindex[2, ipara]   = 20.
                # 0.05 km/s 
                self.para.paraindex[3, ipara]   = 0.05
                self.para.paraindex[4, ipara]   = i
                self.para.paraindex[5, ipara]   = j
                ipara   +=1
        if self.nmod >= 3:
            # sediment thickness
            self.para.paraindex[0, ipara]   = 1
            self.para.paraindex[1, ipara]   = -1
            self.para.paraindex[2, ipara]   = 100.
            self.para.paraindex[3, ipara]   = 0.1
            self.para.paraindex[4, ipara]   = 0
            ipara   += 1
        # crustal thickness
        self.para.paraindex[0, ipara]   = 1
        self.para.paraindex[1, ipara]   = -1
        self.para.paraindex[2, ipara]   = 20.
        self.para.paraindex[3, ipara]   = 1.
        if self.nmod >= 3:
            self.para.paraindex[4, ipara]   = 1.
        else:
            self.para.paraindex[4, ipara]   = 0.
        return
                
        
    def mod2para(self):
        """
        convert model to parameter arrays for perturbation
        """
        paralst     = [] 
        for i in xrange(self.para.npara):
            ig      = int(self.para.paraindex[4, i])
            # velocity coefficient 
            if int(self.para.paraindex[0, i]) == 0:
                ip  = int(self.para.paraindex[5, i])
                val = self.cvel[ip , ig]
            # total thickness of the group
            elif int(self.para.paraindex[0, i]) == 1:
                val = self.thickness[ig]
            # vp/vs ratio
            elif int(self.para.paraindex[0, i]) == -1:
                val = self.vpvs[ig]
            else:
                raise ValueError('Unexpected value in paraindex!')
            paralst.append(val)
            #-------------------------------------------
            # defining parameter space for perturbation
            #-------------------------------------------
            if not self.para.isspace:
                step    = self.para.paraindex[3, i]
                if int(self.para.paraindex[1, i]) == 1:
                    valmin  = val - self.para.paraindex[2, i]
                    valmax  = val + self.para.paraindex[2, i]
                else:
                    valmin  = val - val*self.para.paraindex[2, i]/100.
                    valmax  = val + val*self.para.paraindex[2, i]/100.
                valmin  = max (0.,valmin)
                valmax  = max (valmin + 0.0001, valmax)
                if (self.para.paraindex[0, i] == 0 and i == 0 and self.para.paraindex[5, i] == 0): # if it is the upper sedi:
                    valmin  = max (0.2, valmin)
                    valmax  = max (0.5, valmax)            
                self.para.space[:, i] = np.array([valmin, valmax, step], dtype=np.float32)
        self.para.space2true()
        paraval             = np.array(paralst, dtype = np.float32)
        self.para.paraval[:]= paraval
        return
    # 
    def para2mod(self):
        """
        Convert paratemers (for perturbation) to model parameters
        """
        for i in xrange(self.para.npara):
            val = self.para.paraval[i]
            ig  = int(self.para.paraindex[4, i])
            # velocity coeficient for splines
            if int(self.para.paraindex[0, i]) == 0:
                ip                  = int(self.para.paraindex[5, i])
                self.cvel[ip , ig]  = val
            # total thickness of the group
            elif int(self.para.paraindex[0, i]) == 1:
                self.thickness[ig]  = val
            # vp/vs ratio
            elif int(self.para.paraindex[0, i]) == -1:
                self.vpvs[ig]       = val
            else:
                raise ValueError('Unexpected value in paraindex!')
        return
    
    def isgood(self, m0, m1, g0, g1):
        """
        check the model is good or not
        """
        # velocity constrast, contraint (5) in 4.2 of Shen et al., 2012
        for i in xrange (self.nmod-1):
            if self.vs[0, i+1] < self.vs[-1, i]:
                return False
        if m1 >= self.nmod:
            m1  = self.nmod -1
        if m0 < 0:
            m0  = 0
        if g1 >= self.nmod:
            g1  = self.nmod -1
        if g0 < 0:
            g0  = 0
        # monotonic change
        # velocity constrast, contraint (3) and (4) in 4.2 of Shen et al., 2012
        if m0 <= m1:
            for j in range(m0, m1+1):
                for i in xrange(self.nlay[j]-1):
                    if self.vs[i, j] > self.vs[i+1, j]:
                        return False
        # gradient check
        if g0<=g1:
            for j in range(g0, g1+1):
                if self.vs[0, j] > self.vs[1, j]:
                    return False
        return True
    
    def copy(self):
        """
        return a copy of the object
        """
        outmod          = isomod()
        outmod.init_arr(self.nmod)
        outmod.numbp    = self.numbp.copy()
        outmod.mtype    = self.mtype.copy()
        outmod.thickness= self.thickness.copy()
        outmod.nlay     = self.nlay.copy()
        outmod.vpvs     = self.vpvs.copy()
        outmod.isspl    = self.isspl.copy()
        outmod.spl      = self.spl.copy()
        outmod.ratio    = self.ratio.copy()
        outmod.cvel     = self.cvel.copy()
        outmod.vs       = self.vs.copy()
        outmod.hArr     = self.hArr.copy()
        outmod.t        = self.t.copy()
        outmod.para     = self.para.copy()
        return outmod
    
####################################################
# Predefine the parameters for the ttimod object
####################################################
spec_ttimod = [
        ('nmod',        numba.int32),
        ('maxlay',      numba.int32),
        ('maxspl',      numba.int32),
        # number of control points
        ('numbp',       numba.int32[:]),
        ('mtype',       numba.int32[:]),
        ('thickness',   numba.float32[:]),
        ('nlay',        numba.int32[:]),
        ('vpvs',        numba.float32[:]),
        ('isspl',       numba.int32[:]),
        # arrays
        ('spl',         numba.float32[:, :, :]),
        ('ratio',       numba.float32[:, :]),
        # model coefficients 
        ('cvph',        numba.float32[:, :]),
        ('cvpv',        numba.float32[:, :]),
        ('cvsh',        numba.float32[:, :]),
        ('cvsv',        numba.float32[:, :]),
        ('ceta',        numba.float32[:, :]),
        ('crho',        numba.float32[:, :]),
        # model arrays
        ('vph',         numba.float32[:, :]),
        ('vpv',         numba.float32[:, :]),
        ('vsh',         numba.float32[:, :]),
        ('vsv',         numba.float32[:, :]),
        ('eta',         numba.float32[:, :]),
        ('rho',         numba.float32[:, :]),
        # orientation angles
        ('cdip',        numba.float32[:, :]),
        ('cstrike',     numba.float32[:, :]),
        ('dip',         numba.float32[:, :]),
        ('strike',      numba.float32[:, :]),
        ('maxtilt',     numba.int32),
        #
        ('dipjump',     numba.float32),
        
        ('hArr',        numba.float32[:, :]),
        ('t',           numba.float32[:, :]),
        # para1d object
        ('para',        para1d_type)
        ]

@numba.jitclass(spec_ttimod)
class ttimod(object):
    """
    An object for handling parameterization of 1D tilted TI model for the inversion
    =====================================================================================================================
    ::: parameters :::
    :   numbers     :
    nmod        - number of model groups
    maxlay      - maximum layers for each group (default - 100)
    maxspl      - maximum spline coefficients for each group (default - 20)
    :   1D arrays   :
    numbp       - number of control points/basis (1D int array with length nmod)
    mtype       - model parameterization types (1D int array with length nmod)
                    1   - layer         - nlay  = numbp, hArr = ratio*thickness, vs = cvel
                    2   - B-splines     - hArr  = thickness/nlay, vs    = (cvel*spl)_sum over numbp
                    4   - gradient layer- nlay is defined depends on thickness
                                            hArr  = thickness/nlay, vs  = from cvel[0, i] to cvel[1, i]
                    5   - water         - nlay  = 1, vs = 0., hArr = thickness
    thickness   - thickness of each group (1D float array with length nmod)
    nlay        - number of layres in each group (1D int array with length nmod)
    vpvs        - vp/vs ratio in each group (1D float array with length nmod)
    isspl       - flag array indicating the existence of basis B spline (1D int array with length nmod)
                    0 - spline basis has NOT been computed
                    1 - spline basis has been computed
    :   multi-dim arrays    :
    t           - knot vectors for B splines (2D array - [:(self.numb[i]+degBs), i]; i indicating group id)
    spl         - B spline basis array (3D array - [:self.numb[i], :self.nlay[i], i]; i indicating group id)
                    ONLY used for mtype == 2
    ratio       - array for the ratio of each layer (2D array - [:self.nlay[i], i]; i indicating group id)
                    ONLY used for mtype == 1
    cvel        - velocity coefficients (2D array - [:self.numbp[i], i]; i indicating group id)
                    layer mod   - input velocities for each layer
                    spline mod  - coefficients for B spline
                    gradient mod- top/bottom layer velocity
    :   model arrays        :
    vs          - vs velocity arrays (2D array - [:self.nlay[i], i]; i indicating group id)
    hArr        - layer arrays (2D array - [:self.nlay[i], i]; i indicating group id)
    :   para1d  :
    para        - object storing parameters for perturbation
    =====================================================================================================================
    """
    
    def __init__(self):

        self.nmod       = 0
        self.maxlay     = 100
        self.maxspl     = 20
        self.para       = para1d()
        self.maxtilt    = 5
        self.dipjump    = -1.
        return
    
    def init_arr(self, nmod):
        """
        initialization of arrays
        """
        self.nmod       = nmod
        # arrays of size nmod
        self.numbp      = np.zeros(np.int64(self.nmod), dtype=np.int32)
        self.mtype      = np.zeros(np.int64(self.nmod), dtype=np.int32)
        self.thickness  = np.zeros(np.int64(self.nmod), dtype=np.float32)
        self.nlay       = np.ones(np.int64(self.nmod), dtype=np.int32)*np.int32(20) 
        self.vpvs       = np.ones(np.int64(self.nmod), dtype=np.float32)*np.float32(1.75)
        self.isspl      = np.zeros(np.int64(self.nmod), dtype=np.int32)
        # arrays of size maxspl, nmod
        self.cvph       = np.zeros((np.int64(self.maxspl), np.int64(self.nmod)), dtype = np.float32)
        self.cvpv       = np.zeros((np.int64(self.maxspl), np.int64(self.nmod)), dtype = np.float32)
        self.cvsh       = np.zeros((np.int64(self.maxspl), np.int64(self.nmod)), dtype = np.float32)
        self.cvsv       = np.zeros((np.int64(self.maxspl), np.int64(self.nmod)), dtype = np.float32)
        self.ceta       = np.zeros((np.int64(self.maxspl), np.int64(self.nmod)), dtype = np.float32)
        self.crho       = np.zeros((np.int64(self.maxspl), np.int64(self.nmod)), dtype = np.float32)
        self.t          = np.zeros((np.int64(self.maxspl), np.int64(self.nmod)), dtype = np.float32)
        # tilt angles
        self.cdip       = np.zeros((np.int64(self.maxtilt), np.int64(self.nmod)), dtype = np.float32)
        self.cstrike    = np.zeros((np.int64(self.maxtilt), np.int64(self.nmod)), dtype = np.float32)
        # arrays of size maxlay, nmod
        self.ratio      = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.vph        = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.vpv        = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.vsh        = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.vsv        = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.eta        = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.rho        = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.dip        = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.strike     = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        self.hArr       = np.zeros((np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        # arrays of size maxspl, maxlay, nmod
        self.spl        = np.zeros((np.int64(self.maxspl), np.int64(self.maxlay), np.int64(self.nmod)), dtype = np.float32)
        return
    
    def bspline(self, i):
        """
        Compute B-spline basis
        The actual degree is k = degBs - 1
        e.g. nBs = 5, degBs = 4, k = 3, cubic B spline
        ::: output :::
        self.spl    - (nBs+k, npts)
                        [:nBs, :] B spline basis for nBs control points
                        [nBs:,:] can be ignored
        """
        if self.thickness[i] >= 150:
            self.nlay[i]    = 60
        elif self.thickness[i] < 10:
            self.nlay[i]    = 5
        elif self.thickness[i] < 20:
            self.nlay[i]    = 10
        else:
            self.nlay[i]    = 30
            
        if self.isspl[i] == 1:
            print("spline basis already exists!")
            return
        if self.mtype[i] != 2:
            print('Not spline parameterization!')
            return 
        # initialize
        if i >= self.nmod:
            raise ValueError('index for spline group out of range!')
            return
        nBs         = self.numbp[i]
        if nBs < 4:
            degBs   = 3
        else:
            degBs   = 4
        zmin_Bs     = 0.
        zmax_Bs     = self.thickness[i]
        disfacBs    = 2.
        npts        = self.nlay[i]
        t, nbasis   = bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts)
        m           = nBs-1+degBs
        if m > self.maxspl:
            raise ValueError('number of splines is too large, change default maxspl!')
        # # # self.spl[:m, :npts, i]  = nbasis[:m, :]
        self.spl[:nBs, :npts, i]= nbasis[:nBs, :]
        self.isspl[i]           = 1
        self.t[:m+1,i]          = t
        return 

    def update(self):
        """
        Update model (vs and hArr arrays)
        """
        for i in xrange(self.nmod):
            if self.nlay[i] > self.maxlay:
                raise ValueError('number of layers is too large, need change default maxlay!')
            # layered model
            if self.mtype[i] == 1:
                self.nlay[i]                = self.numbp[i]
                self.hArr[:, i]             = self.ratio[:, i] * self.thickness[i]
                self.vph[:self.nlay[i], i]  = self.cvph[:self.nlay[i], i]
                self.vpv[:self.nlay[i], i]  = self.cvpv[:self.nlay[i], i]
                self.vsh[:self.nlay[i], i]  = self.cvsh[:self.nlay[i], i]
                self.vsv[:self.nlay[i], i]  = self.cvsv[:self.nlay[i], i]
                self.eta[:self.nlay[i], i]  = self.ceta[:self.nlay[i], i]
                self.rho[:self.nlay[i], i]  = self.crho[:self.nlay[i], i]
                tnlay           = self.nlay[i]
                #------------------------------------
                # orientation angles
                #------------------------------------
                if self.dipjump < 0.:
                    self.dip[:tnlay, i]      = np.ones(tnlay, dtype = np.float32) * self.cdip[0, i]
                else:
                    ratiosum                = self.ratio.cumsum()
                    ind                     = int(np.where(ratiosum <= self.dipjump)[0][-1])
                    self.dip[:ind, i]       = np.ones(ind, dtype = np.float32) * self.cdip[0, i]
                    self.dip[ind:tnlay, i]  = np.ones(tnlay-ind, dtype = np.float32) * self.cdip[1, i]
                self.strike[:tnlay, i]      = np.ones(tnlay, dtype = np.float32) * self.cstrike[0, i]
            
            # B spline model
            elif self.mtype[i] == 2:
                # self.isspl[i]   = 0
                # self.bspline(i)
                if self.isspl[i] != 1:
                    self.bspline(i)
                for ilay in xrange(self.nlay[i]):
                    tvph    = 0.
                    tvpv    = 0.
                    tvsh    = 0.
                    tvsv    = 0.
                    teta    = 0.
                    for ibs in xrange(self.numbp[i]):
                        tvph= tvph + self.spl[ibs, ilay, i] * self.cvph[ibs, i]
                        tvpv= tvpv + self.spl[ibs, ilay, i] * self.cvpv[ibs, i]
                        tvsh= tvsh + self.spl[ibs, ilay, i] * self.cvsh[ibs, i]
                        tvsv= tvsv + self.spl[ibs, ilay, i] * self.cvsv[ibs, i]
                        teta= teta + self.spl[ibs, ilay, i] * self.ceta[ibs, i]
                    
                    self.vph[ilay, i]   = tvph
                    self.vpv[ilay, i]   = tvpv
                    self.vsh[ilay, i]   = tvsh
                    self.vsv[ilay, i]   = tvsv
                    self.eta[ilay, i]   = teta
                    self.hArr[ilay, i]  = self.thickness[i]/self.nlay[i]
                #------------------------------------
                # orientation angles
                #------------------------------------
                tnlay                       = self.nlay[i]
                if self.dipjump < 0.:
                    self.dip[:tnlay, i]     = np.ones(tnlay, dtype = np.float32) * self.cdip[0, i]
                else:
                    ind                     = int(tnlay*self.dipjump)
                    self.dip[:ind, i]       = np.ones(ind, dtype = np.float32) * self.cdip[0, i]
                    self.dip[ind:tnlay, i]  = np.ones(tnlay-ind, dtype = np.float32) * self.cdip[1, i]
                self.strike[:tnlay, i]      = np.ones(tnlay, dtype = np.float32) * self.cstrike[0, i]
            # gradient layer
            elif self.mtype[i] == 4:
                nlay 	    = 4
                if self.thickness[i] >= 20.:
                    nlay    = 20
                if self.thickness[i] > 10. and self.thickness[i] < 20.:
                    nlay    = int(self.thickness[i]/1.)
                if self.thickness[i] > 2. and self.thickness[i] <= 10.:
                    nlay    = int(self.thickness[i]/0.5)
                if self.thickness[i] < 0.5:
                    nlay    = 2
                dh 	        = self.thickness[i]/float(nlay)
                dcvph 		= (self.cvph[1, i] - self.cvph[0, i])/(nlay - 1.)
                dcvpv 		= (self.cvpv[1, i] - self.cvpv[0, i])/(nlay - 1.)
                dcvsh 		= (self.cvsh[1, i] - self.cvsh[0, i])/(nlay - 1.)
                dcvsv 		= (self.cvsv[1, i] - self.cvsv[0, i])/(nlay - 1.)
                dcrho       = (self.crho[1, i] - self.crho[0, i])/(nlay - 1.)
                vph         = np.zeros(np.int64(nlay), dtype=np.float32)
                vpv         = np.zeros(np.int64(nlay), dtype=np.float32)
                vsh         = np.zeros(np.int64(nlay), dtype=np.float32)
                vsv         = np.zeros(np.int64(nlay), dtype=np.float32)
                rho         = np.zeros(np.int64(nlay), dtype=np.float32)
                for ilay in xrange(nlay):
                    vph[ilay]       = self.cvph[0, i] + dcvph*np.float32(ilay)
                    vpv[ilay]       = self.cvpv[0, i] + dcvpv*np.float32(ilay)
                    vsh[ilay]       = self.cvsh[0, i] + dcvsh*np.float32(ilay)
                    vsv[ilay]       = self.cvsv[0, i] + dcvsv*np.float32(ilay)
                    rho[ilay]       = self.crho[0, i] + dcrho*np.float32(ilay)
                hArr 	            = np.ones(nlay, dtype=np.float32)*np.float32(dh)
                self.vph[:nlay, i]  = vph
                self.vpv[:nlay, i]  = vpv
                self.vsh[:nlay, i]  = vsh
                self.vsv[:nlay, i]  = vsv
                self.rho[:nlay, i]  = rho
                
                self.eta[:nlay, i]  = np.ones(nlay, dtype=np.float32)*self.ceta[0, i]
                self.hArr[:nlay, i] = hArr
                self.nlay[i]        = nlay
                #------------------------------------
                # orientation angles
                #------------------------------------
                tnlay                       = nlay
                if self.dipjump < 0.:
                    self.dip[:tnlay, i]     = np.ones(tnlay, dtype = np.float32) * self.cdip[0, i]
                else:
                    ind                     = int(tnlay*self.dipjump)
                    self.dip[:ind, i]       = np.ones(ind, dtype = np.float32) * self.cdip[0, i]
                    self.dip[ind:tnlay, i]  = np.ones(tnlay-ind, dtype = np.float32) * self.cdip[1, i]
                self.strike[:tnlay, i]      = np.ones(tnlay, dtype = np.float32) * self.cstrike[0, i]
            # water layer
            elif self.mtype[i] == 5:
                nlay                = 1
                self.vph[0, i]      = self.cvph[0, i]
                self.vph[0, i]      = self.cvpv[0, i]
                self.vsh[0, i]      = 0.
                self.vsv[0, i]      = 0.
                self.eta[0, i]      = 1.
                self.rho[0, i]      = 1.02
                self.dip[0, i]      = 0.
                self.strike[0, i]   = 0.
                
                self.hArr[0, i]     = self.thickness[i]
                self.nlay[i]        = 1
        return
    
    def get_rho(self):
        for i in xrange(self.nmod):
            if self.mtype[i] == 5:
                self.rho[0, i]          = 1.02
                continue
            ind                         = self.vsv[:, i]*self.vpvs[i] > 7.5
            self.rho[:self.nlay[i], i]  = 0.541 + 0.3601*self.vsv[:self.nlay[i], i]*self.vpvs[i]
            self.rho[ind, i]            = 3.35
        return

    def get_vmodel(self):
        """
        get velocity models
        ==========================================================================
        ::: output :::
        hArr, vph, vpv, vsh, vsv, eta, rho, dip, strike
        ==========================================================================
        """
        vph     = []
        vpv     = []
        vsh     = []
        vsv     = []
        eta     = []
        rho     = []
        dip     = []
        strike  = []
        hArr    = []
        depth   = 0.
        for i in xrange(self.nmod):
            for j in xrange(self.nlay[i]):
                hArr.append(self.hArr[j, i])
                depth += self.hArr[j, i]
                vph.append(self.vph[j, i])
                vpv.append(self.vpv[j, i])
                vsh.append(self.vsh[j, i])
                vsv.append(self.vsv[j, i])
                eta.append(self.eta[j, i])
                rho.append(self.rho[j, i])
                dip.append(self.dip[j, i])
                strike.append(self.strike[j, i])
                
        vph     = np.array(vph, dtype=np.float32)
        vpv     = np.array(vpv, dtype=np.float32)
        vsh     = np.array(vsh, dtype=np.float32)
        vsv     = np.array(vsv, dtype=np.float32)
        eta     = np.array(eta, dtype=np.float32)
        rho     = np.array(rho, dtype=np.float32)
        
        dip     = np.array(dip, dtype=np.float32)
        strike  = np.array(strike, dtype=np.float32)
        hArr    = np.array(hArr, dtype=np.float32)
        return hArr, vph, vpv, vsh, vsv, eta, rho, dip, strike
    
    def get_paraind_US(self):
        """
        get parameter index arrays for para

        
        references:
        Xie, J., M.H. Ritzwoller, S. Brownlee, and B. Hacker,
            Inferring the oriented elastic tensor from surface wave observations: Preliminary application across the Western US,
                Geophys. J. Int., 201, 996-1021, 2015.
        """
        npara   = (self.numbp[1:]).sum()*5 + 4
        self.para.init_arr(npara)
        ipara   = 0
        for i in xrange(self.nmod):
            if i == 0:
                continue # sediment, no perturbation
            for j in xrange(self.numbp[i]):
                # vph
                self.para.paraindex[0, ipara]   = 0
                self.para.paraindex[1, ipara]   = -1
                self.para.paraindex[2, ipara]   = 15. # +- 15 %
                self.para.paraindex[3, ipara]   = 0.05
                self.para.paraindex[4, ipara]   = i
                self.para.paraindex[5, ipara]   = j
                ipara   +=1
                # vpv
                self.para.paraindex[0, ipara]   = 1
                self.para.paraindex[1, ipara]   = -1
                self.para.paraindex[2, ipara]   = 15. # +- 15 %
                self.para.paraindex[3, ipara]   = 0.05
                self.para.paraindex[4, ipara]   = i
                self.para.paraindex[5, ipara]   = j
                ipara   +=1
                # vsh
                self.para.paraindex[0, ipara]   = 2
                self.para.paraindex[1, ipara]   = -1
                self.para.paraindex[2, ipara]   = 15. # +- 15 %
                self.para.paraindex[3, ipara]   = 0.05
                self.para.paraindex[4, ipara]   = i
                self.para.paraindex[5, ipara]   = j
                ipara   +=1
                # vsv
                self.para.paraindex[0, ipara]   = 3
                self.para.paraindex[1, ipara]   = -1
                self.para.paraindex[2, ipara]   = 5. # +- 15 %
                self.para.paraindex[3, ipara]   = 0.05
                self.para.paraindex[4, ipara]   = i
                self.para.paraindex[5, ipara]   = j
                ipara   +=1
                # eta
                self.para.paraindex[0, ipara]   = 4
                self.para.paraindex[1, ipara]   = 1
                self.para.paraindex[2, ipara]   = 15. # +- 15 %
                self.para.paraindex[3, ipara]   = 0.01
                self.para.paraindex[4, ipara]   = i
                self.para.paraindex[5, ipara]   = j
                ipara   +=1
        #--------------------------
        # orientation angles
        #--------------------------
        # crust
        self.para.paraindex[0, ipara]   = 5
        self.para.paraindex[1, ipara]   = 1
        self.para.paraindex[2, ipara]   = 100.
        self.para.paraindex[3, ipara]   = 1.
        self.para.paraindex[4, ipara]   = 1
        self.para.paraindex[5, ipara]   = 0
        ipara   += 1
        self.para.paraindex[0, ipara]   = 6
        self.para.paraindex[1, ipara]   = 1
        self.para.paraindex[2, ipara]   = 100.
        self.para.paraindex[3, ipara]   = 1.
        self.para.paraindex[4, ipara]   = 1
        self.para.paraindex[5, ipara]   = 0
        ipara   += 1
        # mantle
        self.para.paraindex[0, ipara]   = 5
        self.para.paraindex[1, ipara]   = 1
        self.para.paraindex[2, ipara]   = 100.
        self.para.paraindex[3, ipara]   = 1.
        self.para.paraindex[4, ipara]   = 2
        self.para.paraindex[5, ipara]   = 0
        ipara   += 1
        self.para.paraindex[0, ipara]   = 6
        self.para.paraindex[1, ipara]   = 1
        self.para.paraindex[2, ipara]   = 100.
        self.para.paraindex[3, ipara]   = 1.
        self.para.paraindex[4, ipara]   = 2
        self.para.paraindex[5, ipara]   = 0
        ipara   += 1
        if ipara != npara:
            raise ValueError('Inconsistent number of parameters for perturbation!')
        # 
        # if self.nmod >= 3:
        #     # sediment thickness
        #     self.para.paraindex[0, ipara]   = 1
        #     self.para.paraindex[1, ipara]   = -1
        #     self.para.paraindex[2, ipara]   = 100.
        #     self.para.paraindex[3, ipara]   = 0.1
        #     self.para.paraindex[4, ipara]   = 0
        #     ipara   += 1
        # # crustal thickness
        # self.para.paraindex[0, ipara]   = 1
        # self.para.paraindex[1, ipara]   = -1
        # self.para.paraindex[2, ipara]   = 20.
        # self.para.paraindex[3, ipara]   = 1.
        # if self.nmod >= 3:
        #     self.para.paraindex[4, ipara]   = 1.
        # else:
        #     self.para.paraindex[4, ipara]   = 0.
        return
                
        
    def mod2para(self):
        """
        convert model to parameter arrays for perturbation
        """
        paralst     = [] 
        for i in xrange(self.para.npara):
            ig      = int(self.para.paraindex[4, i])
            # vph coefficient 
            if int(self.para.paraindex[0, i]) == 0:
                ip  = int(self.para.paraindex[5, i])
                val = self.cvph[ip , ig]
            # vpv coefficient 
            elif int(self.para.paraindex[0, i]) == 1:
                ip  = int(self.para.paraindex[5, i])
                val = self.cvpv[ip , ig]
            # vsh coefficient 
            elif int(self.para.paraindex[0, i]) == 2:
                ip  = int(self.para.paraindex[5, i])
                val = self.cvsh[ip , ig]
            # vsv coefficient 
            elif int(self.para.paraindex[0, i]) == 3:
                ip  = int(self.para.paraindex[5, i])
                val = self.cvsv[ip , ig]
            # eta coefficient 
            elif int(self.para.paraindex[0, i]) == 4:
                ip  = int(self.para.paraindex[5, i])
                val = self.ceta[ip , ig]
            # dip
            elif int(self.para.paraindex[0, i]) == 5:
                ip  = int(self.para.paraindex[5, i])
                val = self.cdip[ip , ig]
            # strike
            elif int(self.para.paraindex[0, i]) == 6:
                ip  = int(self.para.paraindex[5, i])
                val = self.cstrike[ip , ig]
            
            # # vph coefficient 
            # elif int(self.para.paraindex[0, i]) == 5:
            #     ip  = int(self.para.paraindex[5, i])
            #     val = self.cvph[ip , ig]

            # # total thickness of the group
            # elif int(self.para.paraindex[0, i]) == 1:
            #     val = self.thickness[ig]
            # # vp/vs ratio
            # elif int(self.para.paraindex[0, i]) == -1:
            #     val = self.vpvs[ig]
            else:
                raise ValueError('Unexpected value in paraindex!')
            paralst.append(val)
            #-------------------------------------------
            # defining parameter space for perturbation
            #-------------------------------------------
            if not self.para.isspace:
                step    = self.para.paraindex[3, i]
                if int(self.para.paraindex[1, i]) == 1:
                    valmin  = val - self.para.paraindex[2, i]
                    valmax  = val + self.para.paraindex[2, i]
                else:
                    valmin  = val - val*self.para.paraindex[2, i]/100.
                    valmax  = val + val*self.para.paraindex[2, i]/100.
                valmin  = max (0.,valmin)
                valmax  = max (valmin + 0.0001, valmax)
                # if (self.para.paraindex[0, i] == 0 and i == 0 and self.para.paraindex[5, i] == 0): # if it is the upper sedi:
                #     valmin  = max (0.2, valmin)
                #     valmax  = max (0.5, valmax)
                # eta
                if int(self.para.paraindex[0, i]) == 4:
                    if int(self.para.paraindex[4, i]) == 1:
                        valmin  = 0.6
                        valmax  = 1.1
                        step    = 0.01
                    else:
                        valmin  = 0.85
                        valmax  = 1.1
                        step    = 0.01
                # dip
                if int(self.para.paraindex[0, i]) == 5:
                    valmin      = 0.
                    valmax      = 90.
                    step        = -1.
                # strike
                if int(self.para.paraindex[0, i]) == 6:
                    valmin      = 0.
                    valmax      = 180.
                    step        = -1.
                self.para.space[:, i] = np.array([valmin, valmax, step], dtype=np.float32)
        self.para.space2true()
        paraval             = np.array(paralst, dtype = np.float32)
        self.para.paraval[:]= paraval
        return
    
    def para2mod(self):
        """
        Convert paratemers (for perturbation) to model parameters
        """
        for i in xrange(self.para.npara):
            val     = self.para.paraval[i]
            ig      = int(self.para.paraindex[4, i])
            # vph coefficient 
            if int(self.para.paraindex[0, i]) == 0:
                ip                  = int(self.para.paraindex[5, i])
                self.cvph[ip , ig]  = val
            # vpv coefficient 
            elif int(self.para.paraindex[0, i]) == 1:
                ip                  = int(self.para.paraindex[5, i])
                self.cvpv[ip , ig]  = val
            # vsh coefficient 
            elif int(self.para.paraindex[0, i]) == 2:
                ip                  = int(self.para.paraindex[5, i])
                self.cvsh[ip , ig]  = val
            # vsv coefficient 
            elif int(self.para.paraindex[0, i]) == 3:
                ip                  = int(self.para.paraindex[5, i])
                self.cvsv[ip , ig]  = val
            # eta coefficient 
            elif int(self.para.paraindex[0, i]) == 4:
                ip                  = int(self.para.paraindex[5, i])
                self.ceta[ip , ig]  = val
            # dip
            elif int(self.para.paraindex[0, i]) == 5:
                ip                  = int(self.para.paraindex[5, i])
                self.cdip[ip , ig]  = val
            # strike
            elif int(self.para.paraindex[0, i]) == 6:
                ip                  = int(self.para.paraindex[5, i])
                self.cstrike[ip,ig] = val
            else:
                raise ValueError('Unexpected value in paraindex!')
        return
    
    def isgood(self, m0, m1, g0, g1):
        """
        check the model is good or not
        """
        # velocity constrast, contraint (5) in 4.2 of Shen et al., 2012
        for i in xrange (self.nmod-1):
            if self.vsv[0, i+1] < self.vsv[-1, i]:
                return False
            if self.vsh[0, i+1] < self.vsh[-1, i]:
                return False
            if self.vpv[0, i+1] < self.vpv[-1, i]:
                return False
            if self.vph[0, i+1] < self.vph[-1, i]:
                return False
        
        for i in xrange(self.nmod):
            for j in xrange(self.nlay[i]):
                # inherent anisotropy must be positive
                if self.vsh[i, j] < self.vsv[i,j]:
                    return False
                if self.vph[i, j] < self.vpv[i,j]:
                    return False
                # vp/vs ratio must be within [1.65, 1.85]
                vpvs    = (self.vph + self.vpv)/(self.vsh+self.vsv)
                if vpvs < 1.65 or vpvs > 1.85:
                    return False
        if m1 >= self.nmod:
            m1  = self.nmod -1
        if m0 < 0:
            m0  = 0
        if g1 >= self.nmod:
            g1  = self.nmod -1
        if g0 < 0:
            g0  = 0
        # monotonic change
        # velocity constrast, contraint (3) and (4) in 4.2 of Shen et al., 2012
        if m0 <= m1:
            for j in range(m0, m1+1):
                for i in xrange(self.nlay[j]-1):
                    if self.vsv[i, j] > self.vsv[i+1, j]:
                        return False
                    if self.vsh[i, j] > self.vsh[i+1, j]:
                        return False
                    if self.vpv[i, j] > self.vpv[i+1, j]:
                        return False
                    if self.vph[i, j] > self.vph[i+1, j]:
                        return False
                    
        # gradient check
        if g0<=g1:
            for j in range(g0, g1+1):
                if self.vs[0, j] > self.vs[1, j]:
                    return False
        return True
    
    def copy(self):
        """
        return a copy of the object
        """
        outmod          = ttimod()
        outmod.init_arr(self.nmod)
        outmod.numbp    = self.numbp.copy()
        outmod.mtype    = self.mtype.copy()
        outmod.thickness= self.thickness.copy()
        outmod.nlay     = self.nlay.copy()
        outmod.vpvs     = self.vpvs.copy()
        outmod.isspl    = self.isspl.copy()
        outmod.spl      = self.spl.copy()
        outmod.ratio    = self.ratio.copy()
        
        outmod.cvph     = self.cvph.copy()
        outmod.cvpv     = self.cvpv.copy()
        outmod.cvsh     = self.cvsh.copy()
        outmod.cvsv     = self.cvsv.copy()
        outmod.ceta     = self.ceta.copy()
        outmod.crho     = self.crho.copy()
        outmod.cdip     = self.cdip.copy()
        outmod.cstrike  = self.cstrike.copy()
        
        outmod.dipjump  = self.dipjump
        
        outmod.vph      = self.vph.copy()
        outmod.vpv      = self.vpv.copy()
        outmod.vsh      = self.vsh.copy()
        outmod.vsv      = self.vsv.copy()
        outmod.eta      = self.eta.copy()
        outmod.rho      = self.rho.copy()
        outmod.dip      = self.dip.copy()
        outmod.strike   = self.strike.copy()
        
        
        outmod.hArr     = self.hArr.copy()
        outmod.t        = self.t.copy()
        outmod.para     = self.para.copy()
        return outmod
    