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
        if ptype == 0:
            for i in xrange(self.npara):
                newval  = np.random.uniform(self.space[0, i], self.space[1, i])
                paralst.append(newval)
        else:
            for i in xrange(self.npara):
                oldval 	= self.paraval[i]
                step 	= self.space[2, i]
                run 	= True
                j		= 0
                while (run and j<10000): 
                    newval  = random.gauss(oldval, step)
                    if (newval >= self.space[0, i] and newval <= self.space[1, i]):
                        run = False
                    j   +=1
                paralst.append(newval)
        self.paraval = np.array(paralst, dtype=np.float32)
        return
        
    def copy(self):
        """
        return a copy of the object
        """
        outpara             = para1d()
        outpara.init_arr(self.npara)
        outpara.paraindex   = self.paraindex.copy()
        outpara.numbind     = self.numbind.copy()
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
# Predefine the parameters for the isospl object
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
                if self.isspl[i] != 1:
                    self.bspline(i)
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
        """
        npara   = self.numbp.sum()  + self.nmod - 1
        self.para.init_arr(npara)
        ipara   = 0
        for i in xrange(self.nmod):
            for j in xrange(self.numbp[i]):
                self.para.paraindex[0, ipara]   = 0
                # sediment
                if i == 0:
                    self.para.paraindex[1, ipara]   = 1
                    self.para.paraindex[2, ipara]   = 1.
                else:
                    self.para.paraindex[1, ipara]   = -1
                    self.para.paraindex[2, ipara]   = 20.
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
 