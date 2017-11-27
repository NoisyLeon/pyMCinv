# -*- coding: utf-8 -*-
"""
Module for handling parameterization of the model

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
from __future__ import division

from libc.math cimport sqrt, exp, log, pow, fmax, fmin
from libc.time cimport time,time_t
from posix.time cimport clock_gettime, timespec, CLOCK_REALTIME
from cython.parallel import parallel, prange, threadid
from libc.stdlib cimport rand, srand, RAND_MAX, malloc, free
from libc.stdio cimport printf
cimport cython
#from cython.view cimport array as cvarray

#import random
import numpy as np
cimport numpy as np

cdef time_t t = time(NULL)
srand(t)

cdef float random_uniform(float a, float b) nogil:
#    cdef timespec ts
#    cdef unsigned int current
#    clock_gettime(CLOCK_REALTIME, &ts)
#    current = ts.tv_nsec 
#    srand(current)
    cdef float r = rand()
    return float(r/RAND_MAX)*(b-a)+a


cdef float random_gauss(float mu, float sigma) nogil:
    cdef float x1, x2, w
    w = 2.0
    while (w >= 1.0 or w == 0.):
        x1 = 2.0 * random_uniform(0., 1.) - 1.0
        x2 = 2.0 * random_uniform(0., 1.) - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return mu + sigma*(x1 * w)


def random_uniform_interface(float a, float b):
    return random_uniform(a, b)

def test(int N):
    cdef float[:] out = np.zeros(N, dtype=np.float32)
    cdef int i
    with nogil, parallel(num_threads=5):
        for i in prange(N):
            out[i]  = random_uniform(0., 1.)
#            out[i]  = random_gauss(0., 1.)
    return out


cdef class para1d:
    """
    An object for handling parameter perturbations
    =====================================================================================================================
    ::: parameters :::
    :   values  :
    npara       - number of parameters for perturbations
    maxind      - maximum number of index for each parameters
    isspace     - if space array is computed or not
    :   arrays  :
    paraval     - parameter array for perturbation
    paraindex   - index array indicating numerical setup for each parameter
                1.  isomod
                    paraindex[0, :] - type of parameters
                                        0   - velocity coefficient for splines
                                        1   - thickness
                                       -1   - vp/vs ratio
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
                                        
                                        below are currently not used yet
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
        self.maxind         = 6
        self.isspace        = 0
        return
    
    cpdef init_arr(self, int npara):
        """
        initialize the arrays
        """
        self.npara          = npara
        self.paraval        = np.zeros(self.npara, dtype=np.float32)
        self.paraindex      = np.zeros((self.maxind, self.npara), dtype = np.float32)
        self.space          = np.zeros((3, self.npara), dtype = np.float32)
        return
    
    def readparatxt(self, str infname):
        """
        read txt perturbation parameter file
        ==========================================================================
        ::: input :::
        infname - input file name
        ==========================================================================
        """
        cdef int npara      = 0
        cdef Py_ssize_t i   = 0
        cdef Py_ssize_t j
        cdef int ne
        cdef np.ndarray temp
        for l1 in open(infname,"r"):
            npara   += 1
        print "Number of parameters for perturbation: %d " % npara
        self.init_arr(npara)
        
        with open(infname, 'r') as fid:
            for line in fid.readlines():
                temp                        = np.array(line.split(), dtype=np.float32)
                ne                          = temp.size
                for j in range(ne):
                    self.paraindex[j, i]    = temp[j]
                i                           += 1
        # print "read para over!"
        return
        
        
    def write_paraval_txt(self, str outfname):
        np.savetxt(outfname, self.paraval, fmt='%g')
        return
    
    def read_paraval_txt(self, str infname):
        self.paraval  = np.loadtxt(infname, dtype=np.float32)
        return

    @cython.boundscheck(False)
    cdef int new_paraval(self, int ptype) nogil:
        """
        peturb parameters in paraval array
        ===============================================================================
        ::: input :::
        ptype   - perturbation type
                    0   - uniform random value generated from parameter space
                    1   - Gauss random number generator given mu = oldval, sigma=step
        ===============================================================================
        """
        cdef Py_ssize_t i, j
        cdef float oldval, newval, step
        cdef int run
        if not self.isspace:
            printf('Parameter space for perturbation has not been initialized yet!\n')
            return 0
        for i in range(self.npara):
            if ptype  == 1 and self.space[2, i] > 0.:
                oldval 	= self.paraval[i]
                step 	= self.space[2, i]
                run 	= 1
                j		= 0
                while (run and j<10000): 
                    newval  = random_gauss(oldval, step)
                    if (newval >= self.space[0, i] and newval <= self.space[1, i]):
                        run = 0
                    j   +=1
            else: 
                newval  = random_uniform(self.space[0, i], self.space[1, i])
            self.paraval[i]     = newval
        return 1

    cpdef copy(self):
        """
        return a copy of the object
        """
        outpara             = para1d()
        outpara.init_arr(self.npara)
        outpara.paraindex   = self.paraindex.copy()
        outpara.paraval     = self.paraval.copy()
        outpara.isspace     = self.isspace
        outpara.space       = self.space.copy()
        return outpara
    
    @cython.boundscheck(False)
    cdef void get_para(self, para1d inpara) nogil:
        cdef Py_ssize_t i, j
        if inpara.npara != self.npara:
            with gil:
                self.init_arr(inpara.npara)
        for i in range(self.npara):
            self.paraval[i]         = inpara.paraval[i]
            for j in range(self.maxind):
                self.paraindex[j][i]= inpara.paraindex[j][i]
            for j in range(3):
                self.space[j][i]    = inpara.space[j][i]
        return
            
    
@cython.boundscheck(False)
cdef void bspl_basis(int nBs, int degBs, float zmin_Bs, float zmax_Bs, float disfacBs, int npts, 
                float[20][100] &nbasis) nogil:
    cdef int m, n_temp
    cdef Py_ssize_t i, j, pp
    cdef float *depth   = <float *>malloc(npts * sizeof(float))
    cdef float[20][100] obasis
    cdef float step
    #-------------------------------- 
    # defining the knot vector
    #--------------------------------
    m           = nBs-1+degBs    
    cdef float *t       = <float *>malloc((m+1) * sizeof(float))
    for i in range(degBs):
        t[i]    = zmin_Bs + i*(zmax_Bs-zmin_Bs)/10000.
    for i in range(degBs,m+1-degBs):
        n_temp  = m+1-degBs-degBs+1
        if (disfacBs !=1):
            temp= (zmax_Bs-zmin_Bs)*(disfacBs-1)/(pow(disfacBs,n_temp)-1)
        else:
            temp= (zmax_Bs-zmin_Bs)/n_temp
        t[i]    = temp*pow(disfacBs,(i-degBs)) + zmin_Bs
    for i in range(m+1-degBs,m+1):
        t[i]    = zmax_Bs-(zmax_Bs-zmin_Bs)/10000.*(m-i)
    # depth array
    step        = (zmax_Bs-zmin_Bs)/(npts-1)
    for i in range(npts):
        depth[i]= float(i) *step + zmin_Bs
    # arrays for storing B spline basis
    #-------------------------------- 
    # computing B spline basis
    #-------------------------------- 
    # initialize the arrays
    for i in range(20):
        for j in range(100):
            obasis[i][j] = 0.
            nbasis[i][j] = 0.
            
    for i in range (m):
        for j in range (npts):
            if (depth[j] >=t[i] and depth[j]<t[i+1]):
                obasis[i][j] = 1.
            else:
                obasis[i][j] = 0.
    for pp in range (1,degBs):
        for i in range (m-pp):
            for j in range (npts):
                nbasis[i][j] = (depth[j]-t[i])/(t[i+pp]-t[i])*obasis[i][j] + \
                        (t[i+pp+1]-depth[j])/(t[i+pp+1]-t[i+1])*obasis[i+1][j]
        for i in xrange (m-pp):
            for j in xrange (npts):
                obasis[i][j] = nbasis[i][j]
    nbasis[0][0]            = 1
    nbasis[nBs-1][npts-1]   = 1
    free(t)
    free(depth)
    return 
#
def test_spl(int nBs, int degBs, float zmin_Bs, float zmax_Bs, float disfacBs, int npts):
    cdef float[20][100] nbasis
    bspl_basis(nBs=nBs, degBs=degBs, zmin_Bs=zmin_Bs, zmax_Bs=zmax_Bs, disfacBs=disfacBs, npts=npts, nbasis=nbasis)
    return nbasis

cdef class isomod:
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
    
    cpdef init_arr(self, nmod):
        """
        initialization of arrays
        """
        self.nmod       = nmod
        # arrays of size nmod
        self.numbp      = np.zeros(self.nmod, dtype=np.int32)
        self.mtype      = np.zeros(self.nmod, dtype=np.int32)
        self.thickness  = np.zeros(self.nmod, dtype=np.float32)
        self.nlay       = np.ones(self.nmod, dtype=np.int32)*20
        self.vpvs       = np.ones(self.nmod, dtype=np.float32)*1.75
        self.isspl      = np.zeros(self.nmod, dtype=np.int32)
        # arrays of size maxspl, nmod
        self.cvel       = np.zeros((self.maxspl, self.nmod), dtype = np.float32)
        # arrays of size maxlay, nmod
        self.ratio      = np.zeros((self.maxlay, self.nmod), dtype = np.float32)
        self.vs         = np.zeros((self.maxlay, self.nmod), dtype = np.float32)
        self.hArr       = np.zeros((self.maxlay, self.nmod), dtype = np.float32)
        # arrays of size maxspl, maxlay, nmod
        self.spl        = np.zeros((self.maxspl, self.maxlay, self.nmod), dtype = np.float32)
        return
    
    def readmodtxt(self, str infname):
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
        ==========================================================================
        """
        cdef int nmod   = 0
        cdef Py_ssize_t iid, flag, tnp, nr, i
        cdef float thickness
        
        for l1 in open(infname,"r"):
            nmod    += 1
        print "Number of model parameter groups: %d " % nmod
        self.init_arr(nmod)
        
        for l1 in open(infname,"r"):
            l1 			          = l1.rstrip()
            l2 			          = l1.split()
            iid 		          = int(l2[0])
            flag		          = int(l2[1])
            thickness	          = float(l2[2])
            tnp 		          = int(l2[3]) # number of parameters
            self.mtype[iid]	    = flag
            self.thickness[iid] = thickness
            self.numbp[iid]     = tnp 
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
#            cvel        = []
#            ratio       = []
            nr          = 0
            for i in xrange(tnp):
#                cvel.append(float(l2[4+i]))
                self.cvel[i, iid]       = float(l2[4+i])
                if (int(l2[1]) ==1):  # type 1 layer
                    self.ratio[nr, iid] = float(l2[4+tnp+i])
                    nr  += 1
#                    ratio.append(float(l2[4+tnp+i]))
            self.vpvs[iid]         = (float(l2[-1]))-0.
#            cvel                    = np.array(cvel, dtype=np.float32)
#            ratio                   = np.array(ratio, dtype=np.float32)
#            inmod.cvel[:tnp, iid]   = cvel
#            inmod.ratio[:nr, iid]   = ratio
        return True
    
    @cython.boundscheck(False)
    cdef int bspline(self, Py_ssize_t i) nogil:
        """
        Compute B-spline basis
        The actual degree is k = degBs - 1
        e.g. nBs = 5, degBs = 4, k = 3, cubic B spline
        ::: output :::
        self.spl    - (nBs+k, npts)
                        [:nBs, :] B spline basis for nBs control points
                        [nBs:,:] can be ignored
        """
        cdef int nBs, degBs, npts, m
        cdef Py_ssize_t ibs, ilay
        cdef float zmin_Bs, zmax_Bs, disfacBs
        cdef float[20][100] nbasis
        
        if self.thickness[i] >= 150.:
            self.nlay[i]    = 60
        elif self.thickness[i] < 10.:
            self.nlay[i]    = 5
        elif self.thickness[i] < 20.:
            self.nlay[i]    = 10
        else:
            self.nlay[i]    = 30
            
        if self.isspl[i] == 1:
            printf("spline basis already exists!")
            return 0
        if self.mtype[i] != 2:
            printf('Not spline parameterization!')
            return 0
        # initialize
        if i >= self.nmod:
            printf('index for spline group out of range!')
            return 0
        nBs         = self.numbp[i]
        if nBs < 4:
            degBs   = 3
        else:
            degBs   = 4
        zmin_Bs     = 0.
        zmax_Bs     = self.thickness[i]
        disfacBs    = 2.
        npts        = self.nlay[i]
        bspl_basis(nBs, degBs, zmin_Bs, zmax_Bs, disfacBs, npts, nbasis)
        m           = nBs-1+degBs
        if m > self.maxspl:
            printf('number of splines is too large, change default maxspl!')
            return 0
        for ibs in range(nBs):
            for ilay in range(npts):
                self.spl[ibs, ilay, i]= nbasis[ibs][ilay]
        self.isspl[i]           = 1
        return 1
    
    def bspline_inferface(self, Py_ssize_t i):
        self.bspline(i=i)
        return
    
    @cython.boundscheck(False)
    cdef int update(self) nogil:
        """
        Update model (vs and hArr arrays)
        """
        cdef Py_ssize_t i, ilay, ibs
        cdef int nlay
        cdef float tvalue, dh, dcvel
        
        for i in range(self.nmod):
            if self.nlay[i] > self.maxlay:
                printf('number of layers is too large, need change default maxlay!')
                return 0
            # layered model
            if self.mtype[i] == 1:
                self.nlay[i]    = self.numbp[i]
                for ilay in range(self.nlay[i]):
                    self.hArr[ilay, i] = self.ratio[ilay, i] * self.thickness[i]
                    self.vs[ilay, i]   = self.cvel[ilay, i]
            # B spline model
            elif self.mtype[i] == 2:
                self.isspl[i]   = 0
                self.bspline(i)
                # # if self.isspl[i] != 1:
                # #     self.bspline(i)
                for ilay in range(self.nlay[i]):
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
                dcvel 		   = (self.cvel[1, i] - self.cvel[0, i])/(nlay - 1.)
                for ilay in range(nlay):
                    self.vs[ilay, i]    = self.cvel[0, i] + dcvel*float(ilay)
                    self.hArr[ilay, i]  = dh
                self.nlay[i]        = nlay
            # water layer
            elif self.mtype[i] == 5:
                nlay    = 1
                self.vs[0, i]       = 0.
                self.hArr[0, i]     = self.thickness[i]
                self.nlay[i]        = 1
        return 1
    
    def update_interface(self):
        self.update()
        return
    
    @cython.boundscheck(False)
    cdef void get_paraind(self) nogil:
        """
        get parameter index arrays for para
        Table 1 and 2 in Shen et al. 2012
        
        references:
        Shen, W., Ritzwoller, M.H., Schulte-Pelkum, V. and Lin, F.C., 2012.
            Joint inversion of surface wave dispersion and receiver functions: a Bayesian Monte-Carlo approach.
                Geophysical Journal International, 192(2), pp.807-836.
        """
        cdef int npara, numbp_sum
        cdef Py_ssize_t ipara, i, j
        numbp_sum = 0
        for i in range(self.nmod):
            numbp_sum += self.numbp[i]
        npara   = numbp_sum  + self.nmod - 1
        with gil:
            self.para.init_arr(npara)
        ipara   = 0
        for i in range(self.nmod):
            for j in range(self.numbp[i]):
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
    
    def get_paraind_interface(self):
        self.get_paraind()
        return
    
    @cython.boundscheck(False)
    cdef void mod2para(self) nogil:
        """
        convert model to parameter arrays for perturbation
        """
        cdef Py_ssize_t i, j, ig, ip
        cdef float val, step, valmin, valmax
 
        for i in range(self.para.npara):
            ig      = <int>self.para.paraindex[4, i]
            # velocity coefficient 
            if <int>self.para.paraindex[0, i] == 0:
                ip  = int(self.para.paraindex[5, i])
                val = self.cvel[ip][ig]
            # total thickness of the group
            elif <int>self.para.paraindex[0, i] == 1:
                val = self.thickness[ig]
            # vp/vs ratio
            elif <int> self.para.paraindex[0, i] == -1:
                val = self.vpvs[ig]
            else:
                printf('Unexpected value in paraindex!')
            self.para.paraval[i] = val
            #-------------------------------------------
            # defining parameter space for perturbation
            #-------------------------------------------
            if not self.para.isspace:
                step        = self.para.paraindex[3, i]
                if <int>self.para.paraindex[1, i] == 1:
                    valmin  = val - self.para.paraindex[2, i]
                    valmax  = val + self.para.paraindex[2, i]
                else:
                    valmin  = val - val*self.para.paraindex[2, i]/100.
                    valmax  = val + val*self.para.paraindex[2, i]/100.
                valmin  = fmax (0.,valmin)
                valmax  = fmax (valmin + 0.0001, valmax)
                if (<int>self.para.paraindex[0, i] == 0 and i == 0 \
                    and <int> self.para.paraindex[5, i] == 0): # if it is the upper sedi:
                    valmin  = fmax (0.2, valmin)
                    valmax  = fmax (0.5, valmax) 
                self.para.space[0, i] = valmin
                self.para.space[1, i] = valmax
                self.para.space[2, i] = step
        self.para.isspace = 1
        return
    
    @cython.boundscheck(False)
    cdef void para2mod(self) nogil:
        """
        Convert paratemers (for perturbation) to model parameters
        """
        cdef Py_ssize_t i, ig, ip
        cdef float val 
        
        for i in range(self.para.npara):
            val = self.para.paraval[i]
            ig  = <int>self.para.paraindex[4, i]
            # velocity coeficient for splines
            if <int>self.para.paraindex[0, i] == 0:
                ip                  = <int>self.para.paraindex[5, i]
                self.cvel[ip][ig]   = val
            # total thickness of the group
            elif <int>self.para.paraindex[0, i] == 1:
                self.thickness[ig]  = val
            # vp/vs ratio
            elif <int>self.para.paraindex[0, i] == -1:
                self.vpvs[ig]       = val
            else:
                printf('Unexpected value in paraindex!')
        return
    
    @cython.boundscheck(False)
    cdef int isgood(self, int m0, int m1, int g0, int g1) nogil:
        """
        check the model is good or not
        ==========================================================================
        ::: input   :::
        m0, m1  - index of group for monotonic change checking
        g0, g1  - index of group for gradient change checking
        ==========================================================================
        """
        cdef Py_ssize_t i
        # velocity constrast, contraint (5) in 4.2 of Shen et al., 2012
        for i in range (self.nmod-1):
            if self.vs[0, i+1] < self.vs[-1, i]:
                return 0
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
                        return 0
        # gradient check
        if g0<=g1:
            for j in range(g0, g1+1):
                if self.vs[0, j] > self.vs[1, j]:
                    return 0
        return 1
    
    @cython.boundscheck(False)
    cdef Py_ssize_t get_vmodel(self, float[512] &vs, float[512] &vp, float[512] &rho,\
                    float[512] &qs, float[512] &qp, float[512] &hArr) nogil:
        """
        get velocity models
        ==========================================================================
        ::: output :::
        hArr, vs, vp, rho, qs, qp
        ==========================================================================
        """
#        cdef vector[float] vs, vp, rho, qs, qp, hArr
        cdef float depth   = 0.
        cdef Py_ssize_t i, j, ilay
        
        ilay  = 0
        for i in range(self.nmod):
            for j in range(self.nlay[i]):
                hArr[ilay]  = self.hArr[j][i]
                depth       += self.hArr[j][i]
                if self.mtype[i] == 5:
                    vs[ilay]    = 0
                    vp[ilay]    = self.cvel[0][i]
                    rho[ilay]   = 1.02
                    qs[ilay]    = 10000.
                    qp[ilay]    = 57822.
                elif (i == 0 and self.mtype[i] != 5) or (i == 1 and self.mtype[0] == 5):
                    vs[ilay]    = self.vs[j, i]
                    vp[ilay]    = self.vs[j, i]*self.vpvs[i]
                    rho[ilay]   = 0.541 + 0.3601*self.vs[j, i]*self.vpvs[i]
                    qs[ilay]    = 80.
                    qp[ilay]    = 160.
                else:
                    vs[ilay]    = self.vs[j, i]
                    vp[ilay]    = self.vs[j, i]*self.vpvs[i]
                    # if depth < 18.:
                    qs[ilay]    = 600.
                    qp[ilay]    = 1400.
                    if (self.vs[j, i]*self.vpvs[i]) < 7.5:
                        rho[ilay]       = 0.541 + 0.3601*self.vs[j, i]*self.vpvs[i]
                    else:
                        # Kaban, M. K et al. (2003), Density of the continental roots: Compositional and thermal contributions
                        rho[ilay]       = 3.35
                ilay += 1
        return ilay
    
    def get_vmodel_interface(self):
        cdef float[512] vs, vp, rho, qs, qp, hArr
        self.get_vmodel(vs, vp, rho, qs, qp, hArr)
        return vs, vp, rho, qs, qp, hArr
    
    cpdef copy(self):
        """
        return a copy of the object
        """
        cdef isomod outmod          = isomod()
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
        outmod.para     = self.para.copy()
        return outmod
    
    @cython.boundscheck(False)
    cdef void get_mod(self, isomod inmod) nogil:
        cdef Py_ssize_t i, j, k
        if inmod.nmod != self.nmod:
            with gil:
                self.init_arr(inmod.nmod)
        for i in range(self.nmod):
            self.numbp[i]       = inmod.numbp[i]
            self.mtype[i]       = inmod.mtype[i]
            self.thickness[i]   = inmod.thickness[i]
            self.nlay[i]        = inmod.nlay[i]
            self.vpvs[i]        = inmod.vpvs[i]
            self.isspl[i]       = inmod.isspl[i]
            for j in range(self.maxspl):
                self.cvel[j][i] = inmod.cvel[j][i]
                for k in range(self.maxlay):
                    self.spl[j][k][i] = inmod.spl[j][k][i]
            for k in range(self.maxlay):
                self.ratio[k][i]= inmod.ratio[k][i]
                self.vs[k][i]   = inmod.vs[k][i]
                self.hArr[k][i] = inmod.hArr[k][i]
        self.para.get_para(inmod.para)
        self.para.isspace = 1
        return
#                
#
    
    
    
    
    
    
    
    
    


    