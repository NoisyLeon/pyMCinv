# -*- coding: utf-8 -*-
"""
Module for handling parameterization of the model

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""




# import random from c++ random.h
cdecdef extern from "<random>" namespace "std" nogil:
    cdef cppclass mt19937:
        mt19937()  # we need to define this constructor to stack allocate classes in Cython
        mt19937(unsigned int seed)  # not worrying about matching the exact int type for seed

    cdef cppclass uniform_real_distribution[T]:
        uniform_real_distribution() 
        uniform_real_distribution(T a, T b) 
        T operator()(mt19937 gen) nogil # ignore the possibility of using other classes for "gen"
        
    cdef cppclass normal_distribution[T]:
        normal_distribution() 
        normal_distribution(T mu, T sigma) 
        T operator()(mt19937 gen)  # ignore the possibility of using other classes for "gen"

# import vector from c++ vector.h
cdef extern from "<vector>" namespace "std" nogil:
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*() 
            iterator operator++() 
            bint operator==(iterator) 
            bint operator!=(iterator) 
        vector() 
        void push_back(T&) 
        T& operator[](int) 
        T& at(int) 
        iterator begin() 
        iterator end() 

ctypedef vector[vector[float]] FloatMatrix

@cython.boundscheck(False)
cdef float random_gauss(float mu, float sigma) nogil

@cython.boundscheck(False)
cdef float random_uniform(float a, float b) nogil


def test(int N)


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
    cdef public int npara, maxind
    cdef public bool isspace
    cdef public float[:, :] paraindex, space
    cdef public float[:] paraval
    
    def __init__(self)
    
    def init_arr(self, int npara)
    
    def readparatxt(self, str infname)
        
        
    def write_paraval_txt(self, str outfname)
    
    def read_paraval_txt(self, str infname)

    @cython.boundscheck(False)
    cdef bool new_paraval(self, int ptype) nogil
        
    def copy(self)
    
@cython.boundscheck(False)
cdef FloatMatrix bspl_basis(int nBs, int degBs, float zmin_Bs, float zmax_Bs, float disfacBs, int npts) nogil

def test_spl(int nBs, int degBs, float zmin_Bs, float zmax_Bs, float disfacBs, int npts)

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
    cdef public int nmod
    cdef int maxlay, maxspl
    cdef public para1d para
    cdef public int[:] numbp, mtype, nlay, isspl
    cdef public float[:] thickness, vpvs
    cdef public float[:, :] cvel, ratio, vs, hArr
    cdef public float[:, :, :] spl
    
    def __init__(self)
    
    def init_arr(self, nmod)
    
    def readmodtxt(self, str infname)
    
    @cython.boundscheck(False)
    cdef public bool bspline(self, int i) nogil
    
    @cython.boundscheck(False)
    cdef public bool update(self) nogil
    
    def update_interface(self)
    
    @cython.boundscheck(False)
    cdef public void get_paraind(self) nogil
    
    def get_paraind_interface(self)
    
    @cython.boundscheck(False)
    cdef public void mod2para(self) nogil
    
    @cython.boundscheck(False)
    cdef void para2mod(self) nogil
    
    @cython.boundscheck(False)
    cdef bool isgood(self, int m0, int m1, int g0, int g1) nogil
    
    @cython.boundscheck(False)
    cdef void get_vmodel(self, vector[float] &vs, vector[float] &vp, vector[float] &rho,\
                    vector[float] &qs, vector[float] &qp, vector[float] &hArr) nogil
    
    def get_vmodel_interface(self)
    
    def copy(self)
    
    
    
    
    
    
    
    
    
    


    