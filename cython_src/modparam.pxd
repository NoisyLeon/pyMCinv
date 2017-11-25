
"""
Module for handling parameterization of the model

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

cdef float random_gauss(float mu, float sigma) nogil
cdef float random_uniform(float a, float b) nogil

cdef class para1d:
    cdef public:
        int npara, maxind
        int isspace
        float[:, :] paraindex, space
        float[:] paraval
    
    cpdef init_arr(self, int npara)
    cdef int new_paraval(self, int ptype) nogil
    cpdef copy(self)
    
cdef void bspl_basis(int nBs, int degBs, float zmin_Bs, float zmax_Bs, float disfacBs, int npts, 
        float[20][100] &nbasis) nogil
#
cdef class isomod:
    cdef int maxlay, maxspl
    cdef public:
        int nmod
        para1d para
        int[:] numbp, mtype, nlay, isspl
        float[:] thickness, vpvs
        float[:, :] cvel, ratio, vs, hArr
        float[:, :, :] spl
#    
    cpdef init_arr(self, nmod)
    cdef int bspline(self, Py_ssize_t i) nogil
    cdef int update(self) nogil
    cdef void get_paraind(self) nogil
    cdef void mod2para(self) nogil
    cdef void para2mod(self) nogil
    cdef int isgood(self, int m0, int m1, int g0, int g1) nogil
    cdef Py_ssize_t get_vmodel(self, float[512] &vs, float[512] &vp, float[512] &rho,\
                    float[512] &qs, float[512] &qp, float[512] &hArr) nogil
    cpdef copy(self)
#    
    
    
    
    
    
    
    
    
    
    


    