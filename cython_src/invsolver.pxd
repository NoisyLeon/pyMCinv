


cimport vmodel
cimport data
import numpy as np
cimport numpy as np
from libc.stdlib cimport malloc, free

cdef extern from './fast_surf_src/fast_surf.h':
    void fast_surf_(int *n_layer0,int *kind0,
     float *a_ref0, float *b_ref0, float *rho_ref0, float *d_ref0, float *qs_ref0,
     float *cvper, int *ncvper, float uR0[200], float uL0[200], float cR0[200], float cL0[200]) nogil
                    
cdef extern from './rftheo_src/rftheo.h':
    void theo_(int *n,  float fbeta[100], float h[100], float vps[100], float qa[100],
        float qb[100], float *fs, float *din, float *a0, float *c0, float *t0, int *nd, float rx[1024])


cdef class invsolver1d:
    cdef public:
        float [200]     TRpiso, TLpiso, TRgiso, TLgiso, CRiso, CLiso, URiso, ULiso
        float [2049]    TRptti, TLptti, TRgtti, TLgtti, CRtti, CLtti, URtti, ULtti
        vmodel.model1d  model
        data.data1d     data
    
    cdef int update_mod(self, int mtype) nogil
    cdef int get_vmodel(self, int mtype) nogil
    cdef void get_period(self) nogil
    
    cdef void compute_fsurf(self, int ilvry) nogil