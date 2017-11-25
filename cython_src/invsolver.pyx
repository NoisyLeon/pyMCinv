# -*- coding: utf-8 -*-
"""
Module for inversion of 1d models

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""


from libc.stdio cimport printf
import numpy as np
cimport numpy as np
cimport cython
from libc.math cimport asin, pi, fmin

#cdef extern from './fast_surf_src/fast_surf.h':
#    void fast_surf_(int *n_layer0,int *kind0,
#     float *a_ref0, float *b_ref0, float *rho_ref0, float *d_ref0, float *qs_ref0,
#     float *cvper, int *ncvper, float uR0[200], float uL0[200], float cR0[200], float cL0[200])
    

#cdef 
def fast_surf():
    cdef Py_ssize_t i
    cdef int nlay=10
    cdef int kind0 = 1
#    cdef float[:] a, b, rho, d, qs
    cdef int nper = 10
    cdef float[200] ur, ul, cr, cl, per
    
    cdef float *a = <float *>malloc(nlay * sizeof(float))
    cdef float *b = <float *>malloc(nlay * sizeof(float))
    cdef float *rho = <float *>malloc(nlay * sizeof(float))
    cdef float *d = <float *>malloc(nlay * sizeof(float))
    cdef float *qs = <float *>malloc(nlay * sizeof(float))
    
#    a = np.ones(10, dtype=np.float32)*5.
#    b = np.ones(10, dtype=np.float32)*3.
#    rho = np.ones(10, dtype=np.float32)*2.7
#    d= np.ones(10, dtype=np.float32)
#    qs = 1./np.ones(10, dtype=np.float32)/600.

    for i in range(nlay):
        a[i] = 5.
        b[i] = 3.
        rho[i] = 2.7
        d[i] = 5.
        qs[i] = 1./600.
    
    
#    per= np.zeros(200, dtype=np.float32)
    for i in range(10):
        per[i] = float(i+1)
    
    
#    ur = np.zeros(200, dtype=np.float32)
#    ul = np.zeros(200, dtype=np.float32)
#    cr = np.zeros(200, dtype=np.float32)
#    cl = np.zeros(200, dtype=np.float32)
    fast_surf_(&nlay, &kind0, &a[0], &b[0], &rho[0], & d[0],\
               &qs[0], &per[0], &nper, <float*> ur, <float*> ul, <float*> cr, <float*>cl)
#    fast_surf_(&nlay, &kind0, &a[0], &b[0], &rho[0], & d[0],\
#               &qs[0], &per[0], &nper, & ur, & ul, & cr, & cl)
    
    free(a)
    free(b)
    free(rho)
    free(d)
    free(qs)
    return ur, ul, cr, cl


cdef class invsolver1d:
    
    
    
    def __init__(self):
        self.model      = vmodel.model1d()
        self.data       = data.data1d()
        return
    
    def readdisp(self, str infname, str dtype='ph', str wtype='ray'):
        """
        read dispersion curve data from a txt file
        ===========================================================
        ::: input :::
        infname     - input file name
        dtype       - data type (phase or group)
        wtype       - wave type (Rayleigh or Love)
        ===========================================================
        """
        dtype   = dtype.lower()
        wtype   = wtype.lower()
        if wtype=='ray' or wtype=='rayleigh' or wtype=='r':
            self.data.dispR.readdisptxt(infname=infname, dtype=dtype)
            if self.data.dispR.npper>0:
                self.data.dispR.pvelp = np.zeros(self.data.dispR.npper, dtype=np.float32)
                self.data.dispR.gvelp = np.zeros(self.data.dispR.npper, dtype=np.float32)
#            if self.data.dispR.ngper>0:
#                self.data.dispR.gvelp = np.zeros(self.data.dispR.ngper, dtype=np.float32)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            self.data.dispL.readdisptxt(infname=infname, dtype=dtype)
            if self.data.dispL.npper>0:
                self.data.dispL.pvelp = np.zeros(self.data.dispL.npper, dtype=np.float32)
                self.data.dispL.gvelp = np.zeros(self.data.dispL.npper, dtype=np.float32)
#            if self.data.dispL.ngper>0:
#                self.data.dispL.gvelp = np.zeros(self.data.dispL.ngper, dtype=np.float32)
        else:
            raise ValueError('Unexpected wave type: '+wtype)
        return


    def readrf(self, str infname, str dtype='r'):
        """
        read receiver function data from a txt file
        ===========================================================
        ::: input :::
        infname     - input file name
        dtype       - data type (radial or trnasverse)
        ===========================================================
        """
        dtype=dtype.lower()
        if dtype=='r' or dtype == 'radial':
            self.data.rfr.readrftxt(infname)
        elif dtype=='t' or dtype == 'transverse':
            self.data.rft.readrftxt(infname)
        else:
            raise ValueError('Unexpected wave type: '+dtype)
        return
    
    def readmod(self, str infname, str mtype='iso'):
        """
        read model from a txt file
        ===========================================================
        ::: input :::
        infname     - input file name
        mtype       - model type (isotropic or tti)
        ===========================================================
        """
        mtype   = mtype.lower()
        if mtype == 'iso' or mtype == 'isotropic':
            self.model.isomod.readmodtxt(infname)
        elif mtype == 'tti':
            self.model.isomod.readtimodtxt(infname)
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def readpara(self, str infname, str mtype='iso'):
        """
        read parameter index indicating model parameters for perturbation
        =====================================================================
        ::: input :::
        infname     - input file name
        mtype       - model type (isotropic or tti)
        =====================================================================
        """
        mtype   = mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            self.model.isomod.para.readparatxt(infname)
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def getpara(self, str mtype='iso'):
        """
        get parameter index indicating model parameters for perturbation
        =====================================================================
        ::: input :::
        mtype       - model type (isotropic or tti)
        =====================================================================
        """
        mtype   = mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            self.model.isomod.get_paraind()
#        elif mtype=='tti':
#            self.model.ttimod.get_paraind()
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    cdef int update_mod(self, int mtype) nogil:
        """
        update model from model parameters
        =====================================================================
        ::: input :::
        mtype       - model type (0 - isotropic or 1 - tti)
        =====================================================================
        """
        if mtype==0:
            self.model.isomod.update()
#        elif mtype==1:
#            self.model.ttimod.update()
        else:
            printf('Unexpected wave type: %d', mtype)
            return 0
        return 1
    
    def update_mod_interface(self, str mtype='iso'):
        """
        update model from model parameters
        =====================================================================
        ::: input :::
        mtype       - model type (isotropic or tti)
        =====================================================================
        """
        mtype   = mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            self.model.isomod.update()
#        elif mtype=='tti':
#            self.model.ttimod.update()
        else:
            raise ValueError ('Unexpected wave type: '+ mtype)
        return
    
    cdef int get_vmodel(self, int mtype) nogil:
        """
        get the velocity model arrays
        =====================================================================
        ::: input :::
        mtype       - model type (0 - isotropic or 1 - tti)
        =====================================================================
        """
        if mtype==0:
            self.model.get_iso_vmodel()
#        elif mtype==1:
#            self.qsArr, self.qpArr  = self.model.get_tti_vmodel() # get the model arrays and initialize elastic tensor
#            self.model.rot_dip_strike() 
#            self.model.decompose()
#        else:
#            printf('Unexpected wave type: %d', mtype)
#            return 0
        return 1
    
    def get_vmodel_interface(self, mtype='iso'):
        """
        get the velocity model arrays
        =====================================================================
        ::: input :::
        mtype       - model type (isotropic or tti)
        =====================================================================
        """
        mtype   = mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            self.model.get_iso_vmodel()
#        elif mtype == 'tti':
#            self.qsArr, self.qpArr  = self.model.get_tti_vmodel() # get the model arrays and initialize elastic tensor
#            self.model.rot_dip_strike() 
#            self.model.decompose()
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    @cython.boundscheck(False)
    cdef void get_period(self) nogil:
        cdef Py_ssize_t i
        if self.data.dispR.npper>0:
            for i in range(self.data.dispR.npper):
                self.TRpiso[i]  = self.data.dispR.pper[i]
        if self.data.dispR.ngper>0:
            for i in range(self.data.dispR.ngper):
                self.TRgiso[i]  = self.data.dispR.gper[i]
        if self.data.dispL.npper>0:
            for i in range(self.data.dispL.npper):
                self.TLpiso[i]  = self.data.dispL.pper[i]
        if self.data.dispL.ngper>0:
            for i in range(self.data.dispL.ngper):
                self.TLgiso[i]  = self.data.dispL.gper[i]
        return
    
    def get_period_interface(self):
        self.get_period()
        return
    
    @cython.boundscheck(False)
    cdef void compute_fsurf(self, int ilvry) nogil:
        """
        compute surface wave dispersion of isotropic model using fast_surf
        =====================================================================
        ::: input :::
        ilvry       - wave type ( 1 - Love, 2 - Rayleigh )
        =====================================================================
        """
        cdef float[200] ur, ul, cr, cl
        cdef int nlay = self.model.nlay
        cdef int nper
        cdef Py_ssize_t i
        cdef float *qsinv = <float *>malloc(nlay * sizeof(float))
    
        for i in range(nlay):
            qsinv[i]    = 1./self.model.qs[i]
        if ilvry == 2:
            ilvry               = 2
            nper                = self.data.dispR.npper
            fast_surf_(&nlay, &ilvry, &self.model.vpv[0], &self.model.vsv[0], &self.model.rho[0], &self.model.h[0],\
                       &qsinv[0], &self.TRpiso[0], &nper, <float*> ur, <float*> ul, <float*> cr, <float*>cl)
            for i in range(nper):
                self.data.dispR.pvelp[i]    = cr[i]
                self.data.dispR.gvelp[i]    = ur[i]

        elif ilvry == 1:
            nper                = self.data.dispL.npper
            fast_surf_(&nlay, &ilvry, &self.model.vpv[0], &self.model.vsv[0], &self.model.rho[0], &self.model.h[0],\
                       &qsinv[0], &self.TRpiso[0], &nper, <float*> ur, <float*> ul, <float*> cr, <float*>cl)
            for i in range(nper):
                self.data.dispL.pvelp[i]    = cl[i]
                self.data.dispL.gvelp[i]    = ul[i]
        return
    
    def compute_fsurf_interface(self, int ilvry=2):
        self.compute_fsurf(ilvry)
        
        
    def compute_rftheo(self):
        """
        compute receiver function of isotropic model using theo
        ===========================================================================================
        ::: input :::
        dtype   - data type (radial or trnasverse)
        slowness- reference horizontal slowness (default - 0.06 s/km, 1./0.06=16.6667)
        din     - incident angle in degree (default - None, din will be computed from slowness)
        ===========================================================================================
        """
        cdef float[1024] rx
        cdef int nlay = self.model.nlay
        cdef float fs = 40.
        cdef float slowness = 0.06
        cdef float din     = 180.*asin(self.model.vpv[nlay-1]*slowness)/pi
        cdef float[100] beta, h, vps, qa, qb
        cdef Py_ssize_t i
        cdef float a0 = 2.5
        cdef float c0 = 0.005
        cdef float t0 = 0.
        cdef int npts = 600
        
        nlay        = int(fmin(100, nlay))
        for i in range(100):
            if i+1 > nlay:
                beta[i]     = 0.
                h[i]        = 0.
                vps[i]      = 0.
                qa[i]       = 0.
                qb[i]       = 0.
            else:
                beta[i]     = self.model.vsv[i]
                h[i]        = self.model.h[i]
                vps[i]      = self.model.vpv[i]/self.model.vsv[i]
                qa[i]       = self.model.qp[i]
                qb[i]       = self.model.qs[i]
            
        
#        fast_surf_(&nlay, &ilvry, &self.model.vpv[0], &self.model.vsv[0], &self.model.rho[0], &self.model.h[0],\
#                       &qsinv[0], &self.TRpiso[0], &nper, <float*> ur, <float*> ul, <float*> cr, <float*>cl)
        print din
        
        
        theo_(&nlay,  &beta[0], &h[0], &vps[0], &qa[0], &qb[0], &fs, &din, &a0, &c0, &t0, &npts, <float*>rx)
#        
#        dtype   = dtype.lower()
#        if dtype=='r' or dtype == 'radial':
#            # initialize input model arrays
#            hin         = np.zeros(100, dtype=np.float32)
#            vsin        = np.zeros(100, dtype=np.float32)
#            vpvs        = np.zeros(100, dtype=np.float32)
#            qsin        = 600.*np.ones(100, dtype=np.float32)
#            qpin        = 1400.*np.ones(100, dtype=np.float32)
#            # assign model arrays to the input arrays
#            if self.hArr.size<100:
#                nl      = self.hArr.size
#            else:
#                nl      = 100
#            hin[:nl]    = self.hArr
#            vsin[:nl]   = self.vsArr
#            vpvs[:nl]   = self.vpvsArr
#            qsin[:nl]   = self.qsArr
#            qpin[:nl]   = self.qpArr
#            # fs/npts
#            fs          = self.fs
#            # # # ntimes      = 1000
#            ntimes      = self.npts
#            # incident angle
#            if din is None:
#                din     = 180.*np.arcsin(vsin[nl-1]*vpvs[nl-1]*slowness)/np.pi
#            # solve for receiver function using theo
#            rx 	                = theo.theo(nl, vsin, hin, vpvs, qpin, qsin, fs, din, 2.5, 0.005, 0, ntimes)
#            # store the predicted receiver function (ONLY radial component) to the data object
#            self.indata.rfr.rfp = rx[:self.npts]
#            self.indata.rfr.tp  = np.arange(self.npts, dtype=np.float32)*1./self.fs
#        # elif dtype=='t' or dtype == 'transverse':
#        #     
#        else:
#            raise ValueError('Unexpected receiver function type: '+dtype)
        return rx
    
    
    
    
    
    

    
    


    