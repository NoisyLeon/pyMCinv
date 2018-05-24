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
from libc.math cimport asin, pi, fmin, fmod
from libc.string cimport strcpy, strlen, strcat, strcmp
from libc.time cimport time,time_t
from libc.stdlib cimport rand, srand, RAND_MAX, malloc, free
import os
cimport modparam
from cython.parallel import parallel, prange
import multiprocessing
from functools import partial
#cdef extern from './fast_surf_src/fast_surf.h':
#    void fast_surf_(int *n_layer0,int *kind0,
#     float *a_ref0, float *b_ref0, float *rho_ref0, float *d_ref0, float *qs_ref0,
#     float *cvper, int *ncvper, float uR0[200], float uL0[200], float cR0[200], float cL0[200])
    

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
        self.newisomod  = modparam.isomod()
        self.oldisomod  = modparam.isomod()
        self.fs         = 40.
        self.slowness   = 0.06
        self.gausswidth = 2.5
        self.amplevel   = 0.005
        self.t0         = 0.
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
            self.data.rfr.tp    = np.linspace(self.data.rfr.to[0], self.data.rfr.to[-1], \
                        self.data.rfr.npts, dtype=np.float32)
            self.data.rfr.rfp   = np.zeros(self.data.rfr.npts, dtype=np.float32)
            self.npts           = self.data.rfr.npts
            self.fs             = 1./(self.data.rfr.to[1] - self.data.rfr.to[0])
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
    
    cdef int update_mod(self, int mtype=0) nogil:
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
    
    cdef int get_vmodel(self, int mtype=0) nogil:
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
    
    #----------------------------------------------------
    # forward modelling for surface waves
    #----------------------------------------------------
    
    @cython.boundscheck(False)
    cdef void get_period(self) nogil:
#    def get_period(self):
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
    
#        cdef void get_period(self) nogil:
    def get_period_test(self):
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
    cdef void compute_fsurf(self, int ilvry=2) nogil:
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
        
    #----------------------------------------------------
    # forward modelling for receiver function
    #----------------------------------------------------
        
    cdef void compute_rftheo(self) nogil:
        """
        compute receiver function of isotropic model using theo
        =======================================================================
        ::: input :::
        =======================================================================
        """
        cdef float[1024] rx
        cdef int nlay       = self.model.nlay
        cdef float fs       = self.fs
        cdef float slowness = self.slowness
        cdef float din      = 180.*asin(self.model.vpv[nlay-1]*slowness)/pi
        cdef float[100] beta, h, vps, qa, qb
        cdef Py_ssize_t i
        cdef float a0       = self.gausswidth
        cdef float c0       = self.amplevel
        cdef float t0       = self.t0
        cdef int npts       = self.npts
        
        nlay                = int(fmin(100, nlay))
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
        theo_(&nlay,  &beta[0], &h[0], &vps[0], &qa[0], &qb[0], &fs, &din, &a0, &c0, &t0, &npts, <float*>rx)
        for i in range(self.npts):
            self.data.rfr.rfp[i]    = rx[i]
        return 
    
    def compute_rftheo_interface(self):
        self.compute_rftheo()
        return
    
    cdef void get_misfit(self, float wdisp=1., float rffactor=40.) nogil:
        """
        compute data misfit
        =====================================================================
        ::: input :::
        wdisp       - weight for dispersion curves (0.~1., default - 1.)
        rffactor    - downweighting factor for receiver function
        =====================================================================
        """
        self.data.get_misfit(wdisp, rffactor)
        return
    
    #-------------------------------------------------
    # functions for inversions
    #-------------------------------------------------
    
    def mc_inv_iso_singel_thread(self, str outdir, int ind0=0, int ind1=2000, int indid=1, str pfx='MC', str dispdtype='ph', \
                    float wdisp=1., float rffactor=40., int monoc=1):
        
        cdef str outmod, outdisp, outrf
        cdef float oldL, oldmisfit, newL, newmisfit, prob, rnumb
        cdef int run = 1
        cdef Py_ssize_t inew, iacc, igood, i
        cdef int Maxiter = ind1 - ind0
        cdef float[:, :] outArr, dispArr, rfArr
#
#        outArr      = np.zeros([10+self.model.isomod.para.npara, Maxiter], dtype=np.float32)
#        dispArr     = np.zeros([self.data.dispR.npper, Maxiter], dtype=np.float32)
#        rfArr       = np.zeros([self.data.rfr.npts, Maxiter], dtype=np.float32)
        # initializations
        self.get_period()
##        if ind0 != 0:
##            self.newisomod.get_mod(self.model.isomod)
##            self.newisomod.get_mod(self.model.isomod)
##            self.newisomod.para.new_paraval(0)
##            self.newisomod.para2mod()
##            self.newisomod.update()
##            igood   = 0
##            while ( self.newisomod.isgood(0, 1, 1, 0) == 0 and igood<100):
##                igood   += igood + 1
##                self.newisomod.get_mod(self.model.isomod)
##                self.newisomod.para.new_paraval(0)
##                self.newisomod.para2mod()
##                self.newisomod.update()
##            # assign new model to old ones
##            self.model.isomod.get_mod(self.newisomod)
##            self.get_vmodel()
#        self.update_mod(0)
#        self.get_vmodel(0)
#        # initial run
#        self.compute_fsurf()
#        self.compute_rftheo()
#        self.get_misfit(wdisp, rffactor)
#        # write initial model
#        outmod      = outdir+'/'+pfx+'.'+str(indid)+'.mod'
#        self.model.write_model(outfname=outmod, isotropic=1)
#        # write initial predicted data
#        outdisp     = outdir+'/'+pfx+'.'+str(indid)+'.ph.disp'
#        self.data.dispR.writedisptxt(outfname=outdisp, dtype='ph')
#        dispArr[:,0]= self.data.dispR.pvelp
#        outrf       = outdir+'/'+pfx+'.'+str(indid)+'.rf'
#        self.data.rfr.writerftxt(outfname=outrf)
#        rfArr[:,0]  = self.data.rfr.rfp
#        # convert initial model to para
#        self.model.isomod.mod2para()
#        # likelihood/misfit
#        oldL        = self.data.L
#        oldmisfit   = self.data.misfit
#        printf('Initial likelihood = %f,' , oldL)
#        printf('misfit = %f,', oldmisfit)
#        printf(' index id = %d\n', indid)
#        inew        = ind0 +1
#        iacc        = 1     # count acceptance model
#        cdef time_t t0 = time(NULL)
#        cdef time_t t1 
#        self.newisomod.get_mod(self.model.isomod)
#        while ( run==1 ):
#            inew    += 1
##            printf('run step = %d\n',inew)
#            t1      = time(NULL)
#            # # # if ( inew > 100000 or iacc > 20000000 or time.time()-start > 7200.):
#            if ( inew > ind1 or iacc > 20000000):
#                run = 0
#            if (fmod(inew, 500) == 0):
#                printf('Index id = %d, ',indid)
#                printf('run step = %d,',inew-ind0)
#                printf(' elasped time = %d', t1-t0)
#                printf(' sec\n')
#            #-------------------------------
#            # inversion part
#            #-------------------------------
#            # sample the posterior distribution ##########################################
#            if (wdisp >= 0 and wdisp <=1):
#                self.newisomod.get_mod(self.model.isomod)
#                self.newisomod.para.new_paraval(1)
#                self.newisomod.para2mod()
#                self.newisomod.update()
#                if monoc:
#                    # loop to find the "good" model,
#                    # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
#                    if not self.newisomod.isgood(0, 1, 1, 0):
#                        continue
#                # assign new model to old ones
#                self.oldisomod.get_mod(self.model.isomod)
#                self.model.isomod.get_mod(self.newisomod)
#                self.get_vmodel()
#                # forward computation
#                self.compute_fsurf()
#                self.compute_rftheo()
#                self.get_misfit(wdisp, rffactor)
#                newL                = self.data.L
#                newmisfit           = self.data.misfit
#                # 
#                if newmisfit > oldmisfit:
#                    rnumb   = random_uniform(0., 1.)
#                    if oldL == 0.:
#                        continue                        
#                    prob    = (oldL-newL)/oldL
#                    # reject the model
#                    if rnumb < prob:       
#                        outArr[0][inew-ind0-1]      = -1.
#                        outArr[1][inew-ind0-1]      = inew
#                        outArr[2][inew-ind0-1]      = iacc
#                        for i in range(self.newisomod.para.npara):
#                            outArr[3+i][inew-ind0-1] = self.newisomod.para.paraval[i]
#                        outArr[16][inew-ind0-1]     = newL
#                        outArr[17][inew-ind0-1]     = newmisfit
#                        outArr[18][inew-ind0-1]     = self.data.rfr.L
#                        outArr[19][inew-ind0-1]     = self.data.rfr.misfit
#                        outArr[20][inew-ind0-1]     = self.data.dispR.L
#                        outArr[21][inew-ind0-1]     = self.data.dispR.misfit
#                        outArr[22][inew-ind0-1]     = time(NULL)-t0
#                        
#                        self.model.isomod.get_mod(self.oldisomod)
#                        continue
#                # accept the new model
##                fidout.write("1 %d %d " % (inew,iacc))
##                for i in xrange(newmod.para.npara):
##                    fidout.write("%g " % newmod.para.paraval[i])
##                fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.rfr.L, self.indata.rfr.misfit,\
##                        self.indata.dispR.L, self.indata.dispR.misfit, time.time()-start)) 
##                
#                outArr[0][inew-ind0-1]      = -1.
#                outArr[1][inew-ind0-1]      = inew
#                outArr[2][inew-ind0-1]      = iacc
#                for i in range(self.newisomod.para.npara):
#                    outArr[3+i][inew-1]     = self.newisomod.para.paraval[i]
#                outArr[16][inew-ind0-1]     = newL
#                outArr[17][inew-ind0-1]     = newmisfit
#                outArr[18][inew-ind0-1]     = self.data.rfr.L
#                outArr[19][inew-ind0-1]     = self.data.rfr.misfit
#                outArr[20][inew-ind0-1]     = self.data.dispR.L
#                outArr[21][inew-ind0-1]     = self.data.dispR.misfit
#                outArr[22][inew-ind0-1]     = time(NULL)-t0
#                
#                printf('Index id = %d', indid)
#                printf(' ,accept a model: %d', inew)
#                printf(' %d ', iacc)
#                printf(' %f ', oldL)
#                printf(' %f ', oldmisfit)
#                printf(' %f ', newL)
#                printf(' %f ', newmisfit)
#                printf(' %f ', self.data.rfr.L)
#                printf(' %f ', self.data.rfr.misfit)
#                printf(' %f ', self.data.dispR.L)
#                printf(' %f ', self.data.dispR.misfit)
#                printf(' %f\n', time(NULL)-t0)
#                # write accepted model
##                outmod      = outdir+'/'+pfx+'.%d.mod' % iacc
##                vmodel.write_model(model=self.model, outfname=outmod, isotropic=True)
##                # write corresponding data
##                if dispdtype != 'both':
##                    outdisp = outdir+'/'+pfx+'.'+dispdtype+'.%d.disp' % iacc
##                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
##                else:
##                    outdisp = outdir+'/'+pfx+'.ph.%d.disp' % iacc
##                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='ph')
##                    outdisp = outdir+'/'+pfx+'.gr.%d.disp' % iacc
##                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='gr')
##                # # outdisp = outdir+'/'+pfx+'.%d.disp' % iacc
##                # # data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
##                outrf   = outdir+'/'+pfx+'.%d.rf' % iacc
##                data.writerftxt(outfname=outrf, outrf=self.indata.rfr)
#                # assign likelihood/misfit
#                oldL        = newL
#                oldmisfit   = newmisfit
#                iacc        += 1
#                continue
##            else:
##                if monoc:
##                    newmod  = self.model.isomod.copy()
##                    newmod.para.new_paraval(1)
##                    newmod.para2mod()
##                    newmod.update()
##                    if not newmod.isgood(0, 1, 1, 0):
##                        continue
##                else:
##                    newmod  = self.model.isomod.copy()
##                    newmod.para.new_paraval(0)
##                fidout.write("-2 %d 0 " % inew)
##                for i in xrange(newmod.para.npara):
##                    fidout.write("%g " % newmod.para.paraval[i])
##                fidout.write("\n")
##                self.model.isomod   = newmod
##                continue
        return
    
    def mc_inv_iso_mp(self, str outdir, int maxstep=10000, int maxsubstep=2000, str pfx='MC', str dispdtype='ph', \
                    float wdisp=1., float rffactor=40., int monoc=1, int nprocess=2):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        indLst      = []
        cdef Py_ssize_t i
        cdef np.ndarray indArr = np.zeros(3, np.int32)
        
        maxstep     = (np.ceil(maxstep/maxsubstep))*maxsubstep
        for i in range(int(maxstep/maxsubstep)):
            indArr[0]   = i*maxsubstep
            indArr[1]   = (i+1)*maxsubstep
            indArr[2]   = i+1
            indLst.append(indArr.copy())
        print 'Start MC inverion for isotropic model, multiprocessing'
        MCINV = partial(mcinviso4mp, solver=self, outdir=outdir, pfx=pfx, dispdtype=dispdtype,\
                    wdisp=wdisp, rffactor=rffactor, monoc=monoc)
        pool = multiprocessing.Pool(processes=nprocess)
        pool.map(MCINV, indLst) 
        pool.close() 
        pool.join() 
        
        
        
        
        
        
        

    
    @cython.boundscheck(False)
    cdef void mc_inv_iso(self, char *outdir, char *pfx, char *dispdtype, \
                    float wdisp=1., float rffactor=40., int monoc=1) nogil:
        """
        
        """
        cdef char *outmod, *outdisp, *outrf
        cdef float oldL, oldmisfit, newL, newmisfit, prob, rnumb
        cdef int run = 1
        cdef Py_ssize_t inew, iacc, igood, i
        cdef float[23][100000] outArr
#        with gil:
#            def modparam.isomod newmod 
#        with gil:
#            newmod= modparam.isomod()
#        cdef 
        # initializations
        self.get_period()
        self.update_mod(0)
        self.get_vmodel(0)
        # initial run
        self.compute_fsurf()
        self.compute_rftheo()
        self.get_misfit(wdisp, rffactor)
        # write initial model
        outmod = <char *>malloc((strlen(outdir)+1+strlen(pfx)+4) * sizeof(char))
        strcpy(outmod, outdir)
        strcat(outmod, '/')
        strcat(outmod, pfx)
        strcat(outmod, '.mod')
        with gil:
            self.model.write_model(outfname=outmod, isotropic=1)
        free(outmod)
        # write initial predicted data
        with gil:
            if strcmp(dispdtype, 'both') != 0:
                outdisp = <char *>malloc((strlen(outdir)+1+strlen(pfx)+6+strlen(dispdtype)) * sizeof(char))
                strcpy(outdisp, outdir)
                strcat(outdisp, '/')
                strcat(outdisp, pfx)
                strcat(outdisp, '.')
                strcat(outdisp, dispdtype)
                strcat(outdisp, '.disp')
                self.data.dispR.writedisptxt(outfname=outdisp, dtype=dispdtype)
                free(outdisp)
            else:
                outdisp = <char *>malloc((strlen(outdir)+1+strlen(pfx)+6+strlen(dispdtype)) * sizeof(char))
                strcpy(outdisp, outdir)
                strcat(outdisp, '/')
                strcat(outdisp, pfx)
                strcat(outdisp, '.ph.disp')
                self.data.dispR.writedisptxt(outfname=outdisp, dtype='ph')
                free(outdisp)
                outdisp = <char *>malloc((strlen(outdir)+1+strlen(pfx)+6+strlen(dispdtype)) * sizeof(char))
                strcpy(outdisp, outdir)
                strcat(outdisp, '/')
                strcat(outdisp, pfx)
                strcat(outdisp, '.gr.disp')
                self.data.dispR.writedisptxt(outfname=outdisp, dtype='gr')
                free(outdisp)
        with gil:
            outrf = <char *>malloc((strlen(outdir)+1+strlen(pfx)+3) * sizeof(char))
            strcpy(outrf, outdir)
            strcat(outrf, '/')
            strcat(outrf, pfx)
            strcat(outrf, '.rf')
            self.data.rfr.writerftxt(outfname=outrf)
            free(outrf)
        # convert initial model to para
        self.model.isomod.mod2para()
        # likelihood/misfit
        oldL        = self.data.L
        oldmisfit   = self.data.misfit
        printf('Initial likelihood = %f,' , oldL)
        printf('misfit = %f\n', oldmisfit)
        
        inew    = 0     # count step (or new paras)
        iacc    = 1     # count acceptance model
        cdef time_t t0 = time(NULL)
        cdef time_t t1 
        self.newisomod.get_mod(self.model.isomod)
#        newmod.get_mod(self.model.isomod)
        while ( run==1 ):
            inew    += 1
#            printf('run step = %d\n',inew)
            t1      = time(NULL)
            # # # if ( inew > 100000 or iacc > 20000000 or time.time()-start > 7200.):
            if ( inew > 10000 or iacc > 20000000):
                run = 0
            if (fmod(inew, 500) == 0):
                printf('run step = %d,',inew)
                printf(' elasped time = %d', t1-t0)
                printf(' sec\n')
            #------------------------------------------------------------------------------------------
            # every 2500 step, perform a random walk with uniform random value in the paramerter space
            #------------------------------------------------------------------------------------------
            if ( fmod(inew, 1501) == 1500 ):
                self.newisomod.get_mod(self.model.isomod)
                self.newisomod.para.new_paraval(0)
                self.newisomod.para2mod()
                self.newisomod.update()
                # loop to find the "good" model,
                # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                igood   = 0
                while ( self.newisomod.isgood(0, 1, 1, 0) == 0):
                    igood   += igood + 1
                    self.newisomod.get_mod(self.model.isomod)
                    self.newisomod.para.new_paraval(0)
                    self.newisomod.para2mod()
                    self.newisomod.update()
                # assign new model to old ones
                self.model.isomod.get_mod(self.newisomod)
                self.get_vmodel()
                # forward computation
                self.compute_fsurf()
                self.compute_rftheo()
                self.get_misfit(wdisp, rffactor)
                oldL                = self.data.L
                oldmisfit           = self.data.misfit
                iacc                += 1
                printf('Uniform random walk: likelihood = %f', self.data.L)
                printf(' misfit = %f\n', self.data.misfit)
            #-------------------------------
            # inversion part
            #-------------------------------
            # sample the posterior distribution ##########################################
            if (wdisp >= 0 and wdisp <=1):
                self.newisomod.get_mod(self.model.isomod)
                self.newisomod.para.new_paraval(1)
                self.newisomod.para2mod()
                self.newisomod.update()
                if monoc:
                    # loop to find the "good" model,
                    # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                    if not self.newisomod.isgood(0, 1, 1, 0):
                        continue
                # assign new model to old ones
                self.oldisomod.get_mod(self.model.isomod)
                self.model.isomod.get_mod(self.newisomod)
                self.get_vmodel()
                # forward computation
                self.compute_fsurf()
                self.compute_rftheo()
                self.get_misfit(wdisp, rffactor)
                newL                = self.data.L
                newmisfit           = self.data.misfit
                # 
                if newmisfit > oldmisfit:
                    rnumb   = random_uniform(0., 1.)
                    if oldL == 0.:
                        continue                        
                    prob    = (oldL-newL)/oldL
                    # reject the model
                    if rnumb < prob:
#                        fidout.write("-1 %d %d " % (inew,iacc))
#                        for i in xrange(newmod.para.npara):
#                            fidout.write("%g " % newmod.para.paraval[i])
#                        fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.rfr.L, self.indata.rfr.misfit,\
#                                self.indata.dispR.L, self.indata.dispR.misfit, time.time()-start)) 
#                        
                        outArr[0][inew-1]   = -1.
                        outArr[1][inew-1]   = inew
                        outArr[2][inew-1]   = iacc
                        for i in range(13):
                            outArr[3+i][inew-1] = self.newisomod.para.paraval[i]
                        outArr[16][inew-1]      = newL
                        outArr[17][inew-1]      = newmisfit
                        outArr[18][inew-1]      = self.data.rfr.L
                        outArr[19][inew-1]      = self.data.rfr.misfit
                        outArr[20][inew-1]      = self.data.dispR.L
                        outArr[21][inew-1]      = self.data.dispR.misfit
                        outArr[22][inew-1]      = time(NULL)-t0
                        
                        self.model.isomod.get_mod(self.oldisomod)
                        continue
                # accept the new model
#                fidout.write("1 %d %d " % (inew,iacc))
#                for i in xrange(newmod.para.npara):
#                    fidout.write("%g " % newmod.para.paraval[i])
#                fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.rfr.L, self.indata.rfr.misfit,\
#                        self.indata.dispR.L, self.indata.dispR.misfit, time.time()-start)) 
#                
                outArr[0][inew-1]   = -1.
                outArr[1][inew-1]   = inew
                outArr[2][inew-1]   = iacc
                for i in range(13):
                    outArr[3+i][inew-1] = self.newisomod.para.paraval[i]
                outArr[16][inew-1]      = newL
                outArr[17][inew-1]      = newmisfit
                outArr[18][inew-1]      = self.data.rfr.L
                outArr[19][inew-1]      = self.data.rfr.misfit
                outArr[20][inew-1]      = self.data.dispR.L
                outArr[21][inew-1]      = self.data.dispR.misfit
                outArr[22][inew-1]      = time(NULL)-t0
                
                printf('Accept a model: %d', inew)
                printf(' %d ', iacc)
                printf(' %f ', oldL)
                printf(' %f ', oldmisfit)
                printf(' %f ', newL)
                printf(' %f ', newmisfit)
                printf(' %f ', self.data.rfr.L)
                printf(' %f ', self.data.rfr.misfit)
                printf(' %f ', self.data.dispR.L)
                printf(' %f ', self.data.dispR.misfit)
                printf(' %f\n', time(NULL)-t0)
                # write accepted model
#                outmod      = outdir+'/'+pfx+'.%d.mod' % iacc
#                vmodel.write_model(model=self.model, outfname=outmod, isotropic=True)
#                # write corresponding data
#                if dispdtype != 'both':
#                    outdisp = outdir+'/'+pfx+'.'+dispdtype+'.%d.disp' % iacc
#                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
#                else:
#                    outdisp = outdir+'/'+pfx+'.ph.%d.disp' % iacc
#                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='ph')
#                    outdisp = outdir+'/'+pfx+'.gr.%d.disp' % iacc
#                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='gr')
#                # # outdisp = outdir+'/'+pfx+'.%d.disp' % iacc
#                # # data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
#                outrf   = outdir+'/'+pfx+'.%d.rf' % iacc
#                data.writerftxt(outfname=outrf, outrf=self.indata.rfr)
                # assign likelihood/misfit
                oldL        = newL
                oldmisfit   = newmisfit
                iacc        += 1
                continue
#            else:
#                if monoc:
#                    newmod  = self.model.isomod.copy()
#                    newmod.para.new_paraval(1)
#                    newmod.para2mod()
#                    newmod.update()
#                    if not newmod.isgood(0, 1, 1, 0):
#                        continue
#                else:
#                    newmod  = self.model.isomod.copy()
#                    newmod.para.new_paraval(0)
#                fidout.write("-2 %d 0 " % inew)
#                for i in xrange(newmod.para.npara):
#                    fidout.write("%g " % newmod.para.paraval[i])
#                fidout.write("\n")
#                self.model.isomod   = newmod
#                continue
        return
    
    def mc_inv_iso_interface(self, char *outdir='./workingdir_iso', char *pfx='MC',\
                char *dispdtype='ph', float wdisp=0.2, float rffactor=40., int monoc=1):
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
#        cdef char *outdir
#        cdef char *pfx
#        cdef int l = strlen(pfx) 
#        printf('%s', pfx)
#        printf('%d', l)
#        outdir = './work'
#        pfx  = 'MC'
        self.mc_inv_iso(outdir, pfx, dispdtype)
#        self.mc_inv_iso()


def mcinviso4mp(iArr, invsolver1d solver, str outdir, str pfx, str dispdtype, \
                    float wdisp, float rffactor, int monoc):
    print iArr[2], iArr[1]
    solver.mc_inv_iso_singel_thread(outdir=outdir, ind0=iArr[0], ind1=iArr[1], indid=iArr[2], pfx=pfx, dispdtype=dispdtype, wdisp=wdisp,\
                                    rffactor=rffactor, monoc=monoc)
    return
    

    
    
        
    
    
    
    
    

    
    


    