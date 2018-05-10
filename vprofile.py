# -*- coding: utf-8 -*-
"""
Module for inversion of 1d models

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

import numpy as np
import os
import vmodel, modparam, data
import copy
import fast_surf, theo
import multiprocessing
from functools import partial
import time
import random

class vprofile1d(object):
    """
    An object for 1D velocity profile inversion
    =====================================================================================================================
    ::: parameters :::
    data                - object storing input data
    model               - object storing 1D model
    eigkR, eigkL        - eigenkernel objects storing Rayleigh/Love eigenfunctions and sensitivity kernels
    disprefR, disprefL  - flags indicating existence of sensitivity kernels for reference model
    =====================================================================================================================
    """
    def __init__(self):
        self.model      = vmodel.model1d()
        self.data       = data.data1d()
        self.fs         = 40.
        self.slowness   = 0.06
        self.gausswidth = 2.5
        self.amplevel   = 0.005
        self.t0         = 0.
        return
    
    def readdisp(self, infname, dtype='ph', wtype='ray'):
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
                self.data.dispR.pvelp = np.zeros(self.data.dispR.npper, dtype=np.float64)
                self.data.dispR.gvelp = np.zeros(self.data.dispR.npper, dtype=np.float64)
#            if self.data.dispR.ngper>0:
#                self.data.dispR.gvelp = np.zeros(self.data.dispR.ngper, dtype=np.float64)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            self.data.dispL.readdisptxt(infname=infname, dtype=dtype)
            if self.data.dispL.npper>0:
                self.data.dispL.pvelp = np.zeros(self.data.dispL.npper, dtype=np.float64)
                self.data.dispL.gvelp = np.zeros(self.data.dispL.npper, dtype=np.float64)
#            if self.data.dispL.ngper>0:
#                self.data.dispL.gvelp = np.zeros(self.data.dispL.ngper, dtype=np.float64)
        else:
            raise ValueError('Unexpected wave type: '+wtype)
        return
    
    def get_disp(self, indata, dtype='ph', wtype='ray'):
        """
        read dispersion curve data from a txt file
        ===========================================================
        ::: input :::
        indata      - input array (3, N)
        dtype       - data type (phase or group)
        wtype       - wave type (Rayleigh or Love)
        ===========================================================
        """
        dtype   = dtype.lower()
        wtype   = wtype.lower()
        if wtype=='ray' or wtype=='rayleigh' or wtype=='r':
            self.data.dispR.get_disp(indata=indata, dtype=dtype)
            if self.data.dispR.npper>0:
                self.data.dispR.pvelp = np.zeros(self.data.dispR.npper, dtype=np.float64)
                self.data.dispR.gvelp = np.zeros(self.data.dispR.npper, dtype=np.float64)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            self.data.dispL.get_disp(indata=indata, dtype=dtype)
            if self.data.dispL.npper>0:
                self.data.dispL.pvelp = np.zeros(self.data.dispL.npper, dtype=np.float64)
                self.data.dispL.gvelp = np.zeros(self.data.dispL.npper, dtype=np.float64)
        else:
            raise ValueError('Unexpected wave type: '+wtype)
        return

    def readrf(self, infname, dtype='r'):
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
                        self.data.rfr.npts, dtype=np.float64)
            self.data.rfr.rfp   = np.zeros(self.data.rfr.npts, dtype=np.float64)
            self.npts           = self.data.rfr.npts
            self.fs             = 1./(self.data.rfr.to[1] - self.data.rfr.to[0])
        elif dtype=='t' or dtype == 'transverse':
            self.data.rft.readrftxt(infname)
        else:
            raise ValueError('Unexpected wave type: '+dtype)
        return
    
    def get_rf(self, indata, dtype='r'):
        """
        read receiver function data from a txt file
        ===========================================================
        ::: input :::
        indata      - input data array (3, N)
        dtype       - data type (radial or transverse)
        ===========================================================
        """
        dtype   = dtype.lower()
        if dtype=='r' or dtype == 'radial':
            self.data.rfr.get_rf(indata=indata)
            self.data.rfr.tp    = np.linspace(self.data.rfr.to[0], self.data.rfr.to[-1], \
                        self.data.rfr.npts, dtype=np.float64)
            self.data.rfr.rfp   = np.zeros(self.data.rfr.npts, dtype=np.float64)
            self.npts           = self.data.rfr.npts
            self.fs             = 1./(self.data.rfr.to[1] - self.data.rfr.to[0])
        # # elif dtype=='t' or dtype == 'transverse':
        # #     self.data.rft.readrftxt(infname)
        else:
            raise ValueError('Unexpected wave type: '+dtype)
        return
    
    def readmod(self, infname, mtype='iso'):
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
        # elif mtype == 'tti':
        #     self.model.ttimod.readttimodtxt(infname)
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def readpara(self, infname, mtype='iso'):
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
    
    def getpara(self, mtype='iso'):
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
    
    def update_mod(self, mtype='iso'):
        """
        update model from model parameters
        =====================================================================
        ::: input :::
        mtype       - model type (0 - isotropic or 1 - tti)
        =====================================================================
        """
        if mtype == 'iso' or mtype == 'isotropic':
            self.model.isomod.update()
#        elif mtype=='tti':
#            self.model.ttimod.update()
        else:
            raise ValueError('Unexpected wave type: '+ mtype)
        return 
   
    
    def get_vmodel(self, mtype='iso'):
        """
        get the velocity model arrays
        =====================================================================
        ::: input :::
        mtype       - model type (0 - isotropic or 1 - tti)
        =====================================================================
        """
        if mtype == 'iso' or mtype == 'isotropic':
            self.model.get_iso_vmodel()
#        elif mtype=='tti':
#            self.qsArr, self.qpArr  = self.model.get_tti_vmodel() # get the model arrays and initialize elastic tensor
#            self.model.rot_dip_strike() 
#            self.model.decompose()
        else:
            raise ValueError('Unexpected wave type: '+ mtype)
        return 
    
    #----------------------------------------------------
    # forward modelling for surface waves
    #----------------------------------------------------
    
    def get_period(self):
        """
        get period array for forward modelling
        """
        if self.data.dispR.npper>0:
            self.TRpiso     = self.data.dispR.pper.copy()
        if self.data.dispR.ngper>0:
            self.TRgiso     = self.data.dispR.gper.copy()
        if self.data.dispL.npper>0:
            self.TLpiso     = self.data.dispL.pper.copy()
        if self.data.dispL.ngper>0:
            self.TLgiso     = self.data.dispL.gper.copy()
        return

    def compute_fsurf(self, wtype='ray'):
        """
        compute surface wave dispersion of isotropic model using fast_surf
        =====================================================================
        ::: input :::
        wtype       - wave type (Rayleigh or Love)
        =====================================================================
        """
        wtype   = wtype.lower()
        if self.model.nlay == 0:
            raise ValueError('No layerized model stored!')
        if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
            ilvry                   = 2
            nper                    = self.TRpiso.size
            per                     = np.zeros(200, dtype=np.float64)
            per[:nper]              = self.TRpiso[:]
            qsinv                   = 1./self.model.qs
            (ur0,ul0,cr0,cl0)       = fast_surf.fast_surf(self.model.nlay, ilvry, \
                                        self.model.vpv, self.model.vsv, self.model.rho, self.model.h, qsinv, per, nper)
            self.data.dispR.pvelp   = cr0[:nper]
            self.data.dispR.gvelp   = ur0[:nper]
        elif wtype=='l' or wtype == 'love':
            ilvry                   = 1
            nper                    = self.TLpiso.size
            per                     = np.zeros(200, dtype=np.float64)
            per[:nper]              = self.TLpiso[:]
            (ur0,ul0,cr0,cl0)       = fast_surf.fast_surf(self.model.nlay, ilvry, \
                                        self.model.vpv, self.model.vsv, self.model.rho, self.model.h, qsinv, per, nper)
            self.data.dispL.pvelp   = cl0[:nper]
            self.data.dispL.gvelp   = ul0[:nper]
        return

    #----------------------------------------------------
    # forward modelling for receiver function
    #----------------------------------------------------
    
    def compute_rftheo(self, slowness = 0.06, din=None, npts=None):
        """
        compute receiver function of isotropic model using theo
        =============================================================================================
        ::: input :::
        slowness- reference horizontal slowness (default - 0.06 s/km, 1./0.06=16.6667)
        din     - incident angle in degree      (default - None, din will be computed from slowness)
        =============================================================================================
        """
        # initialize input model arrays
        hin         = np.zeros(100, dtype=np.float64)
        vsin        = np.zeros(100, dtype=np.float64)
        vpvs        = np.zeros(100, dtype=np.float64)
        qsin        = 600.*np.ones(100, dtype=np.float64)
        qpin        = 1400.*np.ones(100, dtype=np.float64)
        # assign model arrays to the input arrays
        if self.model.nlay<100:
            nl      = self.model.nlay
        else:
            nl      = 100
        hin[:nl]    = self.model.h[:nl]
        vsin[:nl]   = self.model.vsv[:nl]
        vpvs[:nl]   = self.model.vpv[:nl]/self.model.vsv[:nl]
        qsin[:nl]   = self.model.qs[:nl]
        qpin[:nl]   = self.model.qp[:nl]
        # fs/npts
        fs          = self.fs
        # # # ntimes      = 1000
        if npts is None:
            ntimes  = self.npts
        else:
            ntimes  = npts
        # incident angle
        if din is None:
            din     = 180.*np.arcsin(vsin[nl-1]*vpvs[nl-1]*slowness)/np.pi
        # solve for receiver function using theo
        rx 	        = theo.theo(nl, vsin, hin, vpvs, qpin, qsin, fs, din, 2.5, 0.005, 0, ntimes)
        # store the predicted receiver function (ONLY radial component) to the data object
        self.data.rfr.rfp   = rx[:self.npts]
        self.data.rfr.tp    = np.arange(self.npts, dtype=np.float64)*1./self.fs
        return
    #-------------------------------------------------
    # computing misfit
    #-------------------------------------------------
    def get_misfit(self, wdisp=1., rffactor=40.):
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
    
    def mc_joint_inv_iso(self, outdir='./workingdir', dispdtype='ph', wdisp=0.2, rffactor=40.,\
                   monoc=True, pfx='MC', verbose=False, step4uwalk=1500, numbrun=10000, init_run=True, savedata=True):
        """
        Bayesian Monte Carlo joint inversion of receiver function and surface wave data for an isotropic model
        ========================================================================================================
        ::: input :::
        outdir      - output directory
        disptype    - type of dispersion curves (ph/gr/both, default - ph)
        wdisp       - weight of dispersion curve data (0. ~ 1.)
        rffactor    - factor for downweighting the misfit for likelihood computation of rf
        monoc       - require monotonical increase in the crust or not
        pfx         - prefix for output, typically station id
        step4uwalk  - step interval for uniform random walk in the parameter space
        numbrun     - total number of runs
        init_run    - run and output prediction for inital model or not
                        IMPORTANT NOTE: if False, no uniform random walk will perform !
        savedata    - save data to npz binary file or not
        ========================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        #-------------------------------
        # initializations
        #-------------------------------
        self.get_period()
        self.update_mod(mtype = 'iso')
        self.get_vmodel(mtype = 'iso')
        # output arrays
        outmodarr       = np.zeros((numbrun, self.model.isomod.para.npara+9))
        outdisparr_ph   = np.zeros((numbrun, self.data.dispR.npper))
        outdisparr_gr   = np.zeros((numbrun, self.data.dispR.ngper))
        outrfarr        = np.zeros((numbrun, self.data.rfr.npts))
        # initial run
        if init_run:
            self.compute_fsurf()
            self.compute_rftheo()
            self.get_misfit(wdisp=wdisp, rffactor=rffactor)
            # write initial model
            outmod      = outdir+'/'+pfx+'.mod'
            self.model.write_model(outfname=outmod, isotropic=True)
            # write initial predicted data
            if dispdtype != 'both':
                outdisp = outdir+'/'+pfx+'.'+dispdtype+'.disp'
                self.data.dispR.writedisptxt(outfname=outdisp, dtype=dispdtype)
            else:
                outdisp = outdir+'/'+pfx+'.ph.disp'
                self.data.dispR.writedisptxt(outfname=outdisp, dtype='ph')
                outdisp = outdir+'/'+pfx+'.gr.disp'
                self.data.dispR.writedisptxt(outfname=outdisp, dtype='gr')
            outrf       = outdir+'/'+pfx+'.rf'
            self.data.rfr.writerftxt(outfname=outrf)
            # convert initial model to para
            self.model.isomod.mod2para()
        else:
            self.model.isomod.mod2para()
            newmod      = copy.deepcopy(self.model.isomod)
            newmod.para.new_paraval(0)
            newmod.para2mod()
            newmod.update()
            # loop to find the "good" model,
            # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
            igood       = 0
            while ( not newmod.isgood(0, 1, 1, 0)):
                igood   += igood + 1
                newmod  = copy.deepcopy(self.model.isomod)
                newmod.para.new_paraval(0)
                newmod.para2mod()
                newmod.update()
            # assign new model to old ones
            self.model.isomod   = newmod
            self.get_vmodel(mtype = 'iso')
            # forward computation
            self.compute_fsurf()
            self.compute_rftheo()
            self.get_misfit(wdisp=wdisp, rffactor=rffactor)
            if verbose:
                print pfx+', uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit
            self.model.isomod.mod2para()
        # likelihood/misfit
        oldL        = self.data.L
        oldmisfit   = self.data.misfit
        run         = True     # the key that controls the sampling
        inew        = 0     # count step (or new paras)
        iacc        = 0     # count acceptance model
        start       = time.time()
        while ( run ):
            inew    += 1
            if ( inew > numbrun ):
                break
            if (np.fmod(inew, 500) == 0) and verbose:
                print pfx, 'step =',inew, 'elasped time =', time.time()-start,' sec'
            #------------------------------------------------------------------------------------------
            # every step4uwalk step, perform a random walk with uniform random value in the paramerter space
            #------------------------------------------------------------------------------------------
            if ( np.fmod(inew, step4uwalk+1) == step4uwalk and init_run ):
                newmod      = copy.deepcopy(self.model.isomod)
                newmod.para.new_paraval(0)
                newmod.para2mod()
                newmod.update()
                # loop to find the "good" model,
                # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                igood       = 0
                while ( not newmod.isgood(0, 1, 1, 0)):
                    igood   += igood + 1
                    newmod  = copy.deepcopy(self.model.isomod)
                    newmod.para.new_paraval(0)
                    newmod.para2mod()
                    newmod.update()
                # assign new model to old ones
                self.model.isomod   = newmod
                self.get_vmodel()
                # forward computation
                self.compute_fsurf()
                self.compute_rftheo()
                self.get_misfit(wdisp=wdisp, rffactor=rffactor)
                oldL                = self.data.L
                oldmisfit           = self.data.misfit
                if verbose:
                    print pfx+', uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit
            #-------------------------------
            # inversion part
            #-------------------------------
            # sample the posterior distribution 
            if (wdisp >= 0. and wdisp <=1.):
                newmod      = copy.deepcopy(self.model.isomod)
                newmod.para.new_paraval(1)
                newmod.para2mod()
                newmod.update()
                if monoc:
                    # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                    # loop to find the "good" model, added on May 3rd, 2018
                    itemp   = 0
                    while (not newmod.isgood(0, 1, 1, 0)) and itemp < 100:
                        itemp       += 1
                        newmod      = copy.deepcopy(self.model.isomod)
                        newmod.para.new_paraval(1)
                        newmod.para2mod()
                        newmod.update()
                    if not newmod.isgood(0, 1, 1, 0):
                        print 'No good model found!'
                        continue
                # assign new model to old ones
                oldmod              = copy.deepcopy(self.model.isomod)
                self.model.isomod   = newmod
                self.get_vmodel()
                #--------------------------------
                # forward computation
                #--------------------------------
                if wdisp > 0.:
                    self.compute_fsurf()
                if wdisp < 1.:
                    self.compute_rftheo()
                self.get_misfit(wdisp=wdisp, rffactor=rffactor)
                newL                = self.data.L
                newmisfit           = self.data.misfit
                if newL < oldL:
                    prob    = (oldL-newL)/oldL
                    rnumb   = random.random()
                    # reject the model
                    if rnumb < prob:
                        outmodarr[inew-1, 0]                        = -1 # index for acceptance
                        outmodarr[inew-1, 1]                        = iacc
                        outmodarr[inew-1, 2:(newmod.para.npara+2)]  = newmod.para.paraval[:]
                        outmodarr[inew-1, newmod.para.npara+2]      = newL
                        outmodarr[inew-1, newmod.para.npara+3]      = newmisfit
                        outmodarr[inew-1, newmod.para.npara+4]      = self.data.rfr.L
                        outmodarr[inew-1, newmod.para.npara+5]      = self.data.rfr.misfit
                        outmodarr[inew-1, newmod.para.npara+6]      = self.data.dispR.L
                        outmodarr[inew-1, newmod.para.npara+7]      = self.data.dispR.misfit
                        outmodarr[inew-1, newmod.para.npara+8]      = time.time()-start
                        self.model.isomod                           = oldmod
                        continue
                # accept the new model
                outmodarr[inew-1, 0]                        = 1 # index for acceptance
                outmodarr[inew-1, 1]                        = iacc
                outmodarr[inew-1, 2:(newmod.para.npara+2)]  = newmod.para.paraval[:]
                outmodarr[inew-1, newmod.para.npara+2]      = newL
                outmodarr[inew-1, newmod.para.npara+3]      = newmisfit
                outmodarr[inew-1, newmod.para.npara+4]      = self.data.rfr.L
                outmodarr[inew-1, newmod.para.npara+5]      = self.data.rfr.misfit
                outmodarr[inew-1, newmod.para.npara+6]      = self.data.dispR.L
                outmodarr[inew-1, newmod.para.npara+7]      = self.data.dispR.misfit
                outmodarr[inew-1, newmod.para.npara+8]      = time.time()-start
                # predicted dispersion data
                if dispdtype == 'ph' or dispdtype == 'both':
                    outdisparr_ph[inew-1, :]    = self.data.dispR.pvelp[:]
                if dispdtype == 'gr' or dispdtype == 'both':
                    outdisparr_gr[inew-1, :]    = self.data.dispR.gvelp[:]
                # predicted receiver function data
                outrfarr[inew-1, :]             = self.data.rfr.rfp[:]
                # assign likelihood/misfit
                oldL        = newL
                oldmisfit   = newmisfit
                iacc        += 1
                continue
            # # # else:
            # # #     if monoc:
            # # #         newmod  = self.model.isomod.copy()
            # # #         newmod.para.new_paraval(1)
            # # #         newmod.para2mod()
            # # #         newmod.update()
            # # #         if not newmod.isgood(0, 1, 1, 0):
            # # #             continue
            # # #     else:
            # # #         newmod  = self.model.isomod.copy()
            # # #         newmod.para.new_paraval(0)
            # # #     fidout.write("-2 %d 0 " % inew)
            # # #     for i in xrange(newmod.para.npara):
            # # #         fidout.write("%g " % newmod.para.paraval[i])
            # # #     fidout.write("\n")
            # # #     self.model.isomod   = newmod
            # # #     continue
        #-----------------------------------
        # write results to binary npz files
        #-----------------------------------
        outfname    = outdir+'/mc_inv.'+pfx+'.npz'
        np.savez_compressed(outfname, outmodarr, outdisparr_ph, outdisparr_gr, outrfarr)
        if savedata:
            outfname    = outdir+'/mc_data.'+pfx+'.npz'
            try:
                np.savez_compressed(outfname, np.array([1, 1, 1]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                        self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo, \
                        self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
            except AttributeError:
                try:
                    np.savez_compressed(outfname, np.array([1, 0, 1]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                            self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
                except AttributeError:
                    np.savez_compressed(outfname, np.array([0, 1, 1]), self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo,\
                        self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
        del outmodarr
        del outdisparr_ph
        del outdisparr_gr
        del outrfarr
        return
    
    def mc_joint_inv_iso_mp(self, outdir='./workingdir', dispdtype='ph', wdisp=0.2, rffactor=40., monoc=True, \
            pfx='MC', verbose=False, step4uwalk=1500, numbrun=15000, savedata=True, subsize=1000, nprocess=None, merge=True):
        """
        Parallelized version of mc_joint_inv_iso
        ==================================================================================================================
        ::: input :::
        outdir      - output directory
        disptype    - type of dispersion curves (ph/gr/both, default - ph)
        wdisp       - weight of dispersion curve data (0. ~ 1.)
        rffactor    - factor for downweighting the misfit for likelihood computation of rf
        monoc       - require monotonical increase in the crust or not
        pfx         - prefix for output, typically station id
        step4uwalk  - step interval for uniform random walk in the parameter space
        numbrun     - total number of runs
        savedata    - save data to npz binary file or not
        subsize     - size of subsets, used if the number of elements in the parallel list is too large to avoid deadlock
        nprocess    - number of process
        merge       - merge data into one single npz file or not
        ==================================================================================================================
        """
        #-------------------------
        # prepare data
        #-------------------------
        vpr_lst = []
        Nvpr    = int(numbrun/step4uwalk)
        if Nvpr*step4uwalk != numbrun:
            print 'WARNING: number of runs changes: '+str(numbrun)+' --> '+str(Nvpr*step4uwalk)
            numbrun     = Nvpr*step4uwalk
        for i in range(Nvpr):
            temp_vpr            = copy.deepcopy(self)
            temp_vpr.process_id = i
            vpr_lst.append(temp_vpr)
        #----------------------------------------
        # Joint inversion with multiprocessing
        #----------------------------------------
        if verbose:
            print 'Start MC joint inversion: '+pfx+' '+time.ctime()
            stime   = time.time()
        if Nvpr > subsize:
            Nsub                = int(len(vpr_lst)/subsize)
            for isub in xrange(Nsub):
                print 'Subset:', isub,'in',Nsub,'sets'
                cvpr_lst        = vpr_lst[isub*subsize:(isub+1)*subsize]
                MCINV           = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                                    monoc=monoc, pfx=pfx, verbose=verbose, numbrun=step4uwalk)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(MCINV, cvpr_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cvpr_lst            = vpr_lst[(isub+1)*subsize:]
            MCINV               = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                                    monoc=monoc, pfx=pfx, verbose=verbose, numbrun=step4uwalk)
            pool                = multiprocessing.Pool(processes=nprocess)
            pool.map(MCINV, cvpr_lst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            MCINV               = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                                    monoc=monoc, pfx=pfx, verbose=verbose, numbrun=step4uwalk)
            pool                = multiprocessing.Pool(processes=nprocess)
            pool.map(MCINV, vpr_lst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        #----------------------------------------
        # Merge inversion results
        #----------------------------------------
        if merge:
            outmodarr           = np.array([])
            outdisparr_ph       = np.array([])
            outdisparr_gr       = np.array([])
            outrfarr            = np.array([])
            for i in range(Nvpr):
                invfname        = outdir+'/mc_inv.'+pfx+'_'+str(i)+'.npz'
                inarr           = np.load(invfname)
                outmodarr       = np.append(outmodarr, inarr['arr_0'])
                outdisparr_ph   = np.append(outdisparr_ph, inarr['arr_1'])
                outdisparr_gr   = np.append(outdisparr_gr, inarr['arr_2'])
                outrfarr        = np.append(outrfarr, inarr['arr_3'])
                os.remove(invfname)
            outmodarr           = outmodarr.reshape(numbrun, outmodarr.size/numbrun)
            outdisparr_ph       = outdisparr_ph.reshape(numbrun, outdisparr_ph.size/numbrun)
            outdisparr_gr       = outdisparr_gr.reshape(numbrun, outdisparr_gr.size/numbrun)
            outrfarr            = outrfarr.reshape(numbrun, outrfarr.size/numbrun)
            outinvfname         = outdir+'/mc_inv.'+pfx+'.npz'
            np.savez_compressed(outinvfname, outmodarr, outdisparr_ph, outdisparr_gr, outrfarr)
        # save data
        if savedata:
            outdatafname    = outdir+'/mc_data.'+pfx+'.npz'
            try:
                np.savez_compressed(outdatafname, np.array([1, 1, 1]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                        self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo, \
                        self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
            except AttributeError:
                try:
                    np.savez_compressed(outdatafname, np.array([1, 0, 1]), self.data.dispR.pper, self.data.dispR.pvelo, self.data.dispR.stdpvelo,\
                            self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
                except AttributeError:
                    np.savez_compressed(outdatafname, np.array([0, 1, 1]), self.data.dispR.gper, self.data.dispR.gvelo, self.data.dispR.stdgvelo,\
                        self.data.rfr.to, self.data.rfr.rfo, self.data.rfr.stdrfo)
        if verbose:
            print 'End MC joint inversion: '+pfx+' '+time.ctime()
            etime   = time.time()
            print 'Elapsed time: '+str(etime-stime)+' secs'
        return
        
def mc4mp(invpr, outdir, dispdtype, wdisp, rffactor, monoc, pfx, verbose, numbrun):
    print '--- Joint MC inversion for station: '+pfx+', process id: '+str(invpr.process_id)
    pfx     = pfx +'_'+str(invpr.process_id)
    if invpr.process_id == 0:
        invpr.mc_joint_inv_iso(outdir=outdir, wdisp=wdisp, rffactor=rffactor,\
                       monoc=monoc, pfx=pfx, verbose=False, step4uwalk=numbrun, numbrun=numbrun, init_run=True, savedata=False)
    else:
        invpr.mc_joint_inv_iso(outdir=outdir, wdisp=wdisp, rffactor=rffactor,\
                       monoc=monoc, pfx=pfx, verbose=False, step4uwalk=numbrun, numbrun=numbrun, init_run=False, savedata=False)
    return 
    
    
    
    # def mc_inv_iso_old(self, outdir='./workingdir', dispdtype='ph', wdisp=0.2, rffactor=40., monoc=True, pfx='MC'):
    #     """
    #     
    #     """
    #     if not os.path.isdir(outdir):
    #         os.makedirs(outdir)
    #     # initializations
    #     self.get_period()
    #     self.update_mod(mtype = 'iso')
    #     self.get_vmodel(mtype = 'iso')
    #     # initial run
    #     self.compute_fsurf()
    #     self.compute_rftheo()
    #     self.get_misfit(wdisp=wdisp, rffactor=rffactor)
    #     # write initial model
    #     outmod  = outdir+'/'+pfx+'.mod'
    #     self.model.write_model(outfname=outmod, isotropic=True)
    #     # write initial predicted data
    #     if dispdtype != 'both':
    #         outdisp = outdir+'/'+pfx+'.'+dispdtype+'.disp'
    #         self.data.dispR.writedisptxt(outfname=outdisp, dtype=dispdtype)
    #     else:
    #         outdisp = outdir+'/'+pfx+'.ph.disp'
    #         self.data.dispR.writedisptxt(outfname=outdisp, dtype='ph')
    #         outdisp = outdir+'/'+pfx+'.gr.disp'
    #         self.data.dispR.writedisptxt(outfname=outdisp, dtype='gr')
    #     outrf   = outdir+'/'+pfx+'.rf'
    #     self.data.rfr.writerftxt(outfname=outrf)
    #     # convert initial model to para
    #     self.model.isomod.mod2para()
    #     # likelihood/misfit
    #     oldL        = self.data.L
    #     oldmisfit   = self.data.misfit
    #     print "Initial likelihood = ", oldL, ' misfit =',oldmisfit
    #     run         = True     # the key that controls the sampling
    #     inew        = 0     # count step (or new paras)
    #     iacc        = 1     # count acceptance model
    #     start       = time.time()
    #     # output log files
    #     outtxtfname = outdir+'/'+pfx+'.out'
    #     outbinfname = outdir+'/MC.bin'
    #     fidout      = open(outtxtfname, "w")
    #     # fidoutb     = open(outbinfname, "wb")
    #     while ( run ):
    #         inew+= 1
    #         # print 'run step = ',inew
    #         # # # if ( inew > 100000 or iacc > 20000000 or time.time()-start > 7200.):
    #         if ( inew > 10000 or iacc > 20000000):
    #             run   = False
    #         if (np.fmod(inew, 500) == 0):
    #             print 'step =',inew, 'elasped time =', time.time()-start, ' sec'
    #         #------------------------------------------------------------------------------------------
    #         # every 2500 step, perform a random walk with uniform random value in the paramerter space
    #         #------------------------------------------------------------------------------------------
    #         if ( np.fmod(inew, 1501) == 1500 ):
    #             newmod      = copy.deepcopy(self.model.isomod)
    #             newmod.para.new_paraval(0)
    #             newmod.para2mod()
    #             newmod.update()
    #             # loop to find the "good" model,
    #             # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
    #             igood   = 0
    #             while ( not newmod.isgood(0, 1, 1, 0)):
    #                 igood   += igood + 1
    #                 newmod  = copy.deepcopy(self.model.isomod)
    #                 newmod.para.new_paraval(0)
    #                 newmod.para2mod()
    #                 newmod.update()
    #             # assign new model to old ones
    #             self.model.isomod   = newmod
    #             self.get_vmodel()
    #             # forward computation
    #             self.compute_fsurf()
    #             self.compute_rftheo()
    #             self.get_misfit(wdisp=wdisp, rffactor=rffactor)
    #             oldL                = self.data.L
    #             oldmisfit           = self.data.misfit
    #             iacc                += 1
    #             print 'Uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit
    #         #-------------------------------
    #         # inversion part
    #         #-------------------------------
    #         # sample the posterior distribution ##########################################
    #         if (wdisp >= 0 and wdisp <=1):
    #             newmod      = copy.deepcopy(self.model.isomod)
    #             # newmod.para = copy.deepcopy(self.model.isomod.para)
    #             newmod.para.new_paraval(1)
    #             newmod.para2mod()
    #             newmod.update()
    #             if monoc:
    #                 # loop to find the "good" model,
    #                 # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
    #                 if not newmod.isgood(0, 1, 1, 0):
    #                     continue
    #             # assign new model to old ones
    #             oldmod              = copy.deepcopy(self.model.isomod)
    #             # oldmod.para         = copy.deepcopy(self.model.isomod.para)
    #             self.model.isomod   = newmod
    #             self.get_vmodel()
    #             # forward computation
    #             self.compute_fsurf()
    #             self.compute_rftheo()
    #             self.get_misfit(wdisp=wdisp, rffactor=rffactor)
    #             newL                = self.data.L
    #             newmisfit           = self.data.misfit
    #             # 
    #             if newL < oldL:
    #                 prob    = (oldL-newL)/oldL
    #                 rnumb   = random.random()
    #                 # reject the model
    #                 if rnumb < prob:
    #                     fidout.write("-1 %d %d " % (inew,iacc))
    #                     for i in xrange(newmod.para.npara):
    #                         fidout.write("%g " % newmod.para.paraval[i])
    #                     fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.data.rfr.L, self.data.rfr.misfit,\
    #                             self.data.dispR.L, self.data.dispR.misfit, time.time()-start))        
    #                     ### ttmodel.writeb (para1, ffb,[-1,i,ii])
    #                     # return to oldmod
    #                     self.model.isomod   = oldmod
    #                     continue
    #             # accept the new model
    #             fidout.write("1 %d %d " % (inew,iacc))
    #             for i in xrange(newmod.para.npara):
    #                 fidout.write("%g " % newmod.para.paraval[i])
    #             fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.data.rfr.L, self.data.rfr.misfit,\
    #                     self.data.dispR.L, self.data.dispR.misfit, time.time()-start))        
    #             print "Accept a model", inew, iacc, oldL, newL, self.data.rfr.L, self.data.rfr.misfit,\
    #                             self.data.dispR.L, self.data.dispR.misfit, time.time()-start
    #             # # write accepted model
    #             # outmod      = outdir+'/'+pfx+'.%d.mod' % iacc
    #             # vmodel.write_model(model=self.model, outfname=outmod, isotropic=True)
    #             # # write corresponding data
    #             # if dispdtype != 'both':
    #             #     outdisp = outdir+'/'+pfx+'.'+dispdtype+'.%d.disp' % iacc
    #             #     data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
    #             # else:
    #             #     outdisp = outdir+'/'+pfx+'.ph.%d.disp' % iacc
    #             #     data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='ph')
    #             #     outdisp = outdir+'/'+pfx+'.gr.%d.disp' % iacc
    #             #     data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='gr')
    #             # # outdisp = outdir+'/'+pfx+'.%d.disp' % iacc
    #             # # data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
    #             # outrf   = outdir+'/'+pfx+'.%d.rf' % iacc
    #             # data.writerftxt(outfname=outrf, outrf=self.indata.rfr)
    #             # assign likelihood/misfit
    #             oldL        = newL
    #             oldmisfit   = newmisfit
    #             iacc        += 1
    #             continue
    #     #     else:
    #     #         if monoc:
    #     #             newmod  = self.model.isomod.copy()
    #     #             newmod.para.new_paraval(1)
    #     #             newmod.para2mod()
    #     #             newmod.update()
    #     #             if not newmod.isgood(0, 1, 1, 0):
    #     #                 continue
    #     #         else:
    #     #             newmod  = self.model.isomod.copy()
    #     #             newmod.para.new_paraval(0)
    #     #         fidout.write("-2 %d 0 " % inew)
    #     #         for i in xrange(newmod.para.npara):
    #     #             fidout.write("%g " % newmod.para.paraval[i])
    #     #         fidout.write("\n")
    #     #         self.model.isomod   = newmod
    #     #         continue
    #     # fidout.close()
    #     return