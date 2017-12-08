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
    
    def compute_rftheo(self, slowness = 0.06, din=None):
        """
        compute receiver function of isotropic model using theo
        ===========================================================================================
        ::: input :::
        slowness- reference horizontal slowness (default - 0.06 s/km, 1./0.06=16.6667)
        din     - incident angle in degree (default - None, din will be computed from slowness)
        ===========================================================================================
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
        hin[:nl]    = self.model.h
        vsin[:nl]   = self.model.vsv
        vpvs[:nl]   = self.model.vpv/self.model.vsv
        qsin[:nl]   = self.model.qs
        qpin[:nl]   = self.model.qp
        # fs/npts
        fs          = self.fs
        # # # ntimes      = 1000
        ntimes      = self.npts
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
    
    def mc_inv_single_thread_iso(self, outdir, ind0=0, ind1=2000, indid=1, pfx='MC', dispdtype='ph', \
                 wdisp=1., rffactor=40., monoc=True, randstart=True):
        """
        
        """
        if wdisp > 1. or wdisp < 0.:
            raise ValueError('Weight for surface wave should be within [0., 1.]')
        if ind0 > ind1 or ind0<0:
            raise ValueError('Error input for index, ind0 = ',+str(ind0)+' ind1 = ', str(ind1))
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # initializations
        self.get_period()
        if ind0 != 0 and randstart:
            self.model.isomod.mod2para()
            newmod      = copy.deepcopy(self.model.isomod)
            newmod.para.new_paraval(0)
            newmod.para2mod()
            newmod.update()
            # loop to find the "good" model,
            # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
            igood   = 0
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
            print 'Random start, runid = '+str(indid)+', likelihood =', self.data.L, 'misfit =',self.data.misfit
        else:
            self.update_mod(mtype = 'iso')
            self.get_vmodel(mtype = 'iso')
            # initial run
            self.compute_fsurf()
            self.compute_rftheo()
            self.get_misfit(wdisp=wdisp, rffactor=rffactor)
            # likelihood/misfit
            oldL        = self.data.L
            oldmisfit   = self.data.misfit
            print 'Initial start, runid = '+str(indid)+', likelihood =', self.data.L, 'misfit =',self.data.misfit
        # write initial model
        outmod  = outdir+'/'+pfx+'.'+str(indid)+'.mod'
        self.model.write_model(outfname=outmod, isotropic=True)
        # write initial predicted data
        if dispdtype != 'both':
            outdisp = outdir+'/'+pfx+'.'+str(indid)+'.'+dispdtype+'.disp'
            self.data.dispR.writedisptxt(outfname=outdisp, dtype=dispdtype)
        else:
            outdisp = outdir+'/'+pfx+'.'+str(indid)+'.ph.disp'
            self.data.dispR.writedisptxt(outfname=outdisp, dtype='ph')
            outdisp = outdir+'/'+pfx+'.'+str(indid)+'.gr.disp'
            self.data.dispR.writedisptxt(outfname=outdisp, dtype='gr')
        outrf       = outdir+'/'+pfx+'.'+str(indid)+'.rf'
        self.data.rfr.writerftxt(outfname=outrf)
        # convert initial model to para
        self.model.isomod.mod2para()
        run         = True      # the key that controls the sampling
        inew        = ind0      # count step (or new paras)
        iacc        = 0         # count acceptance model
        start       = time.time()
        # initialize output array
        dispArr         = np.zeros((self.data.dispR.npper, ind1-ind0), dtype=np.float64)
        rfArr           = np.zeros((self.data.rfr.npts, ind1-ind0), dtype=np.float64)
        misfitArr       = np.zeros((3+self.model.isomod.para.npara+7+1, ind1-ind0), dtype=np.float64)
        # store first run in output arrays
        inew            += 1
        dispArr[:,0]    = self.data.dispR.pvelp[:]
        rfArr[:,0]      = self.data.rfr.rfp[:]
        misfitArr[:4, 0]= np.array([-1, ind0+1, 0, indid])
        misfitArr[4:4+self.model.isomod.para.npara, 0]  = self.model.isomod.para.paraval[:]
        misfitArr[4+self.model.isomod.para.npara:, 0]   = np.array([oldL, oldmisfit, self.data.rfr.L, self.data.rfr.misfit, \
                                                            self.data.dispR.L, self.data.dispR.misfit, time.time()-start])
        while ( run ):
            inew    += 1
            # print 'run step = ',inew
            if ( inew > 10000 or iacc > 20000000):
                run   = False
            if (np.fmod(inew, 500) == 0):
                print 'step =',inew, 'elasped time =', time.time()-start, ' sec'
            #------------------------------------------------------------------------------------------
            # every 2500 step, perform a random walk with uniform random value in the paramerter space
            #------------------------------------------------------------------------------------------
            if ( np.fmod(inew, 1501) == 1500 ):
                newmod      = copy.deepcopy(self.model.isomod)
                newmod.para.new_paraval(0)
                newmod.para2mod()
                newmod.update()
                # loop to find the "good" model,
                # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                igood   = 0
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
                iacc                += 1
                print 'Uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit
            #-------------------------------
            # inversion part
            #-------------------------------
            # sample the posterior distribution ##########################################
            if (wdisp >= 0 and wdisp <=1):
                newmod      = copy.deepcopy(self.model.isomod)
                # newmod.para = copy.deepcopy(self.model.isomod.para)
                newmod.para.new_paraval(1)
                newmod.para2mod()
                newmod.update()
                if monoc:
                    # loop to find the "good" model,
                    # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                    if not newmod.isgood(0, 1, 1, 0):
                        continue
                # assign new model to old ones
                oldmod              = copy.deepcopy(self.model.isomod)
                # oldmod.para         = copy.deepcopy(self.model.isomod.para)
                self.model.isomod   = newmod
                self.get_vmodel()
                # forward computation
                self.compute_fsurf()
                self.compute_rftheo()
                self.get_misfit(wdisp=wdisp, rffactor=rffactor)
                newL                = self.data.L
                newmisfit           = self.data.misfit
                # 
                if newL < oldL:
                    prob    = (oldL-newL)/oldL
                    rnumb   = random.random()
                    # reject the model
                    if rnumb < prob:
                        fidout.write("-1 %d %d " % (inew,iacc))
                        for i in xrange(newmod.para.npara):
                            fidout.write("%g " % newmod.para.paraval[i])
                        fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.data.rfr.L, self.data.rfr.misfit,\
                                self.data.dispR.L, self.data.dispR.misfit, time.time()-start))        
                        ### ttmodel.writeb (para1, ffb,[-1,i,ii])
                        # return to oldmod
                        self.model.isomod   = oldmod
                        continue
                # accept the new model
                fidout.write("1 %d %d " % (inew,iacc))
                for i in xrange(newmod.para.npara):
                    fidout.write("%g " % newmod.para.paraval[i])
                fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.data.rfr.L, self.data.rfr.misfit,\
                        self.data.dispR.L, self.data.dispR.misfit, time.time()-start))        
                print "Accept a model", inew, iacc, oldL, newL, self.data.rfr.L, self.data.rfr.misfit,\
                                self.data.dispR.L, self.data.dispR.misfit, time.time()-start
                # # write accepted model
                # outmod      = outdir+'/'+pfx+'.%d.mod' % iacc
                # vmodel.write_model(model=self.model, outfname=outmod, isotropic=True)
                # # write corresponding data
                # if dispdtype != 'both':
                #     outdisp = outdir+'/'+pfx+'.'+dispdtype+'.%d.disp' % iacc
                #     data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
                # else:
                #     outdisp = outdir+'/'+pfx+'.ph.%d.disp' % iacc
                #     data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='ph')
                #     outdisp = outdir+'/'+pfx+'.gr.%d.disp' % iacc
                #     data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='gr')
                # # outdisp = outdir+'/'+pfx+'.%d.disp' % iacc
                # # data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
                # outrf   = outdir+'/'+pfx+'.%d.rf' % iacc
                # data.writerftxt(outfname=outrf, outrf=self.indata.rfr)
                # assign likelihood/misfit
                oldL        = newL
                oldmisfit   = newmisfit
                iacc        += 1
                continue
        #     else:
        #         if monoc:
        #             newmod  = self.model.isomod.copy()
        #             newmod.para.new_paraval(1)
        #             newmod.para2mod()
        #             newmod.update()
        #             if not newmod.isgood(0, 1, 1, 0):
        #                 continue
        #         else:
        #             newmod  = self.model.isomod.copy()
        #             newmod.para.new_paraval(0)
        #         fidout.write("-2 %d 0 " % inew)
        #         for i in xrange(newmod.para.npara):
        #             fidout.write("%g " % newmod.para.paraval[i])
        #         fidout.write("\n")
        #         self.model.isomod   = newmod
        #         continue
        # fidout.close()
        return
    
    def mc_inv_iso(self, outdir='./workingdir', dispdtype='ph', wdisp=0.2, rffactor=40., monoc=True, pfx='MC'):
        """
        
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # initializations
        self.get_period()
        self.update_mod(mtype = 'iso')
        self.get_vmodel(mtype = 'iso')
        # initial run
        self.compute_fsurf()
        self.compute_rftheo()
        self.get_misfit(wdisp=wdisp, rffactor=rffactor)
        # write initial model
        outmod  = outdir+'/'+pfx+'.mod'
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
        outrf   = outdir+'/'+pfx+'.rf'
        self.data.rfr.writerftxt(outfname=outrf)
        # convert initial model to para
        self.model.isomod.mod2para()
        # likelihood/misfit
        oldL        = self.data.L
        oldmisfit   = self.data.misfit
        print "Initial likelihood = ", oldL, ' misfit =',oldmisfit
        
        run     = True     # the key that controls the sampling
        inew    = 0     # count step (or new paras)
        iacc    = 1     # count acceptance model
        start   = time.time()
        # output log files
        outtxtfname = outdir+'/'+pfx+'.out'
        outbinfname = outdir+'/MC.bin'
        fidout      = open(outtxtfname, "w")
        # fidoutb     = open(outbinfname, "wb")
        while ( run ):
            inew+= 1
            # print 'run step = ',inew
            # # # if ( inew > 100000 or iacc > 20000000 or time.time()-start > 7200.):
            if ( inew > 10000 or iacc > 20000000):
                run   = False
            if (np.fmod(inew, 500) == 0):
                print 'step =',inew, 'elasped time =', time.time()-start, ' sec'
            #------------------------------------------------------------------------------------------
            # every 2500 step, perform a random walk with uniform random value in the paramerter space
            #------------------------------------------------------------------------------------------
            if ( np.fmod(inew, 1501) == 1500 ):
                newmod      = copy.deepcopy(self.model.isomod)
                newmod.para.new_paraval(0)
                newmod.para2mod()
                newmod.update()
                # loop to find the "good" model,
                # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                igood   = 0
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
                iacc                += 1
                print 'Uniform random walk: likelihood =', self.data.L, 'misfit =',self.data.misfit
            #-------------------------------
            # inversion part
            #-------------------------------
            # sample the posterior distribution ##########################################
            if (wdisp >= 0 and wdisp <=1):
                newmod      = copy.deepcopy(self.model.isomod)
                # newmod.para = copy.deepcopy(self.model.isomod.para)
                newmod.para.new_paraval(1)
                newmod.para2mod()
                newmod.update()
                if monoc:
                    # loop to find the "good" model,
                    # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                    if not newmod.isgood(0, 1, 1, 0):
                        continue
                # assign new model to old ones
                oldmod              = copy.deepcopy(self.model.isomod)
                # oldmod.para         = copy.deepcopy(self.model.isomod.para)
                self.model.isomod   = newmod
                self.get_vmodel()
                # forward computation
                self.compute_fsurf()
                self.compute_rftheo()
                self.get_misfit(wdisp=wdisp, rffactor=rffactor)
                newL                = self.data.L
                newmisfit           = self.data.misfit
                # 
                if newL < oldL:
                    prob    = (oldL-newL)/oldL
                    rnumb   = random.random()
                    # reject the model
                    if rnumb < prob:
                        fidout.write("-1 %d %d " % (inew,iacc))
                        for i in xrange(newmod.para.npara):
                            fidout.write("%g " % newmod.para.paraval[i])
                        fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.data.rfr.L, self.data.rfr.misfit,\
                                self.data.dispR.L, self.data.dispR.misfit, time.time()-start))        
                        ### ttmodel.writeb (para1, ffb,[-1,i,ii])
                        # return to oldmod
                        self.model.isomod   = oldmod
                        continue
                # accept the new model
                fidout.write("1 %d %d " % (inew,iacc))
                for i in xrange(newmod.para.npara):
                    fidout.write("%g " % newmod.para.paraval[i])
                fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.data.rfr.L, self.data.rfr.misfit,\
                        self.data.dispR.L, self.data.dispR.misfit, time.time()-start))        
                print "Accept a model", inew, iacc, oldL, newL, self.data.rfr.L, self.data.rfr.misfit,\
                                self.data.dispR.L, self.data.dispR.misfit, time.time()-start
                # # write accepted model
                # outmod      = outdir+'/'+pfx+'.%d.mod' % iacc
                # vmodel.write_model(model=self.model, outfname=outmod, isotropic=True)
                # # write corresponding data
                # if dispdtype != 'both':
                #     outdisp = outdir+'/'+pfx+'.'+dispdtype+'.%d.disp' % iacc
                #     data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
                # else:
                #     outdisp = outdir+'/'+pfx+'.ph.%d.disp' % iacc
                #     data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='ph')
                #     outdisp = outdir+'/'+pfx+'.gr.%d.disp' % iacc
                #     data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='gr')
                # # outdisp = outdir+'/'+pfx+'.%d.disp' % iacc
                # # data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
                # outrf   = outdir+'/'+pfx+'.%d.rf' % iacc
                # data.writerftxt(outfname=outrf, outrf=self.indata.rfr)
                # assign likelihood/misfit
                oldL        = newL
                oldmisfit   = newmisfit
                iacc        += 1
                continue
        #     else:
        #         if monoc:
        #             newmod  = self.model.isomod.copy()
        #             newmod.para.new_paraval(1)
        #             newmod.para2mod()
        #             newmod.update()
        #             if not newmod.isgood(0, 1, 1, 0):
        #                 continue
        #         else:
        #             newmod  = self.model.isomod.copy()
        #             newmod.para.new_paraval(0)
        #         fidout.write("-2 %d 0 " % inew)
        #         for i in xrange(newmod.para.npara):
        #             fidout.write("%g " % newmod.para.paraval[i])
        #         fidout.write("\n")
        #         self.model.isomod   = newmod
        #         continue
        # fidout.close()
        return
#         
# 
#     
#     @cython.boundscheck(False)
#     cdef void mc_inv_iso(self, char *outdir, char *pfx, char *dispdtype, \
#                     float wdisp=1., float rffactor=40., int monoc=1) nogil:
#         """
#         
#         """
#         cdef char *outmod, *outdisp, *outrf
#         cdef float oldL, oldmisfit, newL, newmisfit, prob, rnumb
#         cdef int run = 1
#         cdef Py_ssize_t inew, iacc, igood, i
#         cdef float[23][100000] outArr
# #        with gil:
# #            def modparam.isomod newmod 
# #        with gil:
# #            newmod= modparam.isomod()
# #        cdef 
#         # initializations
#         self.get_period()
#         self.update_mod(0)
#         self.get_vmodel(0)
#         # initial run
#         self.compute_fsurf()
#         self.compute_rftheo()
#         self.get_misfit(wdisp, rffactor)
#         # write initial model
#         outmod = <char *>malloc((strlen(outdir)+1+strlen(pfx)+4) * sizeof(char))
#         strcpy(outmod, outdir)
#         strcat(outmod, '/')
#         strcat(outmod, pfx)
#         strcat(outmod, '.mod')
#         with gil:
#             self.model.write_model(outfname=outmod, isotropic=1)
#         free(outmod)
#         # write initial predicted data
#         with gil:
#             if strcmp(dispdtype, 'both') != 0:
#                 outdisp = <char *>malloc((strlen(outdir)+1+strlen(pfx)+6+strlen(dispdtype)) * sizeof(char))
#                 strcpy(outdisp, outdir)
#                 strcat(outdisp, '/')
#                 strcat(outdisp, pfx)
#                 strcat(outdisp, '.')
#                 strcat(outdisp, dispdtype)
#                 strcat(outdisp, '.disp')
#                 self.data.dispR.writedisptxt(outfname=outdisp, dtype=dispdtype)
#                 free(outdisp)
#             else:
#                 outdisp = <char *>malloc((strlen(outdir)+1+strlen(pfx)+6+strlen(dispdtype)) * sizeof(char))
#                 strcpy(outdisp, outdir)
#                 strcat(outdisp, '/')
#                 strcat(outdisp, pfx)
#                 strcat(outdisp, '.ph.disp')
#                 self.data.dispR.writedisptxt(outfname=outdisp, dtype='ph')
#                 free(outdisp)
#                 outdisp = <char *>malloc((strlen(outdir)+1+strlen(pfx)+6+strlen(dispdtype)) * sizeof(char))
#                 strcpy(outdisp, outdir)
#                 strcat(outdisp, '/')
#                 strcat(outdisp, pfx)
#                 strcat(outdisp, '.gr.disp')
#                 self.data.dispR.writedisptxt(outfname=outdisp, dtype='gr')
#                 free(outdisp)
#         with gil:
#             outrf = <char *>malloc((strlen(outdir)+1+strlen(pfx)+3) * sizeof(char))
#             strcpy(outrf, outdir)
#             strcat(outrf, '/')
#             strcat(outrf, pfx)
#             strcat(outrf, '.rf')
#             self.data.rfr.writerftxt(outfname=outrf)
#             free(outrf)
#         # convert initial model to para
#         self.model.isomod.mod2para()
#         # likelihood/misfit
#         oldL        = self.data.L
#         oldmisfit   = self.data.misfit
#         printf('Initial likelihood = %f,' , oldL)
#         printf('misfit = %f\n', oldmisfit)
#         
#         inew    = 0     # count step (or new paras)
#         iacc    = 1     # count acceptance model
#         cdef time_t t0 = time(NULL)
#         cdef time_t t1 
#         self.newisomod.get_mod(self.model.isomod)
# #        newmod.get_mod(self.model.isomod)
#         while ( run==1 ):
#             inew    += 1
# #            printf('run step = %d\n',inew)
#             t1      = time(NULL)
#             # # # if ( inew > 100000 or iacc > 20000000 or time.time()-start > 7200.):
#             if ( inew > 10000 or iacc > 20000000):
#                 run = 0
#             if (fmod(inew, 500) == 0):
#                 printf('run step = %d,',inew)
#                 printf(' elasped time = %d', t1-t0)
#                 printf(' sec\n')
#             #------------------------------------------------------------------------------------------
#             # every 2500 step, perform a random walk with uniform random value in the paramerter space
#             #------------------------------------------------------------------------------------------
#             if ( fmod(inew, 1501) == 1500 ):
#                 self.newisomod.get_mod(self.model.isomod)
#                 self.newisomod.para.new_paraval(0)
#                 self.newisomod.para2mod()
#                 self.newisomod.update()
#                 # loop to find the "good" model,
#                 # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
#                 igood   = 0
#                 while ( self.newisomod.isgood(0, 1, 1, 0) == 0):
#                     igood   += igood + 1
#                     self.newisomod.get_mod(self.model.isomod)
#                     self.newisomod.para.new_paraval(0)
#                     self.newisomod.para2mod()
#                     self.newisomod.update()
#                 # assign new model to old ones
#                 self.model.isomod.get_mod(self.newisomod)
#                 self.get_vmodel()
#                 # forward computation
#                 self.compute_fsurf()
#                 self.compute_rftheo()
#                 self.get_misfit(wdisp, rffactor)
#                 oldL                = self.data.L
#                 oldmisfit           = self.data.misfit
#                 iacc                += 1
#                 printf('Uniform random walk: likelihood = %f', self.data.L)
#                 printf(' misfit = %f\n', self.data.misfit)
#             #-------------------------------
#             # inversion part
#             #-------------------------------
#             # sample the posterior distribution ##########################################
#             if (wdisp >= 0 and wdisp <=1):
#                 self.newisomod.get_mod(self.model.isomod)
#                 self.newisomod.para.new_paraval(1)
#                 self.newisomod.para2mod()
#                 self.newisomod.update()
#                 if monoc:
#                     # loop to find the "good" model,
#                     # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
#                     if not self.newisomod.isgood(0, 1, 1, 0):
#                         continue
#                 # assign new model to old ones
#                 self.oldisomod.get_mod(self.model.isomod)
#                 self.model.isomod.get_mod(self.newisomod)
#                 self.get_vmodel()
#                 # forward computation
#                 self.compute_fsurf()
#                 self.compute_rftheo()
#                 self.get_misfit(wdisp, rffactor)
#                 newL                = self.data.L
#                 newmisfit           = self.data.misfit
#                 # 
#                 if newmisfit > oldmisfit:
#                     rnumb   = random_uniform(0., 1.)
#                     if oldL == 0.:
#                         continue                        
#                     prob    = (oldL-newL)/oldL
#                     # reject the model
#                     if rnumb < prob:
# #                        fidout.write("-1 %d %d " % (inew,iacc))
# #                        for i in xrange(newmod.para.npara):
# #                            fidout.write("%g " % newmod.para.paraval[i])
# #                        fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.rfr.L, self.indata.rfr.misfit,\
# #                                self.indata.dispR.L, self.indata.dispR.misfit, time.time()-start)) 
# #                        
#                         outArr[0][inew-1]   = -1.
#                         outArr[1][inew-1]   = inew
#                         outArr[2][inew-1]   = iacc
#                         for i in range(13):
#                             outArr[3+i][inew-1] = self.newisomod.para.paraval[i]
#                         outArr[16][inew-1]      = newL
#                         outArr[17][inew-1]      = newmisfit
#                         outArr[18][inew-1]      = self.data.rfr.L
#                         outArr[19][inew-1]      = self.data.rfr.misfit
#                         outArr[20][inew-1]      = self.data.dispR.L
#                         outArr[21][inew-1]      = self.data.dispR.misfit
#                         outArr[22][inew-1]      = time(NULL)-t0
#                         
#                         self.model.isomod.get_mod(self.oldisomod)
#                         continue
#                 # accept the new model
# #                fidout.write("1 %d %d " % (inew,iacc))
# #                for i in xrange(newmod.para.npara):
# #                    fidout.write("%g " % newmod.para.paraval[i])
# #                fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.rfr.L, self.indata.rfr.misfit,\
# #                        self.indata.dispR.L, self.indata.dispR.misfit, time.time()-start)) 
# #                
#                 outArr[0][inew-1]   = -1.
#                 outArr[1][inew-1]   = inew
#                 outArr[2][inew-1]   = iacc
#                 for i in range(13):
#                     outArr[3+i][inew-1] = self.newisomod.para.paraval[i]
#                 outArr[16][inew-1]      = newL
#                 outArr[17][inew-1]      = newmisfit
#                 outArr[18][inew-1]      = self.data.rfr.L
#                 outArr[19][inew-1]      = self.data.rfr.misfit
#                 outArr[20][inew-1]      = self.data.dispR.L
#                 outArr[21][inew-1]      = self.data.dispR.misfit
#                 outArr[22][inew-1]      = time(NULL)-t0
#                 
#                 printf('Accept a model: %d', inew)
#                 printf(' %d ', iacc)
#                 printf(' %f ', oldL)
#                 printf(' %f ', oldmisfit)
#                 printf(' %f ', newL)
#                 printf(' %f ', newmisfit)
#                 printf(' %f ', self.data.rfr.L)
#                 printf(' %f ', self.data.rfr.misfit)
#                 printf(' %f ', self.data.dispR.L)
#                 printf(' %f ', self.data.dispR.misfit)
#                 printf(' %f\n', time(NULL)-t0)
#                 # write accepted model
# #                outmod      = outdir+'/'+pfx+'.%d.mod' % iacc
# #                vmodel.write_model(model=self.model, outfname=outmod, isotropic=True)
# #                # write corresponding data
# #                if dispdtype != 'both':
# #                    outdisp = outdir+'/'+pfx+'.'+dispdtype+'.%d.disp' % iacc
# #                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
# #                else:
# #                    outdisp = outdir+'/'+pfx+'.ph.%d.disp' % iacc
# #                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='ph')
# #                    outdisp = outdir+'/'+pfx+'.gr.%d.disp' % iacc
# #                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='gr')
# #                # # outdisp = outdir+'/'+pfx+'.%d.disp' % iacc
# #                # # data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
# #                outrf   = outdir+'/'+pfx+'.%d.rf' % iacc
# #                data.writerftxt(outfname=outrf, outrf=self.indata.rfr)
#                 # assign likelihood/misfit
#                 oldL        = newL
#                 oldmisfit   = newmisfit
#                 iacc        += 1
#                 continue
# #            else:
# #                if monoc:
# #                    newmod  = self.model.isomod.copy()
# #                    newmod.para.new_paraval(1)
# #                    newmod.para2mod()
# #                    newmod.update()
# #                    if not newmod.isgood(0, 1, 1, 0):
# #                        continue
# #                else:
# #                    newmod  = self.model.isomod.copy()
# #                    newmod.para.new_paraval(0)
# #                fidout.write("-2 %d 0 " % inew)
# #                for i in xrange(newmod.para.npara):
# #                    fidout.write("%g " % newmod.para.paraval[i])
# #                fidout.write("\n")
# #                self.model.isomod   = newmod
# #                continue
#         return
#     
#     def mc_inv_iso_interface(self, char *outdir='./workingdir_iso', char *pfx='MC',\
#                 char *dispdtype='ph', float wdisp=0.2, float rffactor=40., int monoc=1):
#         if not os.path.isdir(outdir):
#             os.makedirs(outdir)
# #        cdef char *outdir
# #        cdef char *pfx
# #        cdef int l = strlen(pfx) 
# #        printf('%s', pfx)
# #        printf('%d', l)
# #        outdir = './work'
# #        pfx  = 'MC'
#         self.mc_inv_iso(outdir, pfx, dispdtype)
# #        self.mc_inv_iso()
# 
# 
# def mcinviso4mp(iArr, invsolver1d solver, str outdir, str pfx, str dispdtype, \
#                     float wdisp, float rffactor, int monoc):
#     print iArr[2], iArr[1]
#     solver.mc_inv_iso_singel_thread(outdir=outdir, ind0=iArr[0], ind1=iArr[1], indid=iArr[2], pfx=pfx, dispdtype=dispdtype, wdisp=wdisp,\
#                                     rffactor=rffactor, monoc=monoc)
#     return
    

    
    
        
    
    
    
    
    

    
    


    