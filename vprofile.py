# -*- coding: utf-8 -*-
"""
Module for 1D profile inversion

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

import numpy as np
import vmodel, data, modparam
import fast_surf, theo
import warnings
import os
import time
import random



class vprofile1d(object):
    """
    An object for 1D velocity profile inversion
    =====================================================================================================================
    ::: parameters :::
    indata      - object storing input data
    model       - object storing 1D model
    =====================================================================================================================
    """
    def __init__(self):
        self.model  = vmodel.model1d()
        self.indata = data.data1d()
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
            data.readdisptxt(infname=infname, indisp=self.indata.dispR, dtype=dtype)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            data.readdisptxt(infname=infname, indisp=self.indata.dispL, dtype=dtype)
        else:
            raise ValueError('Unexpected wave type: '+wtype)
        return
    
    def readaziamp(self, infname, dtype='ph', wtype='ray'):
        """
        read azimuthal amplitude data from a txt file
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
            data.readaziamptxt(infname=infname, indisp=self.indata.dispR, dtype=dtype)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            data.readaziamptxt(infname=infname, indisp=self.indata.dispL, dtype=dtype)
        else:
            raise ValueError('Unexpected wave type: '+wtype)
        return
    
    def readaziphi(self, infname, dtype='ph', wtype='ray'):
        """
        read fast direction azimuth data from a txt file
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
            data.readaziphitxt(infname=infname, indisp=self.indata.dispR, dtype=dtype)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            data.readaziphitxt(infname=infname, indisp=self.indata.dispL, dtype=dtype)
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
            data.readrftxt(infname=infname, inrf=self.indata.rfr)
        elif dtype=='t' or dtype == 'transverse':
            data.readrftxt(infname=infname, inrf=self.indata.rft)
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
        if mtype=='iso' or mtype == 'isotropic':
            modparam.readmodtxt(infname=infname, inmod=self.model.isomod)
        elif mtype=='tti':
            modparam.readtimodtxt(infname=infname, inmod=self.model.ttimod)
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
            modparam.readparatxt(infname=infname, inpara=self.model.isomod.para)
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def getpara(self, mtype='iso'):
        mtype   = mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            self.model.isomod.get_paraind()
        elif mtype=='tti':
            self.model.ttimod.get_paraind_US()
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def update_mod(self, mtype='iso'):
        """
        update model from model parameters
        =====================================================================
        ::: input :::
        mtype       - model type (isotropic or tti)
        =====================================================================
        """
        mtype   = mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            warnings.filterwarnings("ignore")
            self.model.isomod.update()
        elif mtype=='tti':
            self.model.ttimod.update()
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def get_vmodel(self, mtype='iso'):
        """
        get the velocity model arrays
        =====================================================================
        ::: input :::
        mtype       - model type (isotropic or tti)
        =====================================================================
        """
        mtype   = mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            hArr, vs, vp, rho, qs, qp = self.model.get_iso_vmodel()
            self.hArr   = np.append(hArr, 0.)
            self.vsArr  = np.append(vs, vs[-1])
            self.vpArr  = np.append(vp, vp[-1])
            self.vpvsArr= self.vpArr/self.vsArr
            self.rhoArr = np.append(rho, rho[-1])
            self.qsArr  = np.append(qs, qs[-1])
            self.qpArr  = np.append(qp, qp[-1])
            self.qsinv  = 1./self.qsArr
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def get_period(self, dtype='ph'):
        """
        get the period for surface wave inversion
        =====================================================================
        ::: input :::
        dtype       - data type (phase or group)
        =====================================================================
        """
        dtype   = dtype.lower()
        # Rayleigh wave
        TR                          = np.array(list(set.union(set(self.indata.dispR.pper), set(self.indata.dispR.gper))), dtype=np.float32)
        TR                          = np.sort(TR)
        self.indata.dispR.period    = TR
        self.indata.dispR.nper      = TR.size
        # Love wave
        TL                          = np.array(list(set.union(set(self.indata.dispL.pper), set(self.indata.dispL.gper))), dtype=np.float32)
        TL                          = np.sort(TL)
        self.indata.dispL.period    = TL
        self.indata.dispL.nper      = TL.size
        
        if dtype == 'ph' or dtype =='phase':
            self.TR = self.indata.dispR.pper.copy()
            self.TL = self.indata.dispL.pper.copy()
        elif dtype == 'gr' or dtype =='group':
            self.TR = self.indata.dispR.gper.copy()
            self.TL = self.indata.dispL.gper.copy()
        elif dtype == 'both':
            try:
                if not np.allclose(self.indata.dispR.pper, self.indata.dispR.gper):
                    raise ValueError ('Unconsistent phase/group period arrays for Rayleigh wave!')
            except:
                raise ValueError ('Unconsistent phase/group period arrays for Rayleigh wave!')
                
            try:
                if not np.allclose(self.indata.dispL.pper, self.indata.dispL.gper):
                    raise ValueError ( 'Unconsistent phase/group period arrays for Love wave!')
            except:
                raise ValueError ( 'Unconsistent phase/group period arrays for Love wave!')
            self.TR = self.indata.dispR.pper.copy()
            self.TL = self.indata.dispL.pper.copy()
        self.surfdtype  = dtype
        return
    
    def get_rf_param(self):
        """
        get fs and npts for receiver function
        """
        self.fs     = max(self.indata.rfr.fs, self.indata.rft.fs)
        self.npts   = max(self.indata.rfr.npts, self.indata.rft.npts)
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
        if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
            ilvry               = 2
            nper                = self.TR.size
            per                 = np.zeros(200, dtype=np.float32)
            per[:nper]          = self.TR[:]
            (ur0,ul0,cr0,cl0)   = fast_surf.fast_surf(self.vsArr.size, ilvry, \
                                    self.vpArr, self.vsArr, self.rhoArr, self.hArr, self.qsinv, per, nper)
            self.indata.dispR.pvelp     = cr0[:nper]
            self.indata.dispR.gvelp     = ur0[:nper]
            
        elif wtype=='l' or wtype == 'love':
            ilvry               = 1
            nper                = self.TL.size
            per                 = np.zeros(200, dtype=np.float32)
            per[:nper]          = self.TL[:]
            (ur0,ul0,cr0,cl0)   = fast_surf.fast_surf(self.vsArr.size, ilvry, \
                                    self.vpArr, self.vsArr, self.rhoArr, self.hArr, self.qsinv, per, nper)
            self.indata.dispL.pvelp     = cl0[:nper]
            self.indata.dispL.gvelp     = ul0[:nper]
        return
    
    def compute_rftheo(self, dtype='r', slowness = 0.06, din=None):
        """
        compute receiver function of isotropic model using theo
        ===========================================================================================
        ::: input :::
        dtype   - data type (radial or trnasverse)
        slowness- reference horizontal slowness (default - 0.06 s/km, 1./0.06=16.6667)
        din     - incident angle in degree (default - None, din will be computed from slowness)
        ===========================================================================================
        """
        dtype   = dtype.lower()
        if dtype=='r' or dtype == 'radial':
            # initialize input model arrays
            hin         = np.zeros(100, dtype=np.float32)
            vsin        = np.zeros(100, dtype=np.float32)
            vpvs        = np.zeros(100, dtype=np.float32)
            qsin        = 600.*np.ones(100, dtype=np.float32)
            qpin        = 1400.*np.ones(100, dtype=np.float32)
            # assign model arrays to the input arrays
            if self.hArr.size<100:
                nl      = self.hArr.size
            else:
                nl      = 100
            hin[:nl]    = self.hArr
            vsin[:nl]   = self.vsArr
            vpvs[:nl]   = self.vpvsArr
            qsin[:nl]   = self.qsArr
            qpin[:nl]   = self.qpArr
            # fs/npts
            fs          = self.fs
            # # # ntimes      = 1000
            ntimes      = self.npts
            # incident angle
            if din is None:
                din     = 180.*np.arcsin(vsin[nl-1]*vpvs[nl-1]*slowness)/np.pi
            # solve for receiver function using theo
            rx 	                = theo.theo(nl, vsin, hin, vpvs, qpin, qsin, fs, din, 2.5, 0.005, 0, ntimes)
            # store the predicted receiver function (ONLY radial component) to the data object
            self.indata.rfr.rfp = rx[:self.npts]
            self.indata.rfr.tp  = np.arange(self.npts, dtype=np.float32)*1./self.fs
        # elif dtype=='t' or dtype == 'transverse':
        #     
        else:
            raise ValueError('Unexpected receiver function type: '+dtype)
        return
    
    def get_misfit(self, wdisp=1., rffactor=40.):
        """
        compute data misfit
        =====================================================================
        ::: input :::
        wdisp       - weight for dispersion curves (0.~1., default - 1.)
        rffactor    - downweighting factor for receiver function
        =====================================================================
        """
        self.indata.get_misfit(wdisp, rffactor)
        return
        
    def mc_inv_iso(self, outdir='./workingdir', dispdtype='ph', wdisp=.714, rffactor=40., monoc=True, pfx='MC'):
        """
        
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # initializations
        self.get_period(dtype = dispdtype)
        self.update_mod(mtype = 'isotropic')
        self.get_rf_param()
        self.get_vmodel(mtype = 'isotropic')
        # initial run
        self.compute_fsurf()
        self.compute_rftheo()
        self.get_misfit(wdisp=wdisp, rffactor=rffactor)
        # write initial model
        outmod  = outdir+'/'+pfx+'.mod'
        vmodel.write_model(model=self.model, outfname=outmod, isotropic=True)
        # write initial predicted data
        if dispdtype != 'both':
            outdisp = outdir+'/'+pfx+'.'+dispdtype+'.disp'
            data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
        else:
            outdisp = outdir+'/'+pfx+'.ph.disp'
            data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='ph')
            outdisp = outdir+'/'+pfx+'.gr.disp'
            data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='gr')
        outrf   = outdir+'/'+pfx+'.rf'
        data.writerftxt(outfname=outrf, outrf=self.indata.rfr)
        # conver initial model to para
        self.model.isomod.mod2para()
        # likelihood/misfit
        oldL        = self.indata.L
        oldmisfit   = self.indata.misfit
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
            print 'run step = ',inew
            if ( inew > 10000 or iacc > 2000 or time.time()-start > 3600.):
                run   = False
            if (np.fmod(inew, 500) == 0):
                print 'step =',inew, 'elasped time =', time.time()-start, ' sec'
            #------------------------------------------------------------------------------------------
            # every 2500 step, perform a random walk with uniform random value in the paramerter space
            #------------------------------------------------------------------------------------------
            if ( np.fmod(inew, 2501) == 2500 ):
                newmod  = self.model.isomod.copy()
                newmod.para.new_paraval(0)
                newmod.para2mod()
                newmod.update()
                # loop to find the "good" model,
                # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                igood   = 0
                while ( not newmod.isgood(0, 1, 1, 0)):
                    igood   += igood + 1
                    newmod  = self.model.isomod.copy()
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
                oldL                = self.indata.L
                oldmisfit           = self.indata.misfit
                iacc                += 1
                print 'Uniform random walk: likelihood =', self.indata.L, 'misfit =',self.indata.misfit
            #-------------------------------
            # inversion part
            #-------------------------------
            # sample the posterior distribution ##########################################
            if (wdisp >= 0 and wdisp <=1):
                newmod  = self.model.isomod.copy()
                newmod.para.new_paraval(1)
                newmod.para2mod()
                newmod.update()
                if monoc:
                    # loop to find the "good" model,
                    # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                    if not newmod.isgood(0, 1, 1, 0):
                        continue
                # assign new model to old ones
                oldmod              = self.model.isomod.copy()
                self.model.isomod   = newmod
                self.get_vmodel()
                # forward computation
                self.compute_fsurf()
                self.compute_rftheo()
                self.get_misfit(wdisp=wdisp, rffactor=rffactor)
                newL                = self.indata.L
                newmisfit           = self.indata.misfit
                # 
                if newL < oldL:
                    prob    = (oldL-newL)/oldL
                    rnumb   = random.random()
                    # reject the model
                    if rnumb < prob:
                        fidout.write("-1 %d %d " % (inew,iacc))
                        for i in xrange(newmod.para.npara):
                            fidout.write("%g " % newmod.para.paraval[i])
                        fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.rfr.L, self.indata.rfr.misfit,\
                                self.indata.dispR.L, self.indata.dispR.misfit, time.time()-start))        
                        ### ttmodel.writeb (para1, ffb,[-1,i,ii])
                        # return to oldmod
                        self.model.isomod   = oldmod
                        continue
                # accept the new model
                fidout.write("1 %d %d " % (inew,iacc))
                for i in xrange(newmod.para.npara):
                    fidout.write("%g " % newmod.para.paraval[i])
                fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.rfr.L, self.indata.rfr.misfit,\
                        self.indata.dispR.L, self.indata.dispR.misfit, time.time()-start))        
                print "Accept a model", inew, iacc, oldL, newL, self.indata.rfr.L, self.indata.rfr.misfit,\
                                self.indata.dispR.L, self.indata.dispR.misfit, time.time()-start
                # write accepted model
                outmod      = outdir+'/'+pfx+'.%d.mod' % iacc
                vmodel.write_model(model=self.model, outfname=outmod, isotropic=True)
                # write corresponding data
                if dispdtype != 'both':
                    outdisp = outdir+'/'+pfx+'.'+dispdtype+'.%d.disp' % iacc
                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
                else:
                    outdisp = outdir+'/'+pfx+'.ph.%d.disp' % iacc
                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='ph')
                    outdisp = outdir+'/'+pfx+'.gr.%d.disp' % iacc
                    data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='gr')
                # # outdisp = outdir+'/'+pfx+'.%d.disp' % iacc
                # # data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
                outrf   = outdir+'/'+pfx+'.%d.rf' % iacc
                data.writerftxt(outfname=outrf, outrf=self.indata.rfr)
                # assign likelihood/misfit
                oldL        = newL
                oldmisfit   = newmisfit
                iacc        += 1
                continue
            else:
                if monoc:
                    newmod  = self.model.isomod.copy()
                    newmod.para.new_paraval(1)
                    newmod.para2mod()
                    newmod.update()
                    if not newmod.isgood(0, 1, 1, 0):
                        continue
                else:
                    newmod  = self.model.isomod.copy()
                    newmod.para.new_paraval(0)
                fidout.write("-2 %d 0 " % inew)
                for i in xrange(newmod.para.npara):
                    fidout.write("%g " % newmod.para.paraval[i])
                fidout.write("\n")
                self.model.isomod   = newmod
                continue
        fidout.close()
        return
    
    # def mc_inv_iso(self, outdir='./workingdir', dispdtype='ph', wdisp=.714, rffactor=40., monoc=True, pfx='MC'):
    
    #-------------------------------------------------
    # post-processing functions
    #-------------------------------------------------
    def read_iso_inv(self, indir, npara=13, pfx='MC'):
        try:
            npara   = self.model.isomod.para.npara
        except:
            print 'Using npara =',npara
        infname     = indir+'/'+pfx+'.out'
        inArr       = np.loadtxt(infname)
        self.isacc  = inArr[:, 0]
        self.inew   = inArr[:, 1]
        self.iacc   = inArr[:, 2]
        self.paraval= np.zeros([self.isacc.size, npara])
        for i in xrange(npara):
            self.paraval[:, i]  = inArr[:, 3+i]
        self.L          = inArr[:, 3+npara]
        self.misfit     = inArr[:, 4+npara]
        self.rfL        = inArr[:, 5+npara]
        self.rfmisfit   = inArr[:, 6+npara]
        self.dispL      = inArr[:, 7+npara]
        self.dispmisfit = inArr[:, 8+npara]
        self.npara      = npara
        return
    
    def get_min_iso_mod(self):
        ind                             = self.misfit.argmin()
        self.model.isomod.para.paraval  = np.float32(self.paraval[ind, :])
        self.model.isomod.para2mod()
        self.model.isomod.update()
        self.get_vmodel(mtype='iso')
        return
    
    def get_avg_iso_mod(self, threshhold=2.0, mtype='rel'):
        minmisfit                       = self.misfit.min()
        if threshhold < 1.:
            raise ValueError('Relative threshhold should be larger than 1!')
        if tmisfit  >= 0.5:
            tmisfit = threshhold*minmisfit
        else:
            tmisfit = minmisfit + 0.5
        ind                             = (self.misfit <= tmisfit)
        self.model.isomod.para.paraval  = np.mean(self.paraval[ind, :], axis = 0, dtype=np.float32)
        self.model.isomod.para2mod()
        self.model.isomod.update()
        self.get_vmodel(mtype='iso')
        return
    
    def get_avg_iso_mod_2(self, threshhold=2.0, mtype='rel'):
        minmisfit                       = self.misfit.min()
        if mtype == 'rel':
            if threshhold < 1.:
                raise ValueError('Relative threshhold should be larger than 1!')
            tmisfit = threshhold*minmisfit
        else:
            tmisfit = threshhold + minmisfit
        ind                             = (self.misfit <= tmisfit)
        self.model.isomod.para.paraval  = np.mean(self.paraval[ind, :], axis = 0, dtype=np.float32)
        self.model.isomod.para2mod()
        self.model.isomod.update()
        self.get_vmodel(mtype='iso')
        return
        
        
        
        
        
    
    # def init_fwrd_compute(self, mtype='iso'):
        
        
            
    
    