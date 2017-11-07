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
        
    def mc_inv_iso(self, outdir='./workingdir', dispdtype='ph', wdisp=1., rffactor=40., monoc=True):
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
        outmod  = outdir+'/MC.0.mod'
        vmodel.write_model(model=self.model, outfname=outmod, isotropic=True)
        # write initial predicted data
        if dispdtype != 'both':
            outdisp = outdir+'/MC.0.'+dispdtype+'.disp'
            data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
        else:
            outdisp = outdir+'/MC.0.ph.disp'
            data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='ph')
            outdisp = outdir+'/MC.0.gr.disp'
            data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype='gr')
        outrf   = outdir+'/MC.0.rf'
        data.writerftxt(outfname=outrf, outrf=self.indata.rfr)
        # conver initial model to para
        self.model.isomod.mod2para()
        # likelihood/misfit
        oldL        = self.indata.L
        oldmisfit   = self.indata.misfit
        print "Initial likelihood = ", oldL, ' misfit =',oldmisfit
        
        run     = True     # the key that controls the sampling
        inew    = 0     # count step (or new paras)
        iacc    = 0     # count acceptance model
        start   = time.time()
        # output log files
        outtxtfname = outdir+'/MC.out'
        outbinfname = outdir+'/MC.bin'
        fidout      = open(outtxtfname, "w")
        # fidoutb     = open(outbinfname, "wb")
        pfx = 'MC.'
        while ( run ):
            inew+= 1
            if ( inew > 10000 or iacc > 2000 or time.time()-start > 3600.):
                run   = False
            if (np.fmod(inew, 500) == 0):
                print 'step =',inew, time.time()-start
            #------------------------------------------------------------------------------------------
            # every 1500 step, perform a random walk with uniform random value in the paramerter space
            #------------------------------------------------------------------------------------------
            if ( np.fmod(inew, 1501) == 1500 ):
                newmod  = self.model.isomod.copy()
                newmod.para.new_paraval(0)
                newmod.para2mod()
                newmod.update()
                
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
                print "new para!!", self.indata.L, self.indata.misfit
            
            
            ######################### do inversion#####################################
            # sample the posterior distribution ##########################################
            if (wdisp >= 0 and wdisp <=1):
                newmod  = self.model.isomod.copy()
                newmod.para.new_paraval(1)
                newmod.para2mod()
                newmod.update()
                if monoc:
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
                # accept because oldL < newL
                fidout.write("1 %d %d " % (inew,iacc))
                for i in xrange(newmod.para.npara):
                    fidout.write("%g " % newmod.para.paraval[i])
                fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.rfr.L, self.indata.rfr.misfit,\
                        self.indata.dispR.L, self.indata.dispR.misfit, time.time()-start))        
                print "accept!! ", inew, iacc, oldL, newL, self.indata.rfr.L, self.indata.rfr.misfit,\
                                self.indata.dispR.L, self.indata.dispR.misfit, time.time()-start
                # write accepted model
                outmod      = outdir+'/'+pfx+'.%d.mod' % iacc
                vmodel.write_model(model=self.model, outfname=outmod, isotropic=True)
                # write corresponding data
                outdisp = outdir+'/MC.p1.%d.disp' % iacc
                data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
                outrf   = outdir+'/MC.p1.%d.rf' % iacc
                data.writerftxt(outfname=outrf, outrf=self.indata.rfr)
                
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
        
        
        
    
    # def init_fwrd_compute(self, mtype='iso'):
        
        
            
    
    