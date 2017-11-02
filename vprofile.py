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



class vprofile1d(object):
    def __init__(self):
        self.model  = vmodel.model1d()
        self.indata = data.data1d()
        return
    
    def readdisp(self, infname, dtype='ph', wtype='ray'):
        dtype=dtype.lower()
        wtype=wtype.lower()
        if wtype=='ray' or wtype=='rayleigh' or wtype=='r':
            data.readdisptxt(infname=infname, indisp=self.indata.dispR, dtype=dtype)
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            data.readdisptxt(infname=infname, indisp=self.indata.dispL, dtype=dtype)
        else:
            raise ValueError('Unexpected wave type: '+wtype)
        return
    
    def readrf(self, infname, dtype='R'):
        dtype=dtype.lower()
        if dtype=='r' or dtype == 'radial':
            data.readrftxt(infname=infname, inrf=self.indata.rfr)
        elif dtype=='t' or dtype == 'transverse':
            data.readrftxt(infname=infname, inrf=self.indata.rft)
        else:
            raise ValueError('Unexpected wave type: '+dtype)
        return
    
    def readmod(self, infname, mtype='iso'):
        mtype=mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            modparam.readmodtxt(infname=infname, inmod=self.model.isomod)
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def readpara(self, infname, mtype='iso'):
        mtype=mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            modparam.readparatxt(infname=infname, inpara=self.model.isomod.para)
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def update_mod(self, mtype='iso'):
        mtype=mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
            warnings.filterwarnings("ignore")
            self.model.isomod.update()
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def get_vmodel(self, mtype='iso'):
        mtype=mtype.lower()
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
        
        TR                          = np.array(list(set.union(set(self.indata.dispR.pper), set(self.indata.dispR.gper))), dtype=np.float32)
        TR                          = np.sort(TR)
        self.indata.dispR.period    = TR
        self.indata.dispR.nper      = TR.size
        
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
        self.fs     = max(self.indata.rfr.fs, self.indata.rft.fs)
        self.npts   = max(self.indata.rfr.npts, self.indata.rft.npts)
        return
    
    def compute_fsurf(self, wtype='ray'):
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
        dtype=dtype.lower()
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
        self.indata.get_misfit(wdisp, rffactor)
        return
        
    def mc_inv_iso(self, outdir='./workingdir', dispdtype='ph', wdisp=1., rffactor=40.):
        
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # initializations
        self.get_period(dtype = dispdtype)
        self.update_mod(mtype = 'isotropic')
        self.get_rf_param()
        self.get_vmodel(mtype = 'isotropic')
        # first run
        self.compute_fsurf()
        self.compute_rftheo()
        self.get_misfit(wdisp=wdisp, rffactor=rffactor)
        
        outmod  = outdir+'/MC.p1.mod'
        vmodel.write_model(model=self.model, outfname=outmod, isotropic=True)
        
        outdisp = outdir+'/MC.p1.p.disp'
        data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispR, dtype=dispdtype)
        outrf   = outdir+'/MC.p1.rf'
        data.writerftxt(outfname=outrf, outrf=self.indata.rfr)
        
        self.model.isomod.mod2para()
        
        print "Original misfit: ", self.indata.L, self.indata.misfit
        
        run     = True     # the key that controls the sampling
        inew    = 0     # count new paras
        iacc    = 0     # count acceptance model
        start   = time.time()
        # nparaval= 
        # 
        # while ( run ):
        #     inew+= 1
        # 
        #     if ( inew > 10000 or iacc > 2000 or time.time()-start > 3600.):
        #         run   = False
        #         
        #     if (np.fmod(inew,500) ==0):
        #         print inew, time.time()-start
        #         
        #     if ( np.fmod(inew,1501) == 1500 ):
        #         
        #         # continue here, need to keep a copy of old para until a good model is found
                # nparaval= self.model.isomod.para.new_paraval(0)
                # 
                # 
                # para1   = para.new_para(0)
        #         ttmodel = model.para2mod(para1)
        #         ttmodel.update()
        #         iii     = 0
        #         while (ttmodel.goodmodel([0,1],[]) == 0):
        #             iii = iii + 1
        #             para1 = para.new_para(0)
        #             ttmodel = model.para2mod(para1)
        #             ttmodel.update()
        #         ttmodel.compute_rf()
        #         ttmodel.compute_disp()
        #         ttmodel.compute_misfit(pp,nn)
        #         oldL = ttmodel.data.L
        #         oldmisfit = ttmodel.data.misfit
        #         para = para1
        #         ii = ii + 1
        #         print para.parameter
        #         print "new para!!", oldL, oldmisfit
            
            # 
            # ############################ do inversion#####################################
            # # sample the posterior distribution ##########################################
            # if (pp >= 0 and pp <=1):
            #     para1 = para.new_para(1)
            #     ttmodel = model.para2mod(para1)
            #     ttmodel.update()
            #     if (monoc == 1):
            #         newL = 0.;
            #         newmisfit = 100;
            #         if (ttmodel.goodmodel([0,1],[]) == 0):
            #             continue
            #     (newL,newmisfit,ttmodel) = get_misfit(para1,model,pp,nn)
            #     if (newL < oldL):
            #         prob = (oldL-newL)/oldL
            #         cvt = random.random()
            #         # reject
            #         if (cvt<prob):
            #             ff.write("-1 %d %d " % (i,ii))
            #             for j in range (para1.npara):
            #                 ff.write("%g " % para1.parameter[j])
            #             ff.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, ttmodel.data.rf.L, ttmodel.data.rf.misfit, ttmodel.data.disp.L,
            #                                                 ttmodel.data.disp.misfit, time.time()-start))
            #             ttmodel.writeb (para1, ffb,[-1,i,ii])
            #             continue
            #     ff.write("1 %d %d " % (i,ii))
            #     for j in range (para1.npara):
            #         ff.write("%g " % para1.parameter[j])
            #     ff.write("%g %g %g %g %g %g %g\n" % (newL,newmisfit,ttmodel.data.rf.L,ttmodel.data.rf.misfit,ttmodel.data.disp.L,ttmodel.data.disp.misfit,time.time()-start));
            #     print "accept!! ", i, ii, oldL, newL, ttmodel.data.rf.L, ttmodel.data.rf.misfit, ttmodel.data.disp.L, \
            #         ttmodel.data.disp.misfit, time.time()-start
            #     tname1 = tname + ".%d" % ii
            #     ttmodel.write_model(tname1, outdir)
            #     ttmodel.writeb (para1,ffb,[1,i,ii])
            #     para = para1
            #     oldL = newL
            #     oldmisfit = newmisfit
            #     ii = ii + 1
            #     continue
            # else:
            #     if (monoc == 1):
            #         para1 = para.new_para(1)
            #         ttmodel = model.para2mod(para1)
            #         ttmodel.update()
            #         if (ttmodel.goodmodel([0,1],[]) == 0):
            #             continue
            #     else:
            #         para1 = para.new_para(0)
            #     ff.write("-2 %d 0 " % i)
            #     for j in range (para1.npara):
            #         ff.write("%g " % para1.parameter[j])
            #     ff.write("\n")
            #     para = para1
            #     continue
        
        
    
    # def init_fwrd_compute(self, mtype='iso'):
        
        
            
    
    