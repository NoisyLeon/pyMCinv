# -*- coding: utf-8 -*-
"""
Module for 1D profile inversion

Numba is used for speeding up of the code.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

import numpy as np
import vmodel, data, modparam
import fast_surf, theo



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
    
    def update_mod(self, mtype='iso'):
        mtype=mtype.lower()
        if mtype=='iso' or mtype == 'isotropic':
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
            self.rhoArr = np.append(rho, rho[-1])
            self.qsArr  = np.append(qs, qs[-1])
            self.qpArr  = np.append(qp, qp[-1])
            self.qsinv  = 1./self.qsArr
        else:
            raise ValueError('Unexpected wave type: '+mtype)
        return
    
    def get_period(self):
        
        TR                          = np.array(list(set.union(set(self.indata.dispR.pper), set(self.indata.dispR.gper))), dtype=np.float32)
        TR                          = np.sort(TR)
        self.indata.dispR.period    = TR
        self.indata.dispR.nper      = TR.size
        
        TL                          = np.array(list(set.union(set(self.indata.dispL.pper), set(self.indata.dispL.gper))), dtype=np.float32)
        TL                          = np.sort(TL)
        self.indata.dispL.period    = TL
        self.indata.dispL.nper      = TL.size
        self.TR = TR.copy()
        self.TL = TL.copy()
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
        
    
    # def compute_rftheo(self):
        
    
    # def init_fwrd_compute(self, mtype='iso'):
        
        
            
    
    