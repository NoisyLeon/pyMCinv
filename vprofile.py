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
import tdisp96, tregn96, tlegn96
import warnings
import os
import time
import random
import eigenkernel
# import fastmc
import matplotlib.pyplot as plt



class vprofile1d(object):
    """
    An object for 1D velocity profile inversion
    =====================================================================================================================
    ::: parameters :::
    indata              - object storing input data
    model               - object storing 1D model
    eigkR, eigkL        - eigenkernel objects storing Rayleigh/Love eigenfunctions and sensitivity kernels
    hArr                - layer array 
    disprefR, disprefL  - flags indicating existence of sensitivity kernels for reference model
    =====================================================================================================================
    """
    def __init__(self):
        self.model      = vmodel.model1d()
        self.indata     = data.data1d()
        self.eigkR      = eigenkernel.eigkernel()
        self.eigkL      = eigenkernel.eigkernel()
        self.hArr       = np.array([], dtype=np.float32)
        self.disprefR   = False
        self.disprefL   = False
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
        if mtype == 'iso' or mtype == 'isotropic':
            modparam.readmodtxt(infname=infname, inmod=self.model.isomod)
        elif mtype == 'tti':
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
        elif mtype=='tti':
            self.model.ttimod.get_paraind()
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
        elif mtype == 'tti':
            self.qsArr, self.qpArr  = self.model.get_tti_vmodel() # get the model arrays and initialize elastic tensor
            self.model.rot_dip_strike() 
            self.model.decompose()
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
                    raise ValueError ('Unconsistent phase/group period arrays for Love wave!')
            except:
                raise ValueError ('Unconsistent phase/group period arrays for Love wave!')
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
    
    def compute_tcps(self, wtype='ray', verbose=0, nmodes=1, cmin=-1., cmax=-1., egn96=True, checkdisp=True, tol=1.):
        """
        compute surface wave dispersion of tilted TI model using tcps
        ====================================================================================
        ::: input :::
        wtype       - wave type (Rayleigh or Love)
        nmodes      - number of modes
        cmin, cmax  - minimum/maximum value for phase velocity root searching
        egn96       - computing eigenfunctions/kernels or not
        checkdisp   - check the reasonability of dispersion curves with fast_surf
        tol         - tolerence of maximum differences between tcps and fast_surf
        ====================================================================================
        """
        wtype   = wtype.lower()
        if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
            nfval       = self.TR.size
            freq        = 1./ self.TR
            self.hArr   = self.model.get_dArr()
            nl_in       = self.hArr.size
            ilvry       = 2
            self.eigkR.init_arr(nfval, nl_in, ilvry)
            #- root-finding algorithm using tdisp96, compute phase velocities 
            if self.model.tilt:
                dArr, rhoArr, AArr, CArr, FArr, LArr, NArr, BcArr, BsArr, GcArr, GsArr, HcArr, HsArr, CcArr, CsArr =\
                            self.model.get_layer_tilt_model(self.hArr, 200, 1.)
                self.eigkR.get_AA(BcArr, BsArr, GcArr, GsArr, HcArr, HsArr, CcArr, CsArr)
            else:
                dArr, rhoArr, AArr, CArr, FArr, LArr, NArr = self.model.get_layer_model(self.hArr, 200, 1.)
            # store reference model and ET model
            self.eigkR.get_ref_model(AArr, CArr, FArr, LArr, NArr, rhoArr)
            self.eigkR.get_ETI(AArr, CArr, FArr, LArr, NArr, rhoArr)
            iflsph_in   = 1 # spherical Earth
            # solve for phase velocity
            c_out,d_out,TA_out,TC_out,TF_out,TL_out,TN_out,TRho_out = tdisp96.disprs(ilvry, 1., nfval, 1, verbose, nfval, \
                    np.append(freq, np.zeros(2049-nfval)), cmin, cmax, dArr, AArr,CArr,FArr,LArr,NArr,rhoArr, dArr.size,\
                    iflsph_in, 0., nmodes, 0.5, 0.5)
            # store the reference dispersion curve
            self.indata.dispR.pvelref   = np.float32(c_out[:nfval])
            self.indata.dispR.pvelp     = np.float32(c_out[:nfval])
            #- compute eigenfunction/kernels
            if egn96:
                hs_in       = 0.
                hr_in       = 0.
                ohr_in      = 0.
                ohs_in      = 0.
                refdep_in   = 0.
                dogam       = False # No attenuation
                k           = 2.*np.pi/c_out[:nfval]/self.TR
                k2d         = np.tile(k, (nl_in, 1))
                k2d         = k2d.T
                omega       = 2.*np.pi/self.TR
                omega2d     = np.tile(omega, (nl_in, 1))
                omega2d     = omega2d.T
                # use spherical transformed model parameters
                d_in        = d_out
                TA_in       = TA_out
                TC_in       = TC_out
                TF_in       = TF_out
                TL_in       = TL_out
                TN_in       = TN_out
                TRho_in     = TRho_out
                
                qai_in      = np.ones(nl_in)*1.e6
                qbi_in      = np.ones(nl_in)*1.e6
                etapi_in    = np.zeros(nl_in)
                etasi_in    = np.zeros(nl_in)
                frefpi_in   = np.ones(nl_in)
                frefsi_in   = np.ones(nl_in)
                # solve for group velocity, kernels and eigenfunctions
                u_out, ur, tur, uz, tuz, dcdh,dcdav,dcdah,dcdbv,dcdbh,dcdn,dcdr = tregn96.tregn96(hs_in, hr_in, ohr_in, ohs_in,\
                    refdep_in, dogam, nl_in, iflsph_in, d_in, TA_in, TC_in, TF_in, TL_in, TN_in, TRho_in, \
                    qai_in,qbi_in,etapi_in,etasi_in, frefpi_in, frefsi_in, self.TR.size, self.TR, c_out[:nfval])
                ######################################################
                # store output
                ######################################################
                self.indata.dispR.gvelp    = np.float32(u_out)
                # eigenfunctions
                self.eigkR.get_eigen_psv(np.float32(uz[:nfval,:nl_in]), np.float32(tuz[:nfval,:nl_in]),\
                                         np.float32(ur[:nfval,:nl_in]), np.float32(tur[:nfval,:nl_in]))
                # sensitivity kernels for velocity parameters and density
                self.eigkR.get_vkernel_psv(np.float32(dcdah[:nfval,:nl_in]), np.float32(dcdav[:nfval,:nl_in]), np.float32(dcdbh[:nfval,:nl_in]),\
                        np.float32(dcdbv[:nfval,:nl_in]), np.float32(dcdn[:nfval,:nl_in]), np.float32(dcdr[:nfval,:nl_in]))
                # Love parameters and density in the shape of nfval, nl_in
                self.eigkR.compute_love_kernels()
                self.disprefR   = True
        elif wtype=='l' or wtype == 'love' or wtype == 'lov':
            nfval       = self.TL.size
            freq        = 1./self.TL
            self.hArr   = self.model.get_dArr()
            nl_in       = self.hArr.size
            ilvry       = 1
            self.eigkL.init_arr(nfval, nl_in, ilvry)
            #- root-finding algorithm using tdisp96, compute phase velocities 
            if self.model.tilt:
                dArr, rhoArr, AArr, CArr, FArr, LArr, NArr, BcArr, BsArr, GcArr, GsArr, HcArr, HsArr, CcArr, CsArr =\
                            self.model.get_layer_tilt_model(self.hArr, 200, 1.)
                self.eigkL.get_AA(BcArr, BsArr, GcArr, GsArr, HcArr, HsArr, CcArr, CsArr)
            else:
                dArr, rhoArr, AArr, CArr, FArr, LArr, NArr = self.model.get_layer_model(self.hArr, 200, 1.)
            self.eigkL.get_ref_model(AArr, CArr, FArr, LArr, NArr, rhoArr)
            self.eigkL.get_ETI(AArr, CArr, FArr, LArr, NArr, rhoArr)
            iflsph_in   = 1 # spherical Earth
            # solve for phase velocity
            c_out,d_out,TA_out,TC_out,TF_out,TL_out,TN_out,TRho_out = tdisp96.disprs(ilvry, 1., nfval, 1, verbose, nfval, \
                np.append(freq, np.zeros(2049-nfval)), cmin, cmax, dArr, AArr,CArr,FArr,LArr,NArr,rhoArr, dArr.size,\
                iflsph_in, 0., nmodes, 0.5, 0.5)
            # store the reference dispersion curve
            self.indata.dispL.pvelref   = np.float32(c_out[:nfval])
            self.indata.dispL.pvelp     = np.float32(c_out[:nfval])
            if egn96:
                hs_in       = 0.
                hr_in       = 0.
                ohr_in      = 0.
                ohs_in      = 0.
                refdep_in   = 0.
                dogam       = False # No attenuation
                nl_in       = dArr.size
                k           = 2.*np.pi/c_out[:nfval]/self.TL
                k2d         = np.tile(k, (nl_in, 1))
                k2d         = k2d.T
                omega       = 2.*np.pi/self.TL
                omega2d     = np.tile(omega, (nl_in, 1))
                omega2d     = omega2d.T
                # use spherical transformed model parameters
                d_in        = d_out
                TA_in       = TA_out
                TC_in       = TC_out
                TF_in       = TF_out
                TL_in       = TL_out
                TN_in       = TN_out
                TRho_in     = TRho_out
                
                qai_in      = np.ones(nl_in)*1.e6
                qbi_in      = np.ones(nl_in)*1.e6
                etapi_in    = np.zeros(nl_in)
                etasi_in    = np.zeros(nl_in)
                frefpi_in   = np.ones(nl_in)
                frefsi_in   = np.ones(nl_in)
                # solve for group velocity, kernels and eigenfunctions
                u_out, ut, tut, dcdh,dcdav,dcdah,dcdbv,dcdbh,dcdn,dcdr = tlegn96.tlegn96(hs_in, hr_in, ohr_in, ohs_in,\
                    refdep_in, dogam, nl_in, iflsph_in, d_in, TA_in, TC_in, TF_in, TL_in, TN_in, TRho_in, \
                    qai_in,qbi_in,etapi_in,etasi_in, frefpi_in, frefsi_in, self.TL.size, self.TL, c_out[:nfval])
                ######################################################
                # store output
                ######################################################
                self.indata.dispL.gvelp    = np.float32(u_out)
                # eigenfunctions
                self.eigkL.get_eigen_sh(np.float32(ut[:nfval,:nl_in]), np.float32(tut[:nfval,:nl_in]) )
                # sensitivity kernels for velocity parameters and density
                self.eigkL.get_vkernel_sh(np.float32(dcdbh[:nfval,:nl_in]), np.float32(dcdbv[:nfval,:nl_in]),np.float32(dcdr[:nfval,:nl_in]))
                # Love parameters and density in the shape of nfval, nl_in
                self.eigkL.compute_love_kernels()
                self.disprefL   = True
        if checkdisp:
            hArr        = np.append(self.hArr, 0.)
            vs          = np.sqrt(LArr/rhoArr)
            vs          = np.append(vs, vs[-1])
            vp          = np.sqrt(CArr/rhoArr)
            vp          = np.append(vp, vp[-1])
            rho         = rhoArr
            rho         = np.append(rho, rho[-1])
            qsinv       = 1./(self.qsArr)
            qsinv       = np.append(qsinv, qsinv[-1])
            
            if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
                ilvry               = 2
                nper                = self.TR.size
                per                 = np.zeros(200, dtype=np.float32)
                per[:nper]          = self.TR[:]
                (ur0,ul0,cr0,cl0)   = fast_surf.fast_surf(vs.size, ilvry, \
                                        vp, vs, rho, hArr, qsinv, per, nper)
                pvelp               = cr0[:nper]
                gvelp               = ur0[:nper]
                if (abs(pvelp - self.indata.dispR.pvelref)).max() > tol:
                    print 'WARNING: reference dispersion curves may be erroneous!'
                    return False
            elif wtype=='l' or wtype == 'love':
                ilvry               = 1
                nper                = self.TL.size
                per                 = np.zeros(200, dtype=np.float32)
                per[:nper]          = self.TL[:]
                (ur0,ul0,cr0,cl0)   = fast_surf.fast_surf(vs.size, ilvry, \
                                       vp, vs, rho, hArr, qsinv, per, nper)
                pvelp               = cl0[:nper]
                gvelp               = ul0[:nper]
                if (abs(pvelp - self.indata.dispL.pvelref)).max() > tol:
                    print 'WARNING: reference dispersion curves may be erroneous!'
                    return False
        return True
    
    def perturb_from_kernel(self, wtype='ray'):
        """
        compute perturbation in dispersion from reference model using sensitivity kernels
        ====================================================================================
        ::: input :::
        wtype       - wave type (Rayleigh or Love)
        ====================================================================================
        """
        wtype   = wtype.lower()
        if wtype=='r' or wtype == 'rayleigh' or wtype=='ray':
            if not self.disprefR:
                raise ValueError('referennce dispersion and kernels for Rayleigh wave not computed!')
            nl_in       = self.hArr.size
            if nl_in == 0:
                raise ValueError('No layer arrays stored!')
            #- root-finding algorithm using tdisp96, compute phase velocities 
            if self.model.tilt:
                dArr, rhoArr, AArr, CArr, FArr, LArr, NArr, BcArr, BsArr, GcArr, GsArr, HcArr, HsArr, CcArr, CsArr =\
                            self.model.get_layer_tilt_model(self.hArr, 200, 1.)
                self.eigkR.get_AA(BcArr, BsArr, GcArr, GsArr, HcArr, HsArr, CcArr, CsArr)
            else:
                dArr, rhoArr, AArr, CArr, FArr, LArr, NArr = self.model.get_layer_model(self.hArr, 200, 1.)
            self.eigkR.get_ETI(AArr, CArr, FArr, LArr, NArr, rhoArr)
            dpvel                       = self.eigkR.eti_perturb()
            self.indata.dispR.pvelp     = self.indata.dispR.pvelref + dpvel
            if self.model.tilt:
                amp, phi                = self.eigkR.aa_perturb()
                self.indata.dispR.pampp = amp
                self.indata.dispR.pphip = phi
        elif wtype=='lov' or wtype=='love' or wtype=='l':
            if not self.disprefL:
                raise ValueError('referennce dispersion and kernels not computed!')
            nl_in       = self.hArr.size
            if nl_in == 0:
                raise ValueError('No layer arrays stored!')
            #- root-finding algorithm using tdisp96, compute phase velocities 
            if self.model.tilt:
                dArr, rhoArr, AArr, CArr, FArr, LArr, NArr, BcArr, BsArr, GcArr, GsArr, HcArr, HsArr, CcArr, CsArr =\
                            self.model.get_layer_tilt_model(self.hArr, 200, 1.)
                self.eigkL.get_AA(BcArr, BsArr, GcArr, GsArr, HcArr, HsArr, CcArr, CsArr)
            else:
                dArr, rhoArr, AArr, CArr, FArr, LArr, NArr = self.model.get_layer_model(self.hArr, 200, 1.)
            self.eigkL.get_ETI(AArr, CArr, FArr, LArr, NArr, rhoArr)
            dpvel                       = self.eigkL.eti_perturb()
            self.indata.dispL.pvelp     = self.indata.dispL.pvelref + dpvel
        else:
            raise ValueError('Unexpected wave type: '+mtype)
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
    
    def get_misfit_tti(self):
        """
        compute data misfit
        =====================================================================
        ::: input :::
        =====================================================================
        """
        self.indata.get_misfit_tti()
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
            # print 'run step = ',inew
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
    
    def mc_inv_tti(self, outdir='./workingdir_tti', monoc=True, pfx='MC'):
        """
        
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        # initializations
        self.get_period(dtype = 'ph')
        self.update_mod(mtype = 'tti')
        self.model.ttimod.get_rho()
        self.get_vmodel(mtype = 'tti')
        # initial run
        if not self.compute_tcps(wtype='ray'):
            raise ValueError('Error in computing reference Rayleigh dispersion for initial model!')
        if not self.compute_tcps(wtype='love'):
            raise ValueError('Error in computing reference Love dispersion for initial model!')
        self.perturb_from_kernel(wtype='ray')
        self.perturb_from_kernel(wtype='love')
        self.get_misfit_tti()
        # write initial model
        outmod  = outdir+'/'+pfx+'.mod'
        vmodel.write_model(model=self.model, outfname=outmod, isotropic=False)
        # write initial predicted data
        outdisp = outdir+'/'+pfx+'.ph.ray.disp'
        data.writedispttitxt(outfname=outdisp, outdisp=self.indata.dispR)
        outdisp = outdir+'/'+pfx+'.ph.lov.disp'
        data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispL)
        
        # conver initial model to para
        self.model.ttimod.mod2para()
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
            # print 'run step = ',inew
            if ( inew > 100000 or iacc > 200000 or time.time()-start > 7200.):
                run   = False
            if (np.fmod(inew, 5000) == 0):
                print 'step =',inew, 'elasped time =', time.time()-start, ' sec'
            #------------------------------------------------------------------------------------------
            # every 2500 step, perform a random walk with uniform random value in the paramerter space
            #------------------------------------------------------------------------------------------
            if ( np.fmod(inew, 15001) == 15000 ):
                self.read_paraval('./synthetic_inv/paraval.txt')
            
            # if ( np.fmod(inew, 15001) == 15000 ):
            #     self.model.ttimod.new_paraval(0, 1, 1, 0, 0)
            #     self.get_vmodel(mtype='tti')
            #     # forward computation
            #     # # # self.compute_tcps(wtype='ray')
            #     # # # self.compute_tcps(wtype='love')
            #     self.perturb_from_kernel(wtype='ray')
            #     self.perturb_from_kernel(wtype='love')
            #     self.get_misfit_tti()
            #     oldL                = self.indata.L
            #     oldmisfit           = self.indata.misfit
            #     iacc                += 1
            #     print 'Uniform random walk: likelihood =', self.indata.L, 'misfit =',self.indata.misfit
            #-------------------------------
            # inversion part
            #-------------------------------
            # sample the posterior distribution ##########################################
            # assign new model to old ones
            oldmod      = self.model.ttimod.copy()
            if monoc:
                # loop to find the "good" model,
                # satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
                self.model.ttimod.new_paraval(0, 1, 1, 0, 1)
            else:
                self.model.ttimod.new_paraval(1, 0, 1, 0, 1)
            newmod  = self.model.ttimod
            self.get_vmodel(mtype   = 'tti')
            # forward computation
            self.perturb_from_kernel(wtype='ray')
            self.perturb_from_kernel(wtype='love')
            self.get_misfit_tti()
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
                    fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.dispR.pL, self.indata.dispR.pmisfit,\
                            self.indata.dispL.pL, self.indata.dispL.pmisfit, time.time()-start))        
                    ### ttmodel.writeb (para1, ffb,[-1,i,ii])
                    # return to oldmod
                    self.model.ttimod   = oldmod
                    continue
            # accept the new model
            fidout.write("1 %d %d " % (inew,iacc))
            for i in xrange(newmod.para.npara):
                fidout.write("%g " % newmod.para.paraval[i])
            fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.dispR.pL, self.indata.dispR.pmisfit,\
                    self.indata.dispL.pL, self.indata.dispL.pmisfit, time.time()-start))        
            print "Accept a model", inew, iacc, oldL, newL, self.indata.dispR.pL, self.indata.dispR.pmisfit,\
                            self.indata.dispL.pL, self.indata.dispL.pmisfit, \
                            self.indata.L, self.indata.misfit, time.time()-start
            # write accepted model
            outmod      = outdir+'/'+pfx+'.%d.mod' % iacc
            vmodel.write_model(model=self.model, outfname=outmod, isotropic=False)
            # write corresponding data
            outdisp     = outdir+'/'+pfx+'.ph.ray.%d.disp' % iacc
            data.writedispttitxt(outfname=outdisp, outdisp=self.indata.dispR)
            outdisp     = outdir+'/'+pfx+'.ph.lov.%d.disp' % iacc
            data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispL)
            # assign likelihood/misfit
            oldL        = newL
            oldmisfit   = newmisfit
            iacc        += 1
            # continue
            
            # else:
            #     if monoc:
            #         newmod  = self.model.ttimod.copy()
            #         newmod.para.new_paraval(1)
            #         newmod.para2mod()
            #         newmod.update()
            #         if not newmod.isgood(0, 1, 1, 0):
            #             continue
            #     else:
            #         newmod  = self.model.ttimod.copy()
            #         newmod.para.new_paraval(0)
            #     fidout.write("-2 %d 0 " % inew)
            #     for i in xrange(newmod.para.npara):
            #         fidout.write("%g " % newmod.para.paraval[i])
            #     fidout.write("\n")
            #     self.model.ttimod   = newmod
            #     continue
        fidout.close()
        return
    
    # def mc_inv_tti_fast(self, outdir='./workingdir_tti', monoc=True, pfx='MC'):
    #     if not os.path.isdir(outdir):
    #         os.makedirs(outdir)
    #     # initializations
    #     self.get_period(dtype = 'ph')
    #     self.update_mod(mtype = 'tti')
    #     self.model.ttimod.get_rho()
    #     self.get_vmodel(mtype = 'tti')
    #     # initial run
    #     if not self.compute_tcps(wtype='ray'):
    #         raise ValueError('Error in computing reference Rayleigh dispersion for initial model!')
    #     if not self.compute_tcps(wtype='love'):
    #         raise ValueError('Error in computing reference Love dispersion for initial model!')
    #     self.perturb_from_kernel(wtype='ray')
    #     self.perturb_from_kernel(wtype='love')
    #     self.get_misfit_tti()
    #     # write initial model
    #     outmod  = outdir+'/'+pfx+'.mod'
    #     vmodel.write_model(model=self.model, outfname=outmod, isotropic=False)
    #     # write initial predicted data
    #     outdisp = outdir+'/'+pfx+'.ph.ray.disp'
    #     data.writedispttitxt(outfname=outdisp, outdisp=self.indata.dispR)
    #     outdisp = outdir+'/'+pfx+'.ph.lov.disp'
    #     data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispL)
    #     
    #     # conver initial model to para
    #     self.model.ttimod.mod2para()
    #     # likelihood/misfit
    #     oldL        = self.indata.L
    #     oldmisfit   = self.indata.misfit
    #     print "Initial likelihood = ", oldL, ' misfit =',oldmisfit
    #     self.ttisolver          = fastmc.ttisolver()
    #     self.ttisolver.hArr     = self.hArr
    #     self.ttisolver.indata   = self.indata
    #     self.ttisolver.model    = self.model
    #     self.ttisolver.eigkR    = self.eigkR
    #     self.ttisolver.eigkL    = self.eigkL
    #     # self.ttisolver.mc_inv(True)
        
        
    # def mc_inv_iso(self, outdir='./workingdir', dispdtype='ph', wdisp=.714, rffactor=40., monoc=True, pfx='MC'):
    
    #-------------------------------------------------
    # post-processing functions for isotropic inversion
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
    
    
    
    #-------------------------------------------------------
    # post-processing functions for tilted TI inversion
    #-------------------------------------------------------
    def read_tti_inv(self, indir, npara=47, pfx='MC'):
        try:
            npara   = self.model.ttimod.para.npara
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
        self.rayL       = inArr[:, 5+npara]
        self.raymisfit  = inArr[:, 6+npara]
        self.loveL      = inArr[:, 7+npara]
        self.lovemisfit = inArr[:, 8+npara]
        self.npara      = npara
        return
    
    
    def get_ind_tti_mod(self, ind):
        self.model.ttimod.para.paraval  = np.float32(self.paraval[ind, :])
        self.model.ttimod.para2mod()
        self.model.ttimod.update()
        self.get_vmodel(mtype='tti')
        return
    
    def plot_ind_fitting_curve(self, ind=None):
        if ind!=None:
            self.get_ind_tti_mod(ind=ind)
        self.get_period()
        if not self.disprefR:
            self.compute_tcps(wtype='ray')
        if not self.disprefL:
            self.compute_tcps(wtype='love')
        self.perturb_from_kernel(wtype='ray')
        self.perturb_from_kernel(wtype='love')
        self.get_misfit_tti()
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
        ax = axs[0,0]
        ax.errorbar(self.indata.dispR.pper, self.indata.dispR.pvelo, yerr=self.indata.dispR.stdpvelo, fmt='o-')
        ax.plot(self.indata.dispR.pper, self.indata.dispR.pvelp, 'x')
        ax.set_title('Rayleigh wave dispersion')
        
        # With 4 subplots, reduce the number of axis ticks to avoid crowding.
        ax.locator_params(nbins=4)
        
        ax = axs[0,1]
        ax.errorbar(self.indata.dispR.pper, self.indata.dispR.pampo, yerr=self.indata.dispR.stdpampo, fmt='o-')
        ax.plot(self.indata.dispR.pper, self.indata.dispR.pampp, 'x')
        ax.set_title('Rayleigh wave azimuthal amplitude')
        
        ax = axs[1,0]
        ax.errorbar(self.indata.dispR.pper, self.indata.dispR.pphio, yerr=self.indata.dispR.stdpphio, fmt='o-')
        ax.plot(self.indata.dispR.pper, self.indata.dispR.pphip, 'x')
        ax.set_title('Rayleigh wave fast-axis azimuth')
        
        ax = axs[1,1]
        ax.errorbar(self.indata.dispL.pper, self.indata.dispL.pvelo, yerr=self.indata.dispL.stdpvelo, fmt='o-')
        ax.plot(self.indata.dispL.pper, self.indata.dispL.pvelp, 'x')
        ax.set_title('Love wave dispersion')
        
        # fig.suptitle('Variable errorbars')
        
        plt.show()
    
    def get_min_tti_mod(self, mintype=0):
        if mintype == 0:
            ind                             = self.misfit.argmin()
        elif mintype == 1:
            ind                             = self.raymisfit.argmin()
        elif mintype == 2:
            ind                             = self.lovemisfit.argmin()
        else:
            raise ValueError('unexpected mintype')
        self.get_ind_tti_mod(ind=ind)
        return 
        
        
    def plot_min_fitting_curve(self, mintype=0):
        self.get_min_tti_mod(mintype=mintype)
        self.plot_ind_fitting_curve()
        return
    
    def get_avg_tti_mod(self, threshhold=2.0, mtype='rel'):
        minmisfit                       = self.misfit.min()
        if threshhold < 1.:
            raise ValueError('Relative threshhold should be larger than 1!')
        if minmisfit  >= 0.5:
            tmisfit = threshhold*minmisfit
        else:
            tmisfit = minmisfit + 0.5
        ind                             = (self.misfit <= tmisfit)
        self.model.ttimod.para.paraval  = np.mean(self.paraval[ind, :], axis = 0, dtype=np.float32)
        self.model.ttimod.para2mod()
        self.model.ttimod.update()
        self.get_vmodel(mtype='tti')
        return
    
    def read_paraval(self, infname, mtype='iso'):
        """
        read paraval array
        """
        mtype   = mtype.lower()
        if mtype == 'iso' or mtype == 'isotropic':
            modparam.read_paraval_txt(infname=infname, inpara=self.model.isomod.para)
            self.model.isomod.para2mod()
            self.get_period(dtype = 'ph')
            self.update_mod(mtype = mtype)
            self.get_vmodel(mtype = mtype)
        elif mtype == 'tti':
            modparam.read_paraval_txt(infname=infname, inpara=self.model.ttimod.para)
            self.model.ttimod.para2mod()
            self.get_period(dtype = 'ph')
            self.update_mod(mtype = 'tti')
            self.model.ttimod.get_rho()
            self.get_vmodel(mtype = 'tti')
        return
        
        
        
            
    
    