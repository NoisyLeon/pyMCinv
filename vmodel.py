# -*- coding: utf-8 -*-
"""
Module for handling 1D velocity model objects.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""

import numpy as np
import modparam

class model1d(object):
    """
    An object for handling a 1D Earth model
    =====================================================================================================================
    ::: parameters :::
    :---grid model---:
    VsvArr, VshArr, - Vsv, Vsh, Vpv, Vph velocity (unit - km/s)
    VpvArr, VphArr  
    rhoArr          - density (g/cm^3)
    etaArr          - eta(F/(A-2L)) dimensionless
    AArr, CArr, FArr- Love parameters (unit - GPa)
    LArr, NArr
    zArr            - depth array (unit - km)
    dipArr,strikeArr- dip/strike angles, used for tilted hexagonal symmetric media
    :---layer model---:
    vsv, vsh, vpv,  - velocity (unit - km/s)
    vph          
    rho             - density (g/cm^3)
    eta             - eta(F/(A-2L)) dimensionless
    h               - layer arry (unit - km)
    :   other parameters :
    flat            - = 0 spherical Earth, = 1 flat Earth (default)
                        Note: different from CPS
    CijArr          - elastic tensor given rotational angles(dip, strike) (unit - GPa)
    CijAA           - azimuthally anisotropic elastic tensor (unit - GPa)
    =====================================================================================================================
    """
    def __init__(self):
        self.flat   = False
        self.tilt   = False
        self.isomod = modparam.isomod()
        self.nlay   = 0
        self.ngrid  = 0
        return
    
    def read_model(self, infname, unit=1., isotropic=True, tilt=False,
                indz=0, indvpv=1, indvsv=2, indrho=3, indvph=4, indvsh=5, 
                    indeta=6, inddip=7, indstrike=8):
        """
        Read model in txt format
        ===========================================================================================================
        ::: input parameters :::
        infname                     - input txt file name
        unit                        - unit of input, default = 1., means input has units of km
        isotropic                   - whether the input is isotrpic or not
        indz, indvpv, indvsv, indrho- column id(index) for depth, vpv, vsv, rho, vph, vsh, eta
        indvph, indvsh, indeta
        reverse                     - revert the arrays or not
        ===========================================================================================================
        """
        inArr   = np.loadtxt(infname, dtype=np.float64)
        z       = inArr[:, indz]
        rho     = inArr[:, indrho]*unit
        vpv     = inArr[:, indvpv]*unit
        vsv     = inArr[:, indvsv]*unit
        N       = inArr.shape[0]
        if isotropic:
            vph     = inArr[:, indvpv]*unit
            vsh     = inArr[:, indvsv]*unit
            eta     = np.ones(N, dtype=np.float64)
        else:
            vph     = inArr[:, indvph]*unit
            vsh     = inArr[:, indvsh]*unit
        if tilt and not isotropic:
            dip     = inArr[:, inddip]
            strike  = inArr[:, indstrike]
        else:
            dip     = np.ones(N, dtype=np.float64)
            strike  = np.ones(N, dtype=np.float64)
        self.get_model_vel(vsv=vsv, vsh=vsh, vpv=vpv, vph=vph,\
                      eta=eta, rho=rho, z=z, dip=dip, strike=strike, tilt=tilt, N=N)
        return
    
    def write_model(self, outfname, isotropic=True):
        """
        Write model in txt format
        ===========================================================================================================
        ::: input parameters :::
        outfname                    - output txt file name
        unit                        - unit of output, default = 1., means output has units of km
        isotropic                   - whether the input is isotrpic or not
        ===========================================================================================================
        """
        outArr  = np.append(self.zArr, self.VsvArr)
        if not isotropic:
            outArr  = np.append(outArr, self.VshArr)
        outArr      = np.append(outArr, self.VpvArr)
        if not isotropic:
            outArr  = np.append(outArr, self.VphArr)
            outArr  = np.append(outArr, self.etaArr)
            if self.tilt:
                outArr  = np.append(outArr, self.dipArr)
                outArr  = np.append(outArr, self.strikeArr)
        outArr      = np.append(outArr, self.rhoArr)
        if isotropic:
            N       = 4
            header  = 'depth vs vp rho'
        else:
            if self.tilt:
                N       = 9
                header  = 'depth vsv vsh vpv vph eta dip strike rho'
            else:
                N       = 7
                header  = 'depth vsv vsh vpv vph eta rho'
        outArr  = outArr.reshape((N, self.ngrid))
        outArr  = outArr.T
        np.savetxt(outfname, outArr, fmt='%g', header=header)
        return 

    def get_model_vel(self, vsv, vsh, vpv, vph,
                    eta, rho, z, dip, strike, tilt, N):
        """
        Get model data given velocity/density/depth arrays
        """
        self.zArr           = z
        self.VsvArr         = vsv
        self.VshArr         = vsh
        self.VpvArr         = vpv
        self.VphArr         = vph
        self.etaArr         = eta
        self.rhoArr         = rho
        if tilt:
            self.dipArr     = dip
            self.strikeArr  = strike
        self.vel2love()
        self.ngrid          = z.size
        return

    def vel2love(self):
        """
        velocity parameters to Love parameters
        """
        if self.ngrid != 0:
            self.AArr   = self.rhoArr * (self.VphArr)**2
            self.CArr   = self.rhoArr * (self.VpvArr)**2
            self.LArr   = self.rhoArr * (self.VsvArr)**2
            self.FArr   = self.etaArr * (self.AArr - 2.* self.LArr)
            self.NArr   = self.rhoArr * (self.VshArr)**2
        if self.nlay != 0:
            self.A      = self.rho * (self.vph)**2
            self.C      = self.rho * (self.vpv)**2
            self.L      = self.rho * (self.vsv)**2
            self.F      = self.eta * (self.A - 2.* self.L)
            self.N      = self.rho * (self.vsh)**2
        return

    def love2vel(self):
        """
        Love parameters to velocity parameters
        """
        if self.ngrid != 0:
            self.VphArr     = np.sqrt(self.AArr/self.rhoArr)
            self.VpvArr     = np.sqrt(self.CArr/self.rhoArr)
            self.VshArr     = np.sqrt(self.NArr/self.rhoArr)
            self.VsvArr     = np.sqrt(self.LArr/self.rhoArr)
            self.etaArr     = self.FArr/(self.AArr - 2.* self.LArr)
        if self.nlay != 0:
            self.vph        = np.sqrt(self.A/self.rho)
            self.vpv        = np.sqrt(self.C/self.rho)
            self.vsh        = np.sqrt(self.N/self.rho)
            self.vsv        = np.sqrt(self.L/self.rho)
            self.eta        = self.F/(self.A - 2.* self.L)
        return
    
    def grid2layer(self, checklayer=True):
        """
        Convert grid point model to layerized model
        """
        if checklayer:
            if not self.is_layer_model():
                return False
        self.nlay   = int(self.ngrid/2)
        indz0       = np.arange(int(self.ngrid/2), dtype = np.int32)*2
        indz1       = np.arange(int(self.ngrid/2), dtype = np.int32)*2 + 1
        z0          = self.zArr[indz0]
        z1          = self.zArr[indz1]
        self.h      = z1 - z0
        indlay      = np.arange(int(self.ngrid/2), dtype = np.int32)*2 + 1
        self.vph    = self.VphArr[indlay]
        self.vpv    = self.VpvArr[indlay]
        self.vsh    = self.VshArr[indlay]
        self.vsv    = self.VsvArr[indlay]
        self.eta    = self.etaArr[indlay]
        self.rho    = self.rhoArr[indlay]
        if self.tilt:
            self.dip    = self.dipArr[indlay]
            self.strike = self.strikeArr[indlay]
        return True
    
    def is_iso(self):
        """Check if the model is isotropic at each point.
        """
        tol     = 1e-5
        if (abs(self.AArr - self.CArr)).max() > tol or (abs(self.LArr - self.NArr)).max() > tol\
            or (abs(self.FArr - (self.AArr- 2.*self.LArr))).max() > tol:
            return False
        return True

    def get_iso_vmodel(self):
        """
        get the isotropic model from isomod
        """
        hArr, vs, vp, rho, qs, qp, nlay = self.isomod.get_vmodel()
        self.vsv                = vs.copy()
        self.vsh                = vs.copy()
        self.vpv                = vp.copy()
        self.vph                = vp.copy()
        self.eta                = np.ones(nlay, dtype=np.float64)
        self.rho                = rho
        self.h                  = hArr
        self.qs                 = qs
        self.qp                 = qp
        self.nlay               = nlay
        self.ngrid              = 2*nlay
        # store grid point model
        indlay                  = np.arange(nlay, dtype=np.int32)
        indgrid0                = indlay*2
        indgrid1                = indlay*2+1
        self.VsvArr             = np.ones(self.ngrid, dtype=np.float64)
        self.VshArr             = np.ones(self.ngrid, dtype=np.float64)
        self.VpvArr             = np.ones(self.ngrid, dtype=np.float64)
        self.VphArr             = np.ones(self.ngrid, dtype=np.float64)
        self.qsArr              = np.ones(self.ngrid, dtype=np.float64)
        self.qpArr              = np.ones(self.ngrid, dtype=np.float64)
        self.rhoArr             = np.ones(self.ngrid, dtype=np.float64)
        self.etaArr             = np.ones(self.ngrid, dtype=np.float64)
        self.zArr               = np.zeros(self.ngrid, dtype=np.float64)
        depth                   = hArr.cumsum()
        # model arrays
        self.VsvArr[indgrid0]   = vs[:]
        self.VsvArr[indgrid1]   = vs[:]
        self.VshArr[indgrid0]   = vs[:]
        self.VshArr[indgrid1]   = vs[:]
        self.VpvArr[indgrid0]   = vp[:]
        self.VpvArr[indgrid1]   = vp[:]
        self.VphArr[indgrid0]   = vp[:]
        self.VphArr[indgrid1]   = vp[:]
        self.rhoArr[indgrid0]   = rho[:]
        self.rhoArr[indgrid1]   = rho[:]
        self.qsArr[indgrid0]    = qs[:]
        self.qsArr[indgrid1]    = qs[:]
        self.qpArr[indgrid0]    = qp[:]
        self.qpArr[indgrid1]    = qp[:]
        # depth array
        indlay2                 = np.arange(nlay-1, dtype=np.int32)
        indgrid2                = indlay2*2+2
        self.zArr[indgrid1]     = depth
        self.zArr[indgrid2]     = depth[:-1]
        self.vel2love()
        return
    
    def is_layer_model(self):
        """
        Check if the grid point model is a layerized one or not
        """
        if self.ngrid %2 !=0:
            return False
        self.vel2love()
        if self.zArr[0] != 0.:
            return False
        indz0   = np.arange(int(self.ngrid/2)-1, dtype = np.int32)*2 + 1
        indz1   = np.arange(int(self.ngrid/2)-1, dtype = np.int32)*2 + 2
        if not np.allclose(self.zArr[indz0], self.zArr[indz1]):
            return False
        ind0    = np.arange(int(self.ngrid/2), dtype = np.int32)*2
        ind1    = np.arange(int(self.ngrid/2), dtype = np.int32)*2 + 1
        if not np.allclose(self.AArr[ind0], self.AArr[ind1]):
            return False
        if not np.allclose(self.CArr[ind0], self.CArr[ind1]):
            return False
        if not np.allclose(self.FArr[ind0], self.FArr[ind1]):
            return False
        if not np.allclose(self.LArr[ind0], self.LArr[ind1]):
            return False
        if not np.allclose(self.NArr[ind0], self.NArr[ind1]):
            return False
        if not np.allclose(self.rhoArr[ind0], self.rhoArr[ind1]):
            return False
        if self.tilt: 
            if not np.allclose(self.dipArr[ind0], self.dipArr[ind1]):
                return False
            if not np.allclose(self.strikeArr[ind0], self.strikeArr[ind1]):
                return False
        return True
    
    
    
    
    
    
    

        
    
    
    
        
        
    
    
    
    
    
