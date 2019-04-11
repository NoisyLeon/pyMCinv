# -*- coding: utf-8 -*-
"""
Module for handling output eigenfunction and sensitivity kernels of surface waves in tilted TI model

Numba is used for speeding up of the code.

references
    Montagner, J.P. and Nataf, H.C., 1986. A simple method for inverting the azimuthal anisotropy of surface waves.
            Journal of Geophysical Research: Solid Earth, 91(B1), pp.511-520.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import numpy as np
import math
import numba


 
####################################################
# Predefine the parameters for the eigkernel object
####################################################
spec_eigkernel = [
        # Love parameters and density, reference value
        ('A',       numba.float32[:]),
        ('C',       numba.float32[:]),
        ('F',       numba.float32[:]),
        ('L',       numba.float32[:]),
        ('N',       numba.float32[:]),
        ('rho',     numba.float32[:]),
        # ETI Love parameters
        ('Aeti',    numba.float32[:]),
        ('Ceti',    numba.float32[:]),
        ('Feti',    numba.float32[:]),
        ('Leti',    numba.float32[:]),
        ('Neti',    numba.float32[:]),
        ('rhoeti',  numba.float32[:]),
        # azimuthal anisotropic terms of the tilted TI model
        ('BcArr',   numba.float32[:]),
        ('BsArr',   numba.float32[:]),
        ('GcArr',   numba.float32[:]),
        ('GsArr',   numba.float32[:]),
        ('HcArr',   numba.float32[:]),
        ('HsArr',   numba.float32[:]),
        ('CcArr',   numba.float32[:]),
        ('CsArr',   numba.float32[:]),
        # PSV eigenfunctions
        ('uz',      numba.float32[:,:]),
        ('tuz',     numba.float32[:,:]),
        ('ur',      numba.float32[:,:]),
        ('tur',     numba.float32[:,:]),
        ('durdz',   numba.float32[:,:]),
        ('duzdz',   numba.float32[:,:]),
        # SH eigenfunctions
        ('ut',      numba.float32[:,:]),
        ('tut',     numba.float32[:,:]),
        ('dutdz',   numba.float32[:,:]),
        # velocity kernels
        ('dcdah',   numba.float32[:,:]),
        ('dcdav',   numba.float32[:,:]),
        ('dcdbh',   numba.float32[:,:]),
        ('dcdbv',   numba.float32[:,:]),
        ('dcdr',    numba.float32[:,:]),
        ('dcdn',    numba.float32[:,:]),
        # Love kernels
        ('dcdA',    numba.float32[:,:]),
        ('dcdC',    numba.float32[:,:]),
        ('dcdF',    numba.float32[:,:]),
        ('dcdL',    numba.float32[:,:]),
        ('dcdN',    numba.float32[:,:]),
        ('dcdrl',   numba.float32[:,:]),
        # number of freq
        ('nfreq',   numba.int32),
        # number of layers
        ('nlay',    numba.int32),
        ('ilvry',   numba.int32)
        
        ]

@numba.jitclass(spec_eigkernel)
class eigkernel(object):
    """
    An object for handling parameter perturbations
    =====================================================================================================================
    ::: parameters :::
    :   values  :
    nlay        - number of layers
    ilvry       - indicator for Love or Rayleigh waves (1 - Love, 2 - Rayleigh)
    :   model   :
    A, C, F, L, N, rho                          - layerized model
    If the model is a tilted hexagonal symmetric model:
    BcArr, BsArr, GcArr, GsArr, HcArr, HsArr    - 2-theta azimuthal terms 
    CcArr, CsArr                                - 4-theta azimuthal terms
    : eigenfunctions :
    uz, ur      - vertical/radial displacement functions
    tuz, tur    - vertical/radial stress functions
    duzdz, durdz- derivatives of vertical/radial displacement functions
    : velocity/density sensitivity kernels :
    dcdah/dcdav - vph/vpv kernel
    dcdbh/dcdbv - vsh/vsv kernel
    dcdn        - eta kernel
    dcdr        - density kernel
    : Love parameters/density sensitivity kernels, derived from the kernels above using chain rule :
    dcdA, dcdC, dcdF, dcdL, dcdN    - Love parameter kernels
    dcdrl                           - density kernel
    =====================================================================================================================
    """
    def __init__(self):
        self.nfreq      = 0
        self.nlay       = -1
        self.ilvry      = -1
        return
    
    def init_arr(self, nfreq, nlay, ilvry):
        """
        initialize arrays
        """
        if ilvry != 1 and ilvry != 2:
            raise ValueError('Unexpected ilvry value!')
        self.nfreq      = nfreq
        self.nlay       = nlay
        self.ilvry      = ilvry
        # reference Love parameters and density
        self.A          = np.zeros(np.int64(nlay), dtype=np.float32)
        self.C          = np.zeros(np.int64(nlay), dtype=np.float32)
        self.F          = np.zeros(np.int64(nlay), dtype=np.float32)
        self.L          = np.zeros(np.int64(nlay), dtype=np.float32)
        self.N          = np.zeros(np.int64(nlay), dtype=np.float32)
        self.rho        = np.zeros(np.int64(nlay), dtype=np.float32)
        # ETI Love parameters and density
        self.Aeti       = np.zeros(np.int64(nlay), dtype=np.float32)
        self.Ceti       = np.zeros(np.int64(nlay), dtype=np.float32)
        self.Feti       = np.zeros(np.int64(nlay), dtype=np.float32)
        self.Leti       = np.zeros(np.int64(nlay), dtype=np.float32)
        self.Neti       = np.zeros(np.int64(nlay), dtype=np.float32)
        self.rhoeti     = np.zeros(np.int64(nlay), dtype=np.float32)
        # azimuthal anisotropic terms
        self.BcArr      = np.zeros(np.int64(nlay), dtype=np.float32)
        self.BsArr      = np.zeros(np.int64(nlay), dtype=np.float32)
        self.GcArr      = np.zeros(np.int64(nlay), dtype=np.float32)
        self.GsArr      = np.zeros(np.int64(nlay), dtype=np.float32)
        self.HcArr      = np.zeros(np.int64(nlay), dtype=np.float32)
        self.HsArr      = np.zeros(np.int64(nlay), dtype=np.float32)
        self.CcArr      = np.zeros(np.int64(nlay), dtype=np.float32)
        self.CsArr      = np.zeros(np.int64(nlay), dtype=np.float32)
        # eigenfunctions
        if ilvry == 1:
            self.ut     = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
            self.tut    = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
        else:
            self.uz     = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
            self.tuz    = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
            self.ur     = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
            self.tur    = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
        # velocity kernels
        if ilvry == 2:
            self.dcdah  = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
            self.dcdav  = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
            self.dcdn   = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
        self.dcdbh      = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
        self.dcdbv      = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
        self.dcdr       = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
        # Love kernels
        if ilvry == 2:
            self.dcdA   = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
            self.dcdC   = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
            self.dcdF   = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
        self.dcdL       = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
        self.dcdN       = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
        # density kernel for Love parameter group
        self.dcdrl      = np.zeros((np.int64(nfreq), np.int64(nlay)), dtype=np.float32)
        return
    
    def get_ref_model(self, A, C, F, L, N, rho):
        """
        get the Love parameter arrays for the reference model
        """
        self.A[:]       = A
        self.C[:]       = C
        self.F[:]       = F
        self.L[:]       = L
        self.N[:]       = N
        self.rho[:]     = rho
        return
    
    def get_ETI(self, Aeti, Ceti, Feti, Leti, Neti, rhoeti):
        """
        get the ETI(effective TI) Love parameter arrays, as perturbation
        """
        self.Aeti[:]    = Aeti
        self.Ceti[:]    = Ceti
        self.Feti[:]    = Feti
        self.Leti[:]    = Leti
        self.Neti[:]    = Neti
        self.rhoeti[:]  = rhoeti
        return
    
    def get_AA(self, BcArr, BsArr, GcArr, GsArr, HcArr, HsArr, CcArr, CsArr):
        """
        get the AA(azimuthally anisotropic) term arrays, as perturbation
        """
        self.BcArr[:]  = BcArr
        self.BsArr[:]  = BsArr
        self.GcArr[:]  = GcArr
        self.GsArr[:]  = GsArr
        self.HcArr[:]  = HcArr
        self.HsArr[:]  = HsArr
        self.CcArr[:]  = CcArr
        self.CsArr[:]  = CsArr
        return
    
    def get_eigen_psv(self, uz, tuz, ur, tur):
        """
        get the P-SV motion eigenfunctions
        """
        self.uz[:,:]    = uz
        self.tuz[:,:]   = tuz
        self.ur[:,:]    = ur
        self.tur[:,:]   = tur
        return
    
    def get_eigen_sh(self, ut, tut):
        """
        get the SH motion eigenfunctions
        """
        self.ut = ut
        self.tut= tut
        return
    
    # # def compute_eig_diff(self):
    # #     if self.ilvry == 1:
    # #         self.dutdz  = self.tut/self.L
    # #     else:
    # #         self.durdz  = 1./self.L*self.tur - k2d*self.uz
    # #         self.duzdz  = k2d*self.F/self.C*self.ur + self.tuz/self.C
    # #     return
    
    def get_vkernel_psv(self, dcdah, dcdav, dcdbh, dcdbv, dcdn, dcdr):
        """
        get the velocity kernels for P-SV motion
        """
        self.dcdah[:,:]     = dcdah
        self.dcdav[:,:]     = dcdav
        self.dcdbh[:,:]     = dcdbh
        self.dcdbv[:,:]     = dcdbv
        self.dcdr[:,:]      = dcdr
        self.dcdn[:,:]      = dcdn
        return
    
    def get_vkernel_sh(self, dcdbh, dcdbv, dcdr):
        """
        get the velocity kernels for SH motion
        """
        self.dcdbh[:,:]     = dcdbh
        self.dcdbv[:,:]     = dcdbv
        self.dcdr[:,:]      = dcdr
        return
    
    def compute_love_kernels(self):
        """
        compute sensitivity kernels for Love paramters using chain rule
        """
        if self.ilvry == 2:
            for i in xrange(self.nfreq):
                for j in xrange(self.nlay):
                    self.dcdA[i, j] = 0.5/np.sqrt(self.A[j]*self.rho[j]) * self.dcdah[i,j] - self.F[j]/((self.A[j]-2.*self.L[j])**2)*self.dcdn[i, j]
                    self.dcdC[i, j] = 0.5/np.sqrt(self.C[j]*self.rho[j]) * self.dcdav[i,j]
                    self.dcdF[i, j] = 1./(self.A[j]-2.*self.L[j])*self.dcdn[i,j]
                    self.dcdL[i, j] = 0.5/np.sqrt(self.L[j]*self.rho[j])*self.dcdbv[i,j] + 2.*self.F[j]/((self.A[j]-2.*self.L[j])**2)*self.dcdn[i, j]
                    ### self.dcdN[i, j] = 0.5/np.sqrt(self.N[j]*self.rho[j])*self.dcdbh[i,j]
                    self.dcdrl[i, j]= -0.5*self.dcdah[i, j]*np.sqrt(self.A[j]/(self.rho[j]**3)) - 0.5*self.dcdav[i, j]*np.sqrt(self.C[j]/(self.rho[j]**3))\
                                        -0.5*self.dcdbh[i, j]*np.sqrt(self.N[j]/(self.rho[j]**3)) -0.5*self.dcdbv[i, j]*np.sqrt(self.L[j]/(self.rho[j]**3))\
                                            + self.dcdr[i, j]
        else:
            for i in xrange(self.nfreq):
                for j in xrange(self.nlay):
                    self.dcdL[i, j] = 0.5/np.sqrt(self.L[j]*self.rho[j])*self.dcdbv[i,j] 
                    self.dcdN[i, j] = 0.5/np.sqrt(self.N[j]*self.rho[j])*self.dcdbh[i,j]
                    self.dcdrl[i, j]= -0.5*self.dcdbh[i,j]*np.sqrt(self.N[j]/(self.rho[j]**3)) \
                                        -0.5*self.dcdbv[i,j]*np.sqrt(self.L[j]/(self.rho[j]**3)) + self.dcdr[i,j]
        return
    
    def eti_perturb(self):
        """
        Compute the phase velocity perturbation from reference to ETI model
        """
        dA      = self.Aeti - self.A
        dC      = self.Ceti - self.C
        dF      = self.Feti - self.F
        dL      = self.Leti - self.L
        dN      = self.Neti - self.N
        dr      = self.rhoeti - self.rho
        dpvel   = np.zeros(np.int64(self.nfreq), dtype = np.float32)
        if self.ilvry == 2:         
            for i in xrange(self.nfreq):
                for j in xrange(self.nlay):
                    dpvel[i]    = dpvel[i] + self.dcdA[i, j] * dA[j] + self.dcdC[i, j] * dC[j] + self.dcdF[i, j] * dF[j]\
                                    + self.dcdL[i, j] * dL[j] # + self.dcdN[i, j] * dN[j] # + self.dcdrl[i, j] * dr[j]
        else:
            for i in xrange(self.nfreq):
                for j in xrange(self.nlay):
                    dpvel[i]    = dpvel[i] + self.dcdL[i, j] * dL[j] + self.dcdN[i, j] * dN[j] # + self.dcdrl[i, j] * dr[j]
        return dpvel
        
    def aa_perturb(self):
        """
        Compute the phase velocity perturbation from ETI to AA(azimuthally anisotropic) model
        """
        az          = np.zeros(360, dtype = np.float32)
        for i in xrange(360):
            az[i]   = np.float32(i+1)
        faz         = np.zeros(360, dtype = np.float32)
        Ac2az       = np.zeros(np.int64(self.nfreq), dtype = np.float32)
        As2az       = np.zeros(np.int64(self.nfreq), dtype = np.float32)
        amp         = np.zeros(np.int64(self.nfreq), dtype = np.float32)
        phi         = np.zeros(np.int64(self.nfreq), dtype = np.float32)
        if self.ilvry != 2:
            raise ValueError('Love wave AA terms computation not supported!')
        for i in xrange(self.nfreq):
            for j in xrange(self.nlay):
                Ac2az[i]    = Ac2az[i] + self.BcArr[j] * self.dcdA[i, j] + self.GcArr[j] * self.dcdL[i, j] + self.HcArr[j] * self.dcdF[i, j]
                As2az[i]    = As2az[i] + self.BsArr[j] * self.dcdA[i, j] + self.GsArr[j] * self.dcdL[i, j] + self.HsArr[j] * self.dcdF[i, j]
            for k in xrange(360):
                faz[k]      = Ac2az[i] * np.cos(2.*az[k]/180.*np.pi) + As2az[i] * np.sin(2.*az[k]/180.*np.pi)
            amp[i]      = (faz.max() - faz.min())/2.
            indmax      = faz.argmax()
            phi[i]      = az[indmax]
            if phi[i] >= 180.:
                phi[i]  = phi[i] - 180.
        return amp, phi

    