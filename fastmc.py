# -*- coding: utf-8 -*-
"""
Module for handling parameterization of the model

Numba is used for speeding up of the code.

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import numpy as np
import math
import numba
import random
import vmodel, modparam, data, eigenkernel
    
model1d_type = numba.deferred_type()
model1d_type.define(vmodel.model1d.class_type.instance_type)

data_type = numba.deferred_type()
data_type.define(data.data1d.class_type.instance_type)

eigk_type = numba.deferred_type()
eigk_type.define(eigenkernel.eigkernel.class_type.instance_type)



####################################################
# Predefine the parameters for the ttisolver object
####################################################
spec_ttisolver = [
        ('model',       model1d_type),
        ('indata',      data_type),
        ('eigkR',       eigk_type),
        ('eigkL',       eigk_type),
        ('hArr',        numba.float32[:])
        ]

@numba.jitclass(spec_ttisolver)
class ttisolver(object):
    """
    An object for handling input data for inversion
    ==========================================================================
    ::: parameters :::
    dispR   - Rayleigh wave dispersion data
    dispL   - Love wave dispersion data
    rfr     - radial receiver function data
    rft     - transverse receiver function data
    misfit  - misfit value
    L       - likelihood value
    ==========================================================================
    """
    def __init__(self):
        self.model  = vmodel.model1d()
        self.indata = data.data1d()
        self.eigkR  = eigenkernel.eigkernel()
        self.eigkL  = eigenkernel.eigkernel()
        return
    
    def get_vmodel(self):
        qs, qp  = self.model.get_tti_vmodel() # get the model arrays and initialize elastic tensor
        self.model.rot_dip_strike() 
        self.model.decompose()
        return
    
    def perturb_from_kernel(self, ilvry):
        if ilvry == 2:
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
            for i in xrange(self.indata.dispR.npper):
                self.indata.dispR.pvelp[i]      = self.indata.dispR.pvelref[i] + dpvel[i]
            if self.model.tilt:
                amp, phi                        = self.eigkR.aa_perturb()
                for i in xrange(self.indata.dispR.npper):
                    self.indata.dispR.pampp[i]  = amp[i]
                    self.indata.dispR.pphip[i]  = phi[i]
        elif ilvry == 1:
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
            for i in xrange(self.indata.dispL.npper):
                self.indata.dispL.pvelp[i]      = self.indata.dispL.pvelref[i] + dpvel[i]
        else:
            raise ValueError('Unexpected wave type!')
        return
    # 
    def mc_inv(self, monoc):
        """
        
        """
        # likelihood/misfit
        oldL        = self.indata.L
        oldmisfit   = self.indata.misfit
        print "Initial likelihood = ", oldL, ' misfit =',oldmisfit
        
        run         = True     # the key that controls the sampling
        inew        = 0     # count step (or new paras)
        iacc        = 1     # count acceptance model

        while ( run ):
            inew+= 1
            # print 'run step = ',inew
            if ( inew > 100000 or iacc > 200000 ):
                run   = False
            if (np.fmod(inew, 5000) == 0):
                print 'step =',inew
            #------------------------------------------------------------------------------------------
            # every 2500 step, perform a random walk with uniform random value in the paramerter space
            #------------------------------------------------------------------------------------------
            # if ( np.fmod(inew, 15001) == 15000 ):
            #     vpr.read_paraval('./synthetic_inv/paraval.txt')
            
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
            self.get_vmodel()
            # forward computation
            self.perturb_from_kernel(1)
            self.perturb_from_kernel(2)
            self.indata.get_misfit_tti()
            newL                = self.indata.L
            newmisfit           = self.indata.misfit
            # # 
            # if newL < oldL:
            #     prob    = (oldL-newL)/oldL
            #     rnumb   = random.random()
            #     # reject the model
            #     if rnumb < prob:
            #         fidout.write("-1 %d %d " % (inew,iacc))
            #         for i in xrange(newmod.para.npara):
            #             fidout.write("%g " % newmod.para.paraval[i])
            #         fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.dispR.pL, self.indata.dispR.pmisfit,\
            #                 self.indata.dispL.pL, self.indata.dispL.pmisfit, time.time()-start))        
            #         # return to oldmod
            #         self.model.ttimod   = oldmod
            #         continue
            # # accept the new model
            # fidout.write("1 %d %d " % (inew,iacc))
            # for i in xrange(newmod.para.npara):
            #     fidout.write("%g " % newmod.para.paraval[i])
            # fidout.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, self.indata.dispR.pL, self.indata.dispR.pmisfit,\
            #         self.indata.dispL.pL, self.indata.dispL.pmisfit, time.time()-start))        
            # print "Accept a model", inew, iacc, oldL, newL, self.indata.dispR.pL, self.indata.dispR.pmisfit,\
            #                 self.indata.dispL.pL, self.indata.dispL.pmisfit, \
            #                 self.indata.L, self.indata.misfit, time.time()-start
            # # write accepted model
            # outmod      = outdir+'/'+pfx+'.%d.mod' % iacc
            # vmodel.write_model(model=self.model, outfname=outmod, isotropic=False)
            # # write corresponding data
            # outdisp     = outdir+'/'+pfx+'.ph.ray.%d.disp' % iacc
            # data.writedispttitxt(outfname=outdisp, outdisp=self.indata.dispR)
            # outdisp     = outdir+'/'+pfx+'.ph.lov.%d.disp' % iacc
            # data.writedisptxt(outfname=outdisp, outdisp=self.indata.dispL)
            # # assign likelihood/misfit
            # oldL        = newL
            # oldmisfit   = newmisfit
            # iacc        += 1
            

    