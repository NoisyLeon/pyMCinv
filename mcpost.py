# -*- coding: utf-8 -*-
"""
Module for results and postprocessing of MC inversion

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import vmodel, modparam, data, vprofile
import numpy as np
import matplotlib.pyplot as plt

class postvpr(object):
    """
    An object for post data processing of 1D velocity profile inversion
    =====================================================================================================================
    ::: parameters :::
    : --- arrays --- :
    invdata             - data arrays storing inversion results
    disppre_ph/gr       - predicted phase/group dispersion
    rfpre               - object storing 1D model
    data                - data object storing obsevred data
    
    =====================================================================================================================
    """
    def __init__(self, thresh=0.5):
        self.data       = data.data1d()
        self.thresh     = thresh
        self.avg_model  = vmodel.model1d()
        self.min_model  = vmodel.model1d()
        self.init_model = vmodel.model1d()
        self.vprfwd     = vprofile.vprofile1d()
        return
    
    def read_inv_data(self, infname, verbose=True):
        inarr           = np.load(infname)
        self.invdata    = inarr['arr_0']
        self.disppre_ph = inarr['arr_1']
        self.disppre_gr = inarr['arr_2']
        self.rfpre      = inarr['arr_3']
        #
        self.numbrun    = self.invdata.shape[0]
        self.npara      = self.invdata.shape[1] - 9
        self.ind_acc    = self.invdata[:, 0] == 1.
        self.ind_rej    = self.invdata[:, 0] == -1.
        self.misfit     = self.invdata[:, self.npara+3]
        self.min_misfit = self.misfit[self.ind_acc + self.ind_rej].min()
        self.ind_min    = np.where(self.misfit == self.min_misfit)[0][0]
        self.ind_thresh = np.where(self.ind_acc*(self.misfit<= self.min_misfit+ self.thresh))[0]
        self.numbacc    = np.where(self.ind_acc)[0].size
        self.numbrej    = np.where(self.ind_rej)[0].size
        if verbose:
            print 'Number of runs = '+str(self.numbrun)
            print 'Number of accepted models = '+str(self.numbacc)
            print 'Number of rejected models = '+str(self.numbrej)
            print 'Number of invalid models = '+str(self.numbrun - self.numbacc - self.numbrej)
            print 'minimum misfit = '+str(self.min_misfit)
        return
    
    def get_vmodel(self):
        min_paraval         = self.invdata[self.ind_min, 2:(self.npara+2)]
        self.min_model.get_para_model(paraval=min_paraval)
        avg_paraval         = (self.invdata[self.ind_thresh, 2:(self.npara+2)]).mean(axis=0)
        self.avg_model.get_para_model(paraval=avg_paraval)
        self.vprfwd.model   = self.avg_model
        return 
        
    def read_data(self, infname):
        inarr           = np.load(infname)
        index           = inarr['arr_0']
        if index[0] == 1:
            indata      = np.append(inarr['arr_1'], inarr['arr_2'])
            indata      = np.append(indata, inarr['arr_3'])
            indata      = indata.reshape(3, indata.size/3)
            self.data.dispR.get_disp(indata=indata, dtype='ph')
        if index[0] == 0:
            indata      = np.append(inarr['arr_1'], inarr['arr_2'])
            indata      = np.append(indata, inarr['arr_3'])
            indata      = indata.reshape(3, indata.size/3)
            self.data.dispR.get_disp(indata=indata, dtype='gr')
        if index[0] == 1 and index[1] == 1:
            indata      = np.append(inarr['arr_4'], inarr['arr_5'])
            indata      = np.append(indata, inarr['arr_6'])
            indata      = indata.reshape(3, indata.size/3)
            self.data.dispR.get_disp(indata=indata, dtype='gr')
        if (index[0] * index[1]) == 1:
            indata      = np.append(inarr['arr_7'], inarr['arr_8'])
            indata      = np.append(indata, inarr['arr_9'])
            indata      = indata.reshape(3, indata.size/3)
            self.data.rfr.get_rf(indata=indata)
        else:
            indata      = np.append(inarr['arr_4'], inarr['arr_5'])
            indata      = np.append(indata, inarr['arr_6'])
            indata      = indata.reshape(3, indata.size/3)
            self.data.rfr.get_rf(indata=indata)
        return
    
    def get_period(self):
        """
        get period array for forward modelling
        """
        if self.data.dispR.npper>0:
            self.vprfwd.TRpiso  = self.data.dispR.pper.copy()
        if self.data.dispR.ngper>0:
            self.vprfwd.TRpiso  = self.data.dispR.gper.copy()
        if self.data.dispL.npper>0:
            self.vprfwd.TLpiso  = self.data.dispL.pper.copy()
        if self.data.dispL.ngper>0:
            self.vprfwd.TLpiso  = self.data.dispL.gper.copy()
        return
    
    def plot_rf(self, obsrf=True, minrf=True, avgrf=True, assemrf=False, showfig=True):
        plt.figure()
        ax  = plt.subplot()
        if obsrf:
            plt.errorbar(self.data.rfr.to, self.data.rfr.rfo, yerr=self.data.rfr.stdrfo)
        # continue here
        if prediction:
            plt.plot(self.to, self.rfp, 'r--', lw=3)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlabel('time (sec)', fontsize=30)
        plt.ylabel('ampltude', fontsize=30)
        plt.title('receiver function', fontsize=30)
        if showfig:
            plt.show()
        if minrf:
            rf_min  = self.rfpre[self.ind_min, :]
        
        return
    
    
    
    
    
    
    