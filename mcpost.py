# -*- coding: utf-8 -*-
"""
Module for results and postprocessing of MC inversion

:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import vmodel, modparam, data
import numpy as np


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
        self.numbacc    = np.where(self.ind_acc)[0].size
        self.numbrej    = np.where(self.ind_rej)[0].size
        if verbose:
            print 'Number of runs = '+str(self.numbrun)
            print 'Number of accepted models = '+str(self.numbacc)
            print 'Number of rejected models = '+str(self.numbrej)
            print 'Number of invalid models = '+str(self.numbrun - self.numbacc - self.numbrej)
            print 'minimum misfit = '+str(self.min_misfit)
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
    
    
    
    
    