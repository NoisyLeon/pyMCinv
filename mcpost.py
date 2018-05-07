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
    invdata         - data arrays storing inversion results
    disppre_ph/gr   - predicted phase/group dispersion
    rfpre           - object storing 1D model
    ind_acc         - index array indicating accepted models
    ind_rej         - index array indicating rejected models
    ind_thresh      - index array indicating models that pass the misfit criterion
    misfit          - misfit array
    : --- values --- :
    numbrun         - number of total runs
    numbacc         - number of accepted models
    numbrej         - number of rejected models
    npara           - number of parameters for inversion
    min_misfit      - minimum misfit value
    ind_min         - index of the minimum misfit
    thresh          - threshhold value for selecting  the finalized model (misfit < min_misfit + thresh)
    : --- object --- :
    data            - data object storing obsevred data
    avg_model       - average model object
    min_model       - minimum misfit model object
    init_model      - inital model object
    temp_model      - temporary model object, used for analysis of the full assemble of the finally accepted models
    vprfwrd         - vprofile1d object for forward modelling of the average model
    =====================================================================================================================
    """
    def __init__(self, thresh=0.5):
        self.data       = data.data1d()
        self.thresh     = thresh
        self.avg_model  = vmodel.model1d()
        self.min_model  = vmodel.model1d()
        self.init_model = vmodel.model1d()
        self.temp_model = vmodel.model1d()
        self.vprfwrd    = vprofile.vprofile1d()
        return
    
    def read_inv_data(self, infname, verbose=True):
        """
        read inversion results from an input compressed npz file
        """
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
        """
        get the minimum misfit and average model from the inversion data array
        """
        min_paraval         = self.invdata[self.ind_min, 2:(self.npara+2)]
        self.min_model.get_para_model(paraval=min_paraval)
        avg_paraval         = (self.invdata[self.ind_thresh, 2:(self.npara+2)]).mean(axis=0)
        self.avg_model.get_para_model(paraval=avg_paraval)
        self.vprfwrd.model  = self.avg_model
        return 
        
    def read_data(self, infname):
        """
        read observed data from an input npz file
        """
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
            self.vprfwrd.TRpiso = self.data.dispR.pper.copy()
        if self.data.dispR.ngper>0:
            self.vprfwrd.TRpiso = self.data.dispR.gper.copy()
        if self.data.dispL.npper>0:
            self.vprfwrd.TLpiso = self.data.dispL.pper.copy()
        if self.data.dispL.ngper>0:
            self.vprfwrd.TLpiso = self.data.dispL.gper.copy()
        return
    
    def run_avg_fwrd(self):
        """
        run and store receiver functions and surface wave dispersion for the average model
        """
        self.get_period()
        self.get_vmodel()
        self.vprfwrd.npts   = self.rfpre.shape[1]
        self.vprfwrd.update_mod(mtype = 'iso')
        self.vprfwrd.get_vmodel(mtype = 'iso')
        self.vprfwrd.compute_fsurf()
        self.vprfwrd.compute_rftheo()
        return
    
    def plot_rf(self, title='Receiver function', obsrf=True, minrf=True, avgrf=True, assemrf=True, showfig=True):
        """
        plot receiver functions
        ==============================================================================================
        ::: input :::
        title   - title for the figure
        obsrf   - plot observed receiver function or not
        minrf   - plot minimum misfit receiver function or not
        avgrf   - plot the receiver function corresponding to the average of accepted models or not 
        assemrf - plot the receiver functions corresponding to the assemble of accepted models or not 
        ==============================================================================================
        """
        plt.figure()
        ax  = plt.subplot()
        if assemrf:
            for i in self.ind_thresh:
                rf_temp = self.rfpre[i, :]
                plt.plot(self.data.rfr.to, rf_temp, '-',color='grey',  alpha=0.05, lw=3)
        if obsrf:
            plt.errorbar(self.data.rfr.to, self.data.rfr.rfo, yerr=self.data.rfr.stdrfo, label='observed')
        if minrf:
            rf_min      = self.rfpre[self.ind_min, :]
            plt.plot(self.data.rfr.to, rf_min, 'k--', lw=3, label='min model')
        if avgrf:
            self.vprfwrd.npts   = self.rfpre.shape[1]
            self.run_avg_fwrd()
            plt.plot(self.data.rfr.to, self.vprfwrd.data.rfr.rfp, 'r--', lw=3, label='avg model')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlabel('time (sec)', fontsize=30)
        plt.ylabel('ampltude', fontsize=30)
        plt.title(title, fontsize=30)
        plt.legend(loc=0, fontsize=20)
        if showfig:
            plt.show()
        return
    
    def plot_disp(self, title='Dispersion curves', obsdisp=True, mindisp=True, avgdisp=True, assemdisp=True,\
                  disptype='ph', showfig=True):
        """
        plot phase/group dispersion curves
        =================================================================================================
        ::: input :::
        title       - title for the figure
        obsdisp     - plot observed disersion curve or not
        mindisp     - plot minimum misfit dispersion curve or not
        avgdisp     - plot the dispersion curve corresponding to the average of accepted models or not 
        assemdisp   - plot the dispersion curves corresponding to the assemble of accepted models or not 
        =================================================================================================
        """
        plt.figure()
        ax  = plt.subplot()
        if assemdisp:
            for i in self.ind_thresh:
                if disptype == 'ph':
                    disp_temp   = self.disppre_ph[i, :]
                    plt.plot(self.data.dispR.pper, disp_temp, '-',color='grey',  alpha=0.01, lw=3)
                else:
                    disp_temp   = self.disppre_gr[i, :]
                    plt.plot(self.data.dispR.gper, disp_temp, '-',color='grey',  alpha=0.01, lw=3)                
        if obsdisp:
            if disptype == 'ph':
                plt.errorbar(self.data.dispR.pper, self.data.dispR.pvelo, yerr=self.data.dispR.stdpvelo, lw=3, label='observed')
            else:
                plt.errorbar(self.data.dispR.gper, self.data.dispR.gvelo, yerr=self.data.dispR.stdgvelo, lw=3, label='observed')
        if mindisp:
            if disptype == 'ph':
                disp_min    = self.disppre_ph[self.ind_min, :]
                plt.plot(self.data.dispR.pper, disp_min, 'r--', lw=3, label='min model')
            else:
                disp_min    = self.disppre_gr[self.ind_min, :]
                plt.plot(self.data.dispR.gper, disp_min, 'r--', lw=3, label='min model')
        if avgdisp:
            self.run_avg_fwrd()
            if disptype == 'ph':
                disp_avg    = self.vprfwrd.data.dispR.pvelp
                plt.plot(self.data.dispR.pper, disp_avg, 'k--', lw=3, label='avg model')
            else:
                disp_avg    = self.vprfwrd.data.dispR.gvelp
                plt.plot(self.data.dispR.gper, disp_avg, 'k--', lw=3, label='avg model')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlabel('Period (sec)', fontsize=30)
        label_type  = {'ph': 'Phase', 'gr': 'Group'}
        plt.ylabel(label_type[disptype]+' velocity (km/s)', fontsize=30)
        plt.title(title, fontsize=30)
        plt.legend(loc=0, fontsize=20)
        if showfig:
            plt.show()
        return
    
    def plot_profile(self, title='Vs profile', minvpr=True, avgvpr=True, assemvpr=True, showfig=True):
        """
        plot vs profiles
        =================================================================================================
        ::: input :::
        title       - title for the figure
        minvpr      - plot minimum misfit vs profile or not
        avgvpr      - plot the the average of accepted models or not 
        assemvpr    - plot the assemble of accepted models or not 
        =================================================================================================
        """
        plt.figure()
        ax  = plt.subplot()
        if assemvpr:
            for i in self.ind_thresh:
                paraval     = self.invdata[i, 2:(self.npara+2)]
                self.temp_model.get_para_model(paraval=paraval)
                plt.plot(self.temp_model.VsvArr, self.temp_model.zArr, '-',color='grey',  alpha=0.01, lw=3)               
        if minvpr:
            plt.plot(self.min_model.VsvArr, self.min_model.zArr, 'r-', lw=3, label='min model')
        if avgvpr:
            plt.plot(self.avg_model.VsvArr, self.avg_model.zArr, 'b-', lw=3, label='avg model')
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlabel('Vs (km/s)', fontsize=30)
        plt.ylabel('Depth (km)', fontsize=30)
        plt.title(title, fontsize=30)
        plt.legend(loc=0, fontsize=20)
        if showfig:
            plt.ylim([0, 200.])
            # plt.xlim([2.5, 4.])
            plt.gca().invert_yaxis()
            # plt.xlabel('Velocity(km/s)', fontsize=30)
            plt.legend(fontsize=20)
            plt.show()
        return
    
    
    
    
    
    