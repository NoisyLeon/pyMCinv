# -*- coding: utf-8 -*-
"""
A python module for joint inversion based on ASDF database

:Methods:


:Dependencies:
    pyasdf and its dependencies
    ObsPy  and its dependencies
    pyproj
    Basemap
    pyfftw 0.10.3 (optional)
    
:Copyright:
    Author: Lili Feng
    Graduate Research Assistant
    CIEI, Department of Physics, University of Colorado Boulder
    email: lili.feng@colorado.edu
"""
import pyasdf, h5py
import numpy as np
import matplotlib.pyplot as plt
import obspy
import warnings
import copy
import os, shutil
from functools import partial
import multiprocessing
from subprocess import call
from mpl_toolkits.basemap import Basemap, shiftgrid, cm
import obspy
import vprofile
import time

class invASDF(pyasdf.ASDFDataSet):
    """ An object to for MCMC inversion based on ASDF database
    =================================================================================================================
    version history:
           - first version
    =================================================================================================================
    """
    def print_info(self):
        """
        print information of the dataset.
        """
        outstr  = '================================================= Ambient Noise Cross-correlation Database =================================================\n'
        outstr  += self.__str__()+'\n'
        outstr  += '--------------------------------------------------------------------------------------------------------------------------------------------\n'
        if 'NoiseXcorr' in self.auxiliary_data.list():
            outstr      += 'NoiseXcorr              - Cross-correlation seismogram\n'
        if 'StaInfo' in self.auxiliary_data.list():
            outstr      += 'StaInfo                 - Auxiliary station information\n'
        if 'DISPbasic1' in self.auxiliary_data.list():
            outstr      += 'DISPbasic1              - Basic dispersion curve, no jump correction\n'
        if 'DISPbasic2' in self.auxiliary_data.list():
            outstr      += 'DISPbasic2              - Basic dispersion curve, with jump correction\n'
        if 'DISPpmf1' in self.auxiliary_data.list():
            outstr      += 'DISPpmf1                - PMF dispersion curve, no jump correction\n'
        if 'DISPpmf2' in self.auxiliary_data.list():
            outstr      += 'DISPpmf2                - PMF dispersion curve, with jump correction\n'
        if 'DISPbasic1interp' in self.auxiliary_data.list():
            outstr      += 'DISPbasic1interp        - Interpolated DISPbasic1\n'
        if 'DISPbasic2interp' in self.auxiliary_data.list():
            outstr      += 'DISPbasic2interp        - Interpolated DISPbasic2\n'
        if 'DISPpmf1interp' in self.auxiliary_data.list():
            outstr      += 'DISPpmf1interp          - Interpolated DISPpmf1\n'
        if 'DISPpmf2interp' in self.auxiliary_data.list():
            outstr      += 'DISPpmf2interp          - Interpolated DISPpmf2\n'
        if 'FieldDISPbasic1interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPbasic1interp   - Field data of DISPbasic1\n'
        if 'FieldDISPbasic2interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPbasic2interp   - Field data of DISPbasic2\n'
        if 'FieldDISPpmf1interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPpmf1interp     - Field data of DISPpmf1\n'
        if 'FieldDISPpmf2interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPpmf2interp     - Field data of DISPpmf2\n'
        outstr += '============================================================================================================================================\n'
        print outstr
        return
    
    def _get_lon_lat_arr(self, path, hd=True):
        """Get longitude/latitude array
        """
        minlon                  = self.auxiliary_data['Header'][path].parameters['minlon']
        maxlon                  = self.auxiliary_data['Header'][path].parameters['maxlon']
        minlat                  = self.auxiliary_data['Header'][path].parameters['minlat']
        maxlat                  = self.auxiliary_data['Header'][path].parameters['maxlat']
        if not hd:
            dlon                = self.auxiliary_data['Header'][path].parameters['dlon']
            dlat                = self.auxiliary_data['Header'][path].parameters['dlat']
        else:
            dlon                = self.auxiliary_data['Header'][path].parameters['dlon_HD']
            dlat                = self.auxiliary_data['Header'][path].parameters['dlat_HD']
        self.lons               = np.arange(int((maxlon-minlon)/dlon)+1)*dlon+minlon
        self.lats               = np.arange(int((maxlat-minlat)/dlat)+1)*dlat+minlat
        self.Nlon               = self.lons.size
        self.Nlat               = self.lats.size
        self.lonArr, self.latArr= np.meshgrid(self.lons, self.lats)
        return
    
    def read_ref_dbase(self, inasdfname, phase='P'):
        """
        read radial receiver function data from input ASDF file
        ==========================================================================
        ::: input :::
        inasdfname  - input ASDF file name
        phase       - default - P, P receiver function
        ::: output :::

        ==========================================================================
        """
        indset      = pyasdf.ASDFDataSet(inasdfname)
        #--------------------
        # station inventory
        #--------------------
        wavlst      = indset.waveforms.list()
        self.inv    = indset.waveforms[wavlst[0]].StationXML
        for staid in wavlst[1:]:
            self.inv+= indset.waveforms[staid].StationXML
        self.add_stationxml(self.inv)
        #--------------------
        # ref data
        #--------------------
        for staid in wavlst:
            netcode, stacode    = staid.split('.')
            staid_aux           = netcode+'_'+stacode+'_'+phase
            if indset.auxiliary_data.RefRHScount[staid_aux].parameters['Nhs'] == 0:
                print 'No harmonic stripping data for '+staid
                continue
            ref_header          = {'Nraw': indset.auxiliary_data['RefRHScount'][staid_aux].parameters['Nraw'], \
                                    'Nhs': indset.auxiliary_data['RefRHScount'][staid_aux].parameters['Nhs'], \
                                    'delta': indset.auxiliary_data['RefRHSmodel'][staid_aux]['A0_A1_A2']['A0'].parameters['delta'], \
                                    'npts': indset.auxiliary_data['RefRHSmodel'][staid_aux]['A0_A1_A2']['A0'].parameters['npts']}
            """
            0       - A0 from A0-A1-A2 inversion
            1       - misfit from raw A0+A1+A2
            2       - misfit from binned A0+A1+A2
            3       - weighted misfit from binned A0+A1+A2
            """
            data                = np.zeros((4, ref_header['npts']))
            data[0, :]          = indset.auxiliary_data['RefRHSmodel'][staid_aux]['A0_A1_A2']['A0'].data.value
            data[1, :]          = indset.auxiliary_data['RefRHSmodel'][staid_aux]['A0_A1_A2']['mf_A0_A1_A2_obs'].data.value
            data[2, :]          = indset.auxiliary_data['RefRHSmodel'][staid_aux]['A0_A1_A2']['mf_A0_A1_A2_bin'].data.value
            data[3, :]          = indset.auxiliary_data['RefRHSmodel'][staid_aux]['A0_A1_A2']['wmf_A0_A1_A2_bin'].data.value
            self.add_auxiliary_data(data=data, data_type='RefR', path=staid_aux, parameters=ref_header)
        return
    
    def read_raytomo_dbase(self, inh5fname, runid, dtype='ph', wtype='ray', create_header=True, Tmin=-999, Tmax=999, verbose=False):
        """
        read ray tomography data base
        =================================================================================
        ::: input :::
        inh5fname   - input hdf5 file name
        runid       - id of run for the ray tomography
        dtype       - data type (ph or gr)
        wtype       - wave type (ray or lov)
        Tmin, Tmax  - minimum and maximum period to extract from the tomographic results
        =================================================================================
        """
        if dtype is not 'ph' and dtype is not 'gr':
            raise ValueError('data type can only be ph or gr!')
        if wtype is not 'ray' and wtype is not 'lov':
            raise ValueError('wave type can only be ray or lov!')
        stalst      = self.waveforms.list()
        if len(stalst) == 0:
            print 'Inversion with surface wave datasets only, not added yet!'
            return
        indset          = h5py.File(inh5fname)
        #--------------------------------------------
        # header information from input hdf5 file
        #--------------------------------------------
        dataid          = 'reshaped_qc_run_'+str(runid)
        pers            = indset.attrs['period_array']
        grp             = indset[dataid]
        isotropic       = grp.attrs['isotropic']
        org_grp         = indset['qc_run_'+str(runid)]
        minlon          = indset.attrs['minlon']
        maxlon          = indset.attrs['maxlon']
        minlat          = indset.attrs['minlat']
        maxlat          = indset.attrs['maxlat']
        if isotropic:
            print 'isotropic inversion results do not output gaussian std!'
            return
        dlon_HD         = org_grp.attrs['dlon_HD']
        dlat_HD         = org_grp.attrs['dlat_HD']
        dlon            = org_grp.attrs['dlon']
        dlat            = org_grp.attrs['dlat']
        if create_header:
            inv_header  = {'minlon': minlon, 'maxlon': maxlon, 'minlat': minlat, 'maxlat': maxlat,
                           'dlon': dlon, 'dlat': dlat, 'dlon_HD': dlon_HD, 'dlat_HD': dlat_HD}
            self.add_auxiliary_data(data=np.array([]), data_type='Header', path='raytomo', parameters=inv_header)
        self._get_lon_lat_arr(path='raytomo', hd=True)
        for staid in stalst:
            netcode, stacode    = staid.split('.')
            staid_aux           = netcode+'_'+stacode
            stla, elev, stlo    = self.waveforms[staid].coordinates.values()
            if stlo < 0.:
                stlo            += 360.
            if stla > maxlat or stla < minlat or stlo > maxlon or stlo < minlon:
                print 'WARNING: station: '+ staid+', lat = '+str(stla)+' lon = '+str(stlo)+', out of the range of tomograpic maps!'
                continue
            disp_v              = np.array([])
            disp_un             = np.array([])
            T                   = np.array([])
            #-----------------------------
            # determine the indices
            #-----------------------------
            ind_lon             = np.where(stlo<=self.lons)[0][0]
            find_lon            = ind_lon            
            ind_lat             = np.where(stla<=self.lats)[0][0]
            find_lat            = ind_lat
            # point 1
            distmin, az, baz    = obspy.geodetics.gps2dist_azimuth(stla, stlo, self.lats[ind_lat], self.lons[ind_lon]) # distance is in m
            # point 2
            dist, az, baz       = obspy.geodetics.gps2dist_azimuth(stla, stlo, self.lats[ind_lat], self.lons[ind_lon-1]) # distance is in m
            if dist < distmin:
                find_lon        = ind_lon-1
                distmin         = dist
            # point 3
            dist, az, baz       = obspy.geodetics.gps2dist_azimuth(stla, stlo, self.lats[ind_lat-1], self.lons[ind_lon]) # distance is in m
            if dist < distmin:
                find_lat        = ind_lat-1
                distmin         = dist
            # point 4
            dist, az, baz       = obspy.geodetics.gps2dist_azimuth(stla, stlo, self.lats[ind_lat-1], self.lons[ind_lon-1]) # distance is in m
            if dist < distmin:
                find_lat        = ind_lat-1
                find_lon        = ind_lon-1
                distmin         = dist
            for per in pers:
                if per < Tmin or per > Tmax:
                    continue
                try:
                    pergrp      = grp['%g_sec'%( per )]
                    vel         = pergrp['vel_iso_HD'].value
                    vel_sem     = pergrp['vel_sem_HD'].value
                except KeyError:
                    if verbose:
                        print 'No data for T = '+str(per)+' sec'
                    continue
                T               = np.append(T, per)
                disp_v          = np.append(disp_v, vel[find_lat, find_lon])
                disp_un         = np.append(disp_un, vel_sem[find_lat, find_lon])
            data                = np.zeros((3, T.size))
            data[0, :]          = T[:]
            data[1, :]          = disp_v[:]
            data[2, :]          = disp_un[:]
            disp_header         = {'Np': T.size}
            self.add_auxiliary_data(data=data, data_type='RayDISPcurve', path=wtype+'/'+dtype+'/'+staid_aux, parameters=disp_header)
        indset.close()
        return
    
    def read_moho_depth(self, infname='crsthk.xyz', source='crust_1.0'):
        """
        read crust thickness from a txt file (crust 1.0 model)
        """
        inArr       = np.loadtxt(infname)
        lonArr      = inArr[:, 0]
        lonArr      = lonArr.reshape(lonArr.size/360, 360)
        latArr      = inArr[:, 1]
        latArr      = latArr.reshape(latArr.size/360, 360)
        depthArr    = inArr[:, 2]
        depthArr    = depthArr.reshape(depthArr.size/360, 360)
        stalst      = self.waveforms.list()
        if len(stalst) == 0:
            print 'Inversion with surface wave datasets only, not added yet!'
            return
        for staid in stalst:
            netcode, stacode    = staid.split('.')
            staid_aux           = netcode+'_'+stacode
            stla, elev, stlo    = self.waveforms[staid].coordinates.values()
            if stlo > 180.:
                stlo            -= 360.
            whereArr= np.where((lonArr>=stlo)*(latArr>=stla))
            ind_lat = whereArr[0][-1]
            ind_lon = whereArr[1][0]
            # check
            lon     = lonArr[ind_lat, ind_lon]
            lat     = latArr[ind_lat, ind_lon]
            if abs(lon-stlo) > 1. or abs(lat - stla) > 1.:
                print 'ERROR!',lon,lat,stlo,stla
            depth   = depthArr[ind_lat, ind_lon]
            header  = {'moho_depth': depth, 'data_source': source}
            self.add_auxiliary_data(data=np.array([]), data_type='MohoDepth', path=staid_aux, parameters=header)
        return
    
    def read_sediment_depth(self, infname='sedthk.xyz'):
        """
        read sediment thickness from a txt file (crust 1.0 model)
        """
        inArr   = np.loadtxt(infname)
        lonArr  = inArr[:, 0]
        lonArr  = lonArr.reshape(lonArr.size/360, 360)
        latArr  = inArr[:, 1]
        latArr  = latArr.reshape(latArr.size/360, 360)
        depthArr= inArr[:, 2]
        depthArr= depthArr.reshape(depthArr.size/360, 360)
        stalst                  = self.waveforms.list()
        if len(stalst) == 0:
            print 'Inversion with surface wave datasets only, not added yet!'
            return
        for staid in stalst:
            netcode, stacode    = staid.split('.')
            staid_aux           = netcode+'_'+stacode
            stla, elev, stlo    = self.waveforms[staid].coordinates.values()
            if stlo > 180.:
                stlo            -= 360.
            whereArr= np.where((lonArr>=stlo)*(latArr>=stla))
            ind_lat = whereArr[0][-1]
            ind_lon = whereArr[1][0]
            # check
            lon     = lonArr[ind_lat, ind_lon]
            lat     = latArr[ind_lat, ind_lon]
            if abs(lon-stlo) > 1. or abs(lat - stla) > 1.:
                print 'ERROR!',lon,lat,stlo,stla
            depth   = depthArr[ind_lat, ind_lon]
            header  = {'sedi_depth': depth, 'data_source': 'crust_1.0'}
            self.add_auxiliary_data(data=np.array([]), data_type='SediDepth', path=staid_aux, parameters=header)
        return
    
    def read_CU_model(self, infname='CU_SDT1.0.mod.h5'):
        """
        read reference model from a hdf5 file (CU Global Vs model)
        """
        indset      = h5py.File(infname)
        lons        = np.mgrid[0.:359.:2.]
        lats        = np.mgrid[-88.:89.:2.]
        stalst                  = self.waveforms.list()
        if len(stalst) == 0:
            print 'Inversion with surface wave datasets only, not added yet!'
            return
        for staid in stalst:
            netcode, stacode    = staid.split('.')
            staid_aux           = netcode+'_'+stacode
            stla, elev, stlo    = self.waveforms[staid].coordinates.values()
            if stlo < 0.:
                stlo            += 360.
            try:
                ind_lon         = np.where(lons>=stlo)[0][0]
            except:
                ind_lon         = lons.size - 1
            try:
                ind_lat         = np.where(lats>=stla)[0][0]
            except:
                ind_lat         = lats.size - 1
            pind                = 0
            while(True):
                if pind == 0:
                    data        = indset[str(lons[ind_lon])+'_'+str(lats[ind_lat])].value
                    if data[0, 1] != 0:
                        outlon  = lons[ind_lon]
                        outlat  = lats[ind_lat]
                        break
                    pind        += 1
                    continue
                data            = indset[str(lons[ind_lon+pind])+'_'+str(lats[ind_lat])].value
                if data[0, 1] != 0:
                    outlon      = lons[ind_lon+pind]
                    outlat      = lats[ind_lat]
                    break
                data            = indset[str(lons[ind_lon-pind])+'_'+str(lats[ind_lat])].value
                if data[0, 1] != 0:
                    outlon      = lons[ind_lon-pind]
                    outlat      = lats[ind_lat]
                    break
                data            = indset[str(lons[ind_lon])+'_'+str(lats[ind_lat+pind])].value
                if data[0, 1] != 0:
                    outlon      = lons[ind_lon]
                    outlat      = lats[ind_lat+pind]
                    break
                data            = indset[str(lons[ind_lon])+'_'+str(lats[ind_lat-pind])].value
                if data[0, 1] != 0:
                    outlon      = lons[ind_lon]
                    outlat      = lats[ind_lat-pind]
                    break
                data            = indset[str(lons[ind_lon-pind])+'_'+str(lats[ind_lat-pind])].value
                if data[0, 1] != 0:
                    outlon      = lons[ind_lon-pind]
                    outlat      = lats[ind_lat-pind]
                    break
                data            = indset[str(lons[ind_lon-pind])+'_'+str(lats[ind_lat+pind])].value
                if data[0, 1] != 0:
                    outlon      = lons[ind_lon-pind]
                    outlat      = lats[ind_lat+pind]
                    break
                data            = indset[str(lons[ind_lon+pind])+'_'+str(lats[ind_lat-pind])].value
                if data[0, 1] != 0:
                    outlon      = lons[ind_lon+pind]
                    outlat      = lats[ind_lat-pind]
                    break
                data            = indset[str(lons[ind_lon+pind])+'_'+str(lats[ind_lat+pind])].value
                if data[0, 1] != 0:
                    outlon      = lons[ind_lon+pind]
                    outlat      = lats[ind_lat+pind]
                    break
                pind            += 1
            if pind >= 5:
                print 'WARNING: Large differences in the finalized points: lon = '+str(outlon)+', lat = '+str(outlat)\
                    + ', station: '+staid+' stlo = '+str(stlo) + ', stla = '+str(stla)
            # print outlon, outlat, stlo, stla, pind
            header  = {'data_source': 'CU_SDT',\
                       'depth': 0, 'vs': 1, 'vsv': 2, 'vsh': 3, 'vsmin': 4, 'vsvmin': 5, 'vshmin': 6, \
                       'vsmax': 7, 'vsvmax': 8, 'vshmax': 9}
            self.add_auxiliary_data(data=data, data_type='ReferenceModel', path=staid_aux, parameters=header)
        return
    
    def mc_inv_iso(self, instafname=None, ref=True, phase=True, group=False, outdir='./workingdir', wdisp=0.2, rffactor=40.,\
                   monoc=True, verbose=False, step4uwalk=1500, numbrun=10000, subsize=1000, nprocess=None, parallel=True):
        """
        Bayesian Monte Carlo joint inversion of receiver function and surface wave data for an isotropic model
        ==================================================================================================================
        ::: input :::
        instafname  - input station list file indicating the stations for joint inversion
        ref         - include receiver function data or not
        phase/group - include phase/group velocity dispersion data or not
        outdir      - output directory
        wdisp       - weight of dispersion curve data (0. ~ 1.)
        rffactor    - factor for downweighting the misfit for likelihood computation of rf
        monoc       - require monotonical increase in the crust or not
        step4uwalk  - step interval for uniform random walk in the parameter space
        numbrun     - total number of runs
        subsize     - size of subsets, used if the number of elements in the parallel list is too large to avoid deadlock
        nprocess    - number of process
        parallel    - run the inversion in parallel or not 
        ==================================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if instafname is None:
            stalst  = self.waveforms.list()
        else:
            stalst  = []
            with open(instafname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    if sline[2] == '1':
                        stalst.append(sline[0])
        if not ref and wdisp != 1.:
            wdisp   = 1.
            print 'wdisp is forced to be 1. for inversion without receiver function data'
        if phase and group:
            dispdtype   = 'both'
        elif phase and not group:
            dispdtype   = 'ph'
        else:
            dispdtype   = 'gr'
        ista        = 0
        Nsta        = len(stalst)
        for staid in stalst:
            netcode, stacode    = staid.split('.')
            staid_aux           = netcode+'_'+stacode
            stla, elev, stlo    = self.waveforms[staid].coordinates.values()
            #-----------------------------
            # get data
            #-----------------------------
            vpr                 = vprofile.vprofile1d()
            if phase:
                try:
                    indisp      = self.auxiliary_data['RayDISPcurve']['ray']['ph'][staid_aux].data.value
                    vpr.get_disp(indata=indisp, dtype='ph', wtype='ray')
                except KeyError:
                    print 'WARNING: No phase dispersion data for station: '+staid
            if group:
                try:
                    indisp      = self.auxiliary_data['RayDISPcurve']['ray']['gr'][staid_aux].data.value
                    vpr.get_disp(indata=indisp, dtype='gr', wtype='ray')
                except KeyError:
                    print 'WARNING: No group dispersion data for station: '+staid
            if vpr.data.dispR.npper == 0 and vpr.data.dispR.ngper == 0:
                print 'WARNING: No dispersion data for station: '+staid 
                continue
            if ref:
                try:
                    inrf        = self.auxiliary_data['RefR'][staid_aux+'_P'].data.value
                    N           = self.auxiliary_data['RefR'][staid_aux+'_P'].parameters['npts']
                    dt          = self.auxiliary_data['RefR'][staid_aux+'_P'].parameters['delta']
                    indata      = np.zeros((3, N))
                    indata[0, :]= np.arange(N)*dt
                    indata[1, :]= inrf[0, :]
                    indata[2, :]= inrf[3, :]
                    vpr.get_rf(indata = indata)
                except KeyError:
                    print 'WARNING: No receiver function data for station: '+staid
            #-----------------------------
            # initial model parameters
            #-----------------------------
            vsdata              = self.auxiliary_data['ReferenceModel'][staid_aux].data.value
            mohodepth           = self.auxiliary_data['MohoDepth'][staid_aux].parameters['moho_depth']
            seddepth            = self.auxiliary_data['SediDepth'][staid_aux].parameters['sedi_depth']
            vpr.model.isomod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], mohodepth=mohodepth, seddepth=seddepth, maxdepth=200.)
            vpr.getpara()
            ista                += 1
            # if staid != 'AK.HDA': continue
            # return vpr
            print '--- Joint MC inversion for station: '+staid+' '+str(ista)+'/'+str(Nsta)
            if parallel:
                vpr.mc_joint_inv_iso_mp(outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                   monoc=monoc, pfx=staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun, subsize=subsize, nprocess=nprocess)
            else:
                vpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                   monoc=monoc, pfx=staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            # vpr.mc_joint_inv_iso(outdir=outdir, wdisp=wdisp, rffactor=rffactor,\
            #        monoc=monoc, pfx=staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            # if staid == 'AK.COLD':
            #     return vpr
        return
    
class invhdf5(h5py.File):
    """ An object to for MCMC inversion based on HDF5 database
    =================================================================================================================
    version history:
           - first version
    =================================================================================================================
    """
    def print_info(self):
        """
        print information of the dataset.
        """
        outstr  = '================================================= Ambient Noise Cross-correlation Database =================================================\n'
        outstr  += self.__str__()+'\n'
        outstr  += '--------------------------------------------------------------------------------------------------------------------------------------------\n'
        if 'NoiseXcorr' in self.auxiliary_data.list():
            outstr      += 'NoiseXcorr              - Cross-correlation seismogram\n'
        if 'StaInfo' in self.auxiliary_data.list():
            outstr      += 'StaInfo                 - Auxiliary station information\n'
        if 'DISPbasic1' in self.auxiliary_data.list():
            outstr      += 'DISPbasic1              - Basic dispersion curve, no jump correction\n'
        if 'DISPbasic2' in self.auxiliary_data.list():
            outstr      += 'DISPbasic2              - Basic dispersion curve, with jump correction\n'
        if 'DISPpmf1' in self.auxiliary_data.list():
            outstr      += 'DISPpmf1                - PMF dispersion curve, no jump correction\n'
        if 'DISPpmf2' in self.auxiliary_data.list():
            outstr      += 'DISPpmf2                - PMF dispersion curve, with jump correction\n'
        if 'DISPbasic1interp' in self.auxiliary_data.list():
            outstr      += 'DISPbasic1interp        - Interpolated DISPbasic1\n'
        if 'DISPbasic2interp' in self.auxiliary_data.list():
            outstr      += 'DISPbasic2interp        - Interpolated DISPbasic2\n'
        if 'DISPpmf1interp' in self.auxiliary_data.list():
            outstr      += 'DISPpmf1interp          - Interpolated DISPpmf1\n'
        if 'DISPpmf2interp' in self.auxiliary_data.list():
            outstr      += 'DISPpmf2interp          - Interpolated DISPpmf2\n'
        if 'FieldDISPbasic1interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPbasic1interp   - Field data of DISPbasic1\n'
        if 'FieldDISPbasic2interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPbasic2interp   - Field data of DISPbasic2\n'
        if 'FieldDISPpmf1interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPpmf1interp     - Field data of DISPpmf1\n'
        if 'FieldDISPpmf2interp' in self.auxiliary_data.list():
            outstr      += 'FieldDISPpmf2interp     - Field data of DISPpmf2\n'
        outstr += '============================================================================================================================================\n'
        print outstr
        return
    
    def _get_lon_lat_arr(self):
        """Get longitude/latitude array
        """
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        dlon            = self.attrs['dlon']
        dlat            = self.attrs['dlat']
        self.lons       = np.arange(int((maxlon-minlon)/dlon)+1)*dlon+minlon
        self.lats       = np.arange(int((maxlat-minlat)/dlat)+1)*dlat+minlat
        self.Nlon       = self.lons.size
        self.Nlat       = self.lats.size
        self.lonArr, self.latArr= np.meshgrid(self.lons, self.lats)
        return
    
    def read_raytomo_dbase(self, inh5fname, runid, dtype='ph', wtype='ray', create_header=True, Tmin=-999, Tmax=999, verbose=False, res='LD'):
        """
        read ray tomography data base
        =================================================================================
        ::: input :::
        inh5fname   - input hdf5 file name
        runid       - id of run for the ray tomography
        dtype       - data type (ph or gr)
        wtype       - wave type (ray or lov)
        Tmin, Tmax  - minimum and maximum period to extract from the tomographic results
        res         - resolution for grid points, default is LD, low-definition
        =================================================================================
        """
        if dtype is not 'ph' and dtype is not 'gr':
            raise ValueError('data type can only be ph or gr!')
        if wtype is not 'ray' and wtype is not 'lov':
            raise ValueError('wave type can only be ray or lov!')
        indset          = h5py.File(inh5fname)
        #--------------------------------------------
        # header information from input hdf5 file
        #--------------------------------------------
        dataid          = 'reshaped_qc_run_'+str(runid)
        pers            = indset.attrs['period_array']
        grp             = indset[dataid]
        isotropic       = grp.attrs['isotropic']
        org_grp         = indset['qc_run_'+str(runid)]
        minlon          = indset.attrs['minlon']
        maxlon          = indset.attrs['maxlon']
        minlat          = indset.attrs['minlat']
        maxlat          = indset.attrs['maxlat']
        if isotropic:
            print 'isotropic inversion results do not output gaussian std!'
            return
        if res == 'LD':
            sfx         = '_LD'
        elif res == 'HD':
            sfx         = '_HD'
        else:
            sfx         = ''
        dlon_interp     = org_grp.attrs['dlon'+sfx]
        dlat_interp     = org_grp.attrs['dlat'+sfx]
        dlon            = org_grp.attrs['dlon']
        dlat            = org_grp.attrs['dlat']
        if sfx == '':
            mask        = indset[dataid+'/mask1']
        else:
            mask        = indset[dataid+'/mask'+sfx]
        if create_header:
            self.attrs.create(name = 'minlon', data=minlon, dtype='f')
            self.attrs.create(name = 'maxlon', data=maxlon, dtype='f')
            self.attrs.create(name = 'minlat', data=minlat, dtype='f')
            self.attrs.create(name = 'maxlat', data=maxlat, dtype='f')
            self.attrs.create(name = 'dlon', data=dlon_interp)
            self.attrs.create(name = 'dlat', data=dlon_interp)
        self._get_lon_lat_arr()
        self.attrs.create(name='mask', data = mask)
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                data_str    = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                group       = self.create_group( name = data_str )
                disp_v      = np.array([])
                disp_un     = np.array([])
                T           = np.array([])
                for per in pers:
                    if per < Tmin or per > Tmax:
                        continue
                    try:
                        pergrp      = grp['%g_sec'%( per )]
                        vel         = pergrp['vel_iso'+sfx].value
                        vel_sem     = pergrp['vel_sem'+sfx].value
                    except KeyError:
                        if verbose:
                            print 'No data for T = '+str(per)+' sec'
                        continue
                    T               = np.append(T, per)
                    disp_v          = np.append(disp_v, vel[ilat, ilon])
                    disp_un         = np.append(disp_un, vel_sem[ilat, ilon])
                data                = np.zeros((3, T.size))
                data[0, :]          = T[:]
                data[1, :]          = disp_v[:]
                data[2, :]          = disp_un[:]
                group.create_dataset(name='disp_'+dtype+'_'+wtype, data=data)
        indset.close()
        return
    
    def read_moho_depth(self, infname='crsthk.xyz', source='crust_1.0'):
        """
        read crust thickness from a txt file (crust 1.0 model)
        """
        inArr       = np.loadtxt(infname)
        lonArr      = inArr[:, 0]
        lonArr      = lonArr.reshape(lonArr.size/360, 360)
        latArr      = inArr[:, 1]
        latArr      = latArr.reshape(latArr.size/360, 360)
        depthArr    = inArr[:, 2]
        depthArr    = depthArr.reshape(depthArr.size/360, 360)
        for grp_id in self.keys():
            grp     = self[grp_id]
            split_id= grp_id.split('_')
            grd_lon = float(split_id[0])
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat = float(split_id[1])
            whereArr= np.where((lonArr>=grd_lon)*(latArr>=grd_lat))
            ind_lat = whereArr[0][-1]
            ind_lon = whereArr[1][0]
            # check
            lon     = lonArr[ind_lat, ind_lon]
            lat     = latArr[ind_lat, ind_lon]
            if abs(lon-grd_lon) > 1. or abs(lat - grd_lat) > 1.:
                print 'ERROR!', lon, lat, grd_lon, grd_lat
            depth   = depthArr[ind_lat, ind_lon]
            grp.attrs.create(name='crust_thk', data=depth)
            grp.attrs.create(name='crust_thk_source', data=source)
        return
    
    def read_sediment_depth(self, infname='sedthk.xyz', source='crust_1.0'):
        """
        read sediment thickness from a txt file (crust 1.0 model)
        """
        inArr       = np.loadtxt(infname)
        lonArr      = inArr[:, 0]
        lonArr      = lonArr.reshape(lonArr.size/360, 360)
        latArr      = inArr[:, 1]
        latArr      = latArr.reshape(latArr.size/360, 360)
        depthArr    = inArr[:, 2]
        depthArr    = depthArr.reshape(depthArr.size/360, 360)
        for grp_id in self.keys():
            grp     = self[grp_id]
            split_id= grp_id.split('_')
            grd_lon = float(split_id[0])
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat = float(split_id[1])
            whereArr= np.where((lonArr>=grd_lon)*(latArr>=grd_lat))
            ind_lat = whereArr[0][-1]
            ind_lon = whereArr[1][0]
            # check
            lon     = lonArr[ind_lat, ind_lon]
            lat     = latArr[ind_lat, ind_lon]
            if abs(lon-grd_lon) > 1. or abs(lat - grd_lat) > 1.:
                print 'ERROR!', lon, lat, grd_lon, grd_lat
            depth   = depthArr[ind_lat, ind_lon]
            grp.attrs.create(name='sedi_thk', data=depth)
            grp.attrs.create(name='sedi_thk_source', data=source)
        return
    
    def read_CU_model(self, infname='CU_SDT1.0.mod.h5'):
        """
        read reference model from a hdf5 file (CU Global Vs model)
        """
        indset      = h5py.File(infname)
        lons        = np.mgrid[0.:359.:2.]
        lats        = np.mgrid[-88.:89.:2.]
        for grp_id in self.keys():
            grp         = self[grp_id]
            split_id    = grp_id.split('_')
            grd_lon     = float(split_id[0])
            if grd_lon < 0.:
                grd_lon += 360.
            grd_lat = float(split_id[1])
            try:
                ind_lon         = np.where(lons>=grd_lon)[0][0]
            except:
                ind_lon         = lons.size - 1
            try:
                ind_lat         = np.where(lats>=grd_lat)[0][0]
            except:
                ind_lat         = lats.size - 1
            if lons[ind_lon] - grd_lon > 1. and ind_lon > 0:
                ind_lon         -= 1
            if lats[ind_lat] - grd_lat > 1. and ind_lat > 0:
                ind_lat         -= 1
            if abs(lons[ind_lon] - grd_lon) > 1. or abs(lats[ind_lat] - grd_lat) > 1.:
                print 'ERROR!', lons[ind_lon], lats[ind_lat] , grd_lon, grd_lat
            data        = indset[str(lons[ind_lon])+'_'+str(lats[ind_lat])].value
            grp.create_dataset(name='reference_vs', data=data)
        indset.close()
        return
    
    def read_etopo(self, infname='../ETOPO2v2g_f4.nc', download=True, delete=True, source='etopo2'):
        """
        read topography data from etopo2 
        """
        from netCDF4 import Dataset
        try:
            etopodbase  = Dataset(infname)
        except IOError:
            if download:
                url     = 'https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2g/netCDF/ETOPO2v2g_f4_netCDF.zip'
                os.system('wget '+url)
                os.system('unzip ETOPO2v2g_f4_netCDF.zip')
                if delete:
                    os.remove('ETOPO2v2g_f4_netCDF.zip')
                etopodbase  = Dataset('./ETOPO2v2g_f4.nc')
            else:
                print 'No etopo data!'
                return
        etopo       = etopodbase.variables['z'][:]
        lons        = etopodbase.variables['x'][:]
        lats        = etopodbase.variables['y'][:]
        for grp_id in self.keys():
            grp     = self[grp_id]
            split_id= grp_id.split('_')
            grd_lon = float(split_id[0])
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat = float(split_id[1])
            try:
                ind_lon         = np.where(lons>=grd_lon)[0][0]
            except:
                ind_lon         = lons.size - 1
            try:
                ind_lat         = np.where(lats>=grd_lat)[0][0]
            except:
                ind_lat         = lats.size - 1
            if lons[ind_lon] - grd_lon > (1./60.):
                ind_lon         -= 1
            if lats[ind_lat] - grd_lat > (1./60.):
                ind_lat         -= 1
            if abs(lons[ind_lon] - grd_lon) > 1./60. or abs(lats[ind_lat] - grd_lat) > 1./60.:
                print 'ERROR!', lons[ind_lon], lats[ind_lat] , grd_lon, grd_lat
            z                   = etopo[ind_lat, ind_lon]/1000. # convert to km
            grp.attrs.create(name='topo', data=z)
            grp.attrs.create(name='etopo_source', data=source)
        if delete and os.path.isfile('./ETOPO2v2g_f4.nc'):
            os.remove('./ETOPO2v2g_f4.nc')
        return
    
    def mc_inv_iso(self, ingrdfname=None, phase=True, group=False, outdir='./workingdir', vp_water=1.5,
                   monoc=True, verbose=False, step4uwalk=1500, numbrun=10000, subsize=1000, nprocess=None, parallel=True):
        """
        Bayesian Monte Carlo inversion of surface wave data for an isotropic model
        ==================================================================================================================
        ::: input :::
        instafname  - input station list file indicating the stations for joint inversion
        phase/group - include phase/group velocity dispersion data or not
        outdir      - output directory
        vp_water    - P wave velocity in water layer (default - 1.5 km/s)
        monoc       - require monotonical increase in the crust or not
        step4uwalk  - step interval for uniform random walk in the parameter space
        numbrun     - total number of runs
        subsize     - size of subsets, used if the number of elements in the parallel list is too large to avoid deadlock
        nprocess    - number of process
        parallel    - run the inversion in parallel or not 
        ==================================================================================================================
        """
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        if ingrdfname is None:
            grdlst  = self.keys()
        else:
            grdlst  = []
            with open(ingrdfname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    lon     = float(sline[0])
                    if lon < 0.:
                        lon += 360.
                    if sline[2] == '1':
                        grdlst.append(str(lon)+'_'+sline[1])
        if phase and group:
            dispdtype   = 'both'
        elif phase and not group:
            dispdtype   = 'ph'
        else:
            dispdtype   = 'gr'
        igrd        = 0
        Ngrd        = len(grdlst)


        for staid in stalst:
            netcode, stacode    = staid.split('.')
            staid_aux           = netcode+'_'+stacode
            stla, elev, stlo    = self.waveforms[staid].coordinates.values()
            #-----------------------------
            # get data
            #-----------------------------
            vpr                 = vprofile.vprofile1d()
            if phase:
                try:
                    indisp      = self.auxiliary_data['RayDISPcurve']['ray']['ph'][staid_aux].data.value
                    vpr.get_disp(indata=indisp, dtype='ph', wtype='ray')
                except KeyError:
                    print 'WARNING: No phase dispersion data for station: '+staid
            if group:
                try:
                    indisp      = self.auxiliary_data['RayDISPcurve']['ray']['gr'][staid_aux].data.value
                    vpr.get_disp(indata=indisp, dtype='gr', wtype='ray')
                except KeyError:
                    print 'WARNING: No group dispersion data for station: '+staid
            if vpr.data.dispR.npper == 0 and vpr.data.dispR.ngper == 0:
                print 'WARNING: No dispersion data for station: '+staid 
                continue
            if ref:
                try:
                    inrf        = self.auxiliary_data['RefR'][staid_aux+'_P'].data.value
                    N           = self.auxiliary_data['RefR'][staid_aux+'_P'].parameters['npts']
                    dt          = self.auxiliary_data['RefR'][staid_aux+'_P'].parameters['delta']
                    indata      = np.zeros((3, N))
                    indata[0, :]= np.arange(N)*dt
                    indata[1, :]= inrf[0, :]
                    indata[2, :]= inrf[3, :]
                    vpr.get_rf(indata = indata)
                except KeyError:
                    print 'WARNING: No receiver function data for station: '+staid
            #-----------------------------
            # initial model parameters
            #-----------------------------
            vsdata              = self.auxiliary_data['ReferenceModel'][staid_aux].data.value
            mohodepth           = self.auxiliary_data['MohoDepth'][staid_aux].parameters['moho_depth']
            seddepth            = self.auxiliary_data['SediDepth'][staid_aux].parameters['sedi_depth']
            vpr.model.isomod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], mohodepth=mohodepth, seddepth=seddepth, maxdepth=200.)
            vpr.getpara()
            ista                += 1
            # if staid != 'AK.HDA': continue
            # return vpr
            print '--- Joint MC inversion for station: '+staid+' '+str(ista)+'/'+str(Nsta)
            if parallel:
                vpr.mc_joint_inv_iso_mp(outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                   monoc=monoc, pfx=staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun, subsize=subsize, nprocess=nprocess)
            else:
                vpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                   monoc=monoc, pfx=staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            # vpr.mc_joint_inv_iso(outdir=outdir, wdisp=wdisp, rffactor=rffactor,\
            #        monoc=monoc, pfx=staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            # if staid == 'AK.COLD':
            #     return vpr
        return
    
# # #             
# # #     def mc_inv_iso_mp(self, instafname=None, ref=True, phase=True, group=False, outdir='./workingdir', dispdtype='ph', wdisp=0.2, rffactor=40.,\
# # #                    monoc=True, verbose=False, step4uwalk=2500, numbrun=10000, subsize=1000, nprocess=None):
# # #         if not os.path.isdir(outdir):
# # #             os.makedirs(outdir)
# # #         if instafname is None:
# # #             stalst  = self.waveforms.list()
# # #         else:
# # #             stalst  = []
# # #             with open(instafname, 'r') as fid:
# # #                 for line in fid.readlines():
# # #                     sline   = line.split()
# # #                     if sline[2] == '1':
# # #                         stalst.append(sline[0])
# # #         #-------------------------
# # #         # prepare data
# # #         #-------------------------
# # #         vpr_lst = []
# # #         ista    = 0
# # #         Nsta    = len(stalst)
# # #         for staid in stalst:
# # #             netcode, stacode    = staid.split('.')
# # #             staid_aux           = netcode+'_'+stacode
# # #             stla, elev, stlo    = self.waveforms[staid].coordinates.values()
# # #             #-----------------------------
# # #             # get data
# # #             #-----------------------------
# # #             vpr                 = vprofile.vprofile1d()
# # #             if phase:
# # #                 try:
# # #                     indisp      = self.auxiliary_data['RayDISPcurve']['ray']['ph'][staid_aux].data.value
# # #                     vpr.get_disp(indata=indisp, dtype='ph', wtype='ray')
# # #                 except KeyError:
# # #                     print 'WARNING: No phase dispersion data for station: '+staid
# # #             if group:
# # #                 try:
# # #                     indisp      = self.auxiliary_data['RayDISPcurve']['ray']['gr'][staid_aux].data.value
# # #                     vpr.get_disp(indata=indisp, dtype='gr', wtype='ray')
# # #                 except KeyError:
# # #                     print 'WARNING: No group dispersion data for station: '+staid
# # #             if vpr.data.dispR.npper == 0 and vpr.data.dispR.ngper == 0:
# # #                 print 'WARNING: No dispersion data for station: '+staid 
# # #                 continue
# # #             if ref:
# # #                 try:
# # #                     inrf        = self.auxiliary_data['RefR'][staid_aux+'_P'].data.value
# # #                     N           = self.auxiliary_data['RefR'][staid_aux+'_P'].parameters['npts']
# # #                     dt          = self.auxiliary_data['RefR'][staid_aux+'_P'].parameters['delta']
# # #                     indata      = np.zeros((3, N))
# # #                     indata[0, :]= np.arange(N)*dt
# # #                     indata[1, :]= inrf[0, :]
# # #                     indata[2, :]= inrf[3, :]
# # #                     vpr.get_rf(indata = indata)
# # #                 except KeyError:
# # #                     print 'WARNING: No phase dispersion data for station: '+staid
# # #             #-----------------------------
# # #             # initial model parameters
# # #             #-----------------------------
# # #             vsdata              = self.auxiliary_data['ReferenceModel'][staid_aux].data.value
# # #             mohodepth           = self.auxiliary_data['MohoDepth'][staid_aux].parameters['moho_depth']
# # #             seddepth            = self.auxiliary_data['SediDepth'][staid_aux].parameters['sedi_depth']
# # #             vpr.model.isomod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], mohodepth=mohodepth, seddepth=seddepth, maxdepth=200.)
# # #             vpr.getpara()
# # #             vpr.staid           = staid
# # #             ista                += 1
# # #             vpr_lst.append(vpr)
# # #         #----------------------------------------
# # #         # Joint inversion with multiprocessing
# # #         #----------------------------------------
# # #         print 'Start MC joint inversion, '+time.ctime()
# # #         stime   = time.time()
# # #         if Nsta > subsize:
# # #             Nsub                = int(len(vpr_lst)/subsize)
# # #             for isub in xrange(Nsub):
# # #                 print 'Subset:', isub,'in',Nsub,'sets'
# # #                 cvpr_lst        = vpr_lst[isub*subsize:(isub+1)*subsize]
# # #                 MCINV           = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
# # #                                     monoc=monoc, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
# # #                 pool            = multiprocessing.Pool(processes=nprocess)
# # #                 pool.map(MCINV, cvpr_lst) #make our results with a map call
# # #                 pool.close() #we are not adding any more processes
# # #                 pool.join() #tell it to wait until all threads are done before going on
# # #             cvpr_lst            = vpr_lst[(isub+1)*subsize:]
# # #             MCINV               = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
# # #                                     monoc=monoc, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
# # #             pool                = multiprocessing.Pool(processes=nprocess)
# # #             pool.map(MCINV, cvpr_lst) #make our results with a map call
# # #             pool.close() #we are not adding any more processes
# # #             pool.join() #tell it to wait until all threads are done before going on
# # #         else:
# # #             MCINV               = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
# # #                                     monoc=monoc, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
# # #             pool                = multiprocessing.Pool(processes=nprocess)
# # #             pool.map(MCINV, vpr_lst) #make our results with a map call
# # #             pool.close() #we are not adding any more processes
# # #             pool.join() #tell it to wait until all threads are done before going on
# # #             
# # #             # if staid != 'AK.MCK': continue
# # #             # print '--- Joint MC inversion for station: '+staid+' '+str(ista)+'/'+str(Nsta)
# # #             # vpr.mc_joint_inv_iso(outdir=outdir, pfx = staid, rffactor=5., wdisp=0.1)
# # #             # vpr.mc_joint_inv_iso(outdir=outdir, pfx = staid)
# # #             # if staid == 'AK.COLD':
# # #             #     return vpr
# # #         print 'End MC joint inversion, '+time.ctime()
# # #         etime   = time.time()
# # #         print 'Elapsed time: '+str(etime-stime)+' secs'
# # #         return
# # #     
# # # 
# # #     
# # # 
# # # def mc4mp(invpr, outdir, dispdtype, wdisp, rffactor, monoc, verbose, step4uwalk, numbrun):
# # #     print '--- Joint MC inversion for station: '+invpr.staid
# # #     invpr.mc_joint_inv_iso(outdir=outdir, wdisp=wdisp, rffactor=rffactor,\
# # #                    monoc=monoc, pfx=invpr.staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
# # #     return


    