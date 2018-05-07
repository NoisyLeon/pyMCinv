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
    
    def read_ref_dbase(self, inasdfname, phase='P', reftype='R'):
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
            ref_header          = {'Nraw': indset.auxiliary_data['Ref'+reftype+'HScount'][staid_aux].parameters['Nraw'], \
                                    'Nhs': indset.auxiliary_data['Ref'+reftype+'HScount'][staid_aux].parameters['Nhs'], \
                                    'delta': indset.auxiliary_data['Ref'+reftype+'HSmodel'][staid_aux]['A0_A1_A2']['A0'].parameters['delta'], \
                                    'npts': indset.auxiliary_data['Ref'+reftype+'HSmodel'][staid_aux]['A0_A1_A2']['A0'].parameters['npts']}
            """
            0       - A0 from A0-A1-A2 inversion
            1       - misfit from raw A0+A1+A2
            2       - misfit from binned A0+A1+A2
            3       - weighted misfit from binned A0+A1+A2
            """
            data                = np.zeros((4, ref_header['npts']))
            data[0, :]          = indset.auxiliary_data['Ref'+reftype+'HSmodel'][staid_aux]['A0_A1_A2']['A0'].data.value
            data[1, :]          = indset.auxiliary_data['Ref'+reftype+'HSmodel'][staid_aux]['A0_A1_A2']['mf_A0_A1_A2_obs'].data.value
            data[2, :]          = indset.auxiliary_data['Ref'+reftype+'HSmodel'][staid_aux]['A0_A1_A2']['mf_A0_A1_A2_bin'].data.value
            data[3, :]          = indset.auxiliary_data['Ref'+reftype+'HSmodel'][staid_aux]['A0_A1_A2']['wmf_A0_A1_A2_bin'].data.value
            self.add_auxiliary_data(data=data, data_type='Ref'+reftype, path=staid_aux, parameters=ref_header)
        return
    
    def read_raytomo_dbase(self, inh5fname, runid, dtype='ph', wtype='ray', create_header=True, Tmin=-999, Tmax=999, verbose=False):
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
    
    def read_moho_depth(self, infname='crsthk.xyz'):
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
            header  = {'moho_depth': depth, 'data_source': 'crust_1.0'}
            self.add_auxiliary_data(data=np.array([]), data_type='MohoDepth', path=staid_aux, parameters=header)
        return
    
    def read_sediment_depth(self, infname='sedthk.xyz'):
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
    
    def mc_inv_iso(self, instafname=None, ref=True, phase=True, group=False, outdir='./workingdir', dispdtype='ph', wdisp=0.2, rffactor=40.,\
                   monoc=True, verbose=False, step4uwalk=2500, numbrun=10000):
        if instafname is None:
            stalst  = self.waveforms.list()
        else:
            stalst  = []
            with open(instafname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    if sline[2] == '1':
                        stalst.append(sline[0])
            # return stalst
        ista    = 0
        Nsta    = len(stalst)
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
                    print 'WARNING: No phase dispersion data for station: '+staid
            #-----------------------------
            # initial model parameters
            #-----------------------------
            vsdata              = self.auxiliary_data['ReferenceModel'][staid_aux].data.value
            mohodepth           = self.auxiliary_data['MohoDepth'][staid_aux].parameters['moho_depth']
            seddepth            = self.auxiliary_data['SediDepth'][staid_aux].parameters['sedi_depth']
            vpr.model.isomod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], mohodepth=mohodepth, seddepth=seddepth, maxdepth=200.)
            vpr.getpara()
            
            ista                += 1
            if staid != 'AK.MCK': continue
            print '--- Joint MC inversion for station: '+staid+' '+str(ista)+'/'+str(Nsta)
            vpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                   monoc=monoc, pfx=staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            # vpr.mc_joint_inv_iso(outdir=outdir, wdisp=wdisp, rffactor=rffactor,\
            #        monoc=monoc, pfx=staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            # if staid == 'AK.COLD':
            #     return vpr
            
    def mc_inv_iso_mp(self, instafname=None, ref=True, phase=True, group=False, outdir='./workingdir', dispdtype='ph', wdisp=0.2, rffactor=40.,\
                   monoc=True, verbose=False, step4uwalk=2500, numbrun=10000, subsize=1000, nprocess=None):
        if instafname is None:
            stalst  = self.waveforms.list()
        else:
            stalst  = []
            with open(instafname, 'r') as fid:
                for line in fid.readlines():
                    sline   = line.split()
                    if sline[2] == '1':
                        stalst.append(sline[0])
        #-------------------------
        # prepare data
        #-------------------------
        vpr_lst = []
        ista    = 0
        Nsta    = len(stalst)
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
                    print 'WARNING: No phase dispersion data for station: '+staid
            #-----------------------------
            # initial model parameters
            #-----------------------------
            vsdata              = self.auxiliary_data['ReferenceModel'][staid_aux].data.value
            mohodepth           = self.auxiliary_data['MohoDepth'][staid_aux].parameters['moho_depth']
            seddepth            = self.auxiliary_data['SediDepth'][staid_aux].parameters['sedi_depth']
            vpr.model.isomod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], mohodepth=mohodepth, seddepth=seddepth, maxdepth=200.)
            vpr.getpara()
            vpr.staid           = staid
            ista                += 1
            vpr_lst.append(vpr)
        #----------------------------------------
        # Joint inversion with multiprocessing
        #----------------------------------------
        print 'Start MC joint inversion, '+time.ctime()
        stime   = time.time()
        if Nsta > subsize:
            Nsub                = int(len(vpr_lst)/subsize)
            for isub in xrange(Nsub):
                print 'Subset:', isub,'in',Nsub,'sets'
                cvpr_lst        = vpr_lst[isub*subsize:(isub+1)*subsize]
                MCINV           = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                                    monoc=monoc, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
                pool            = multiprocessing.Pool(processes=nprocess)
                pool.map(MCINV, cvpr_lst) #make our results with a map call
                pool.close() #we are not adding any more processes
                pool.join() #tell it to wait until all threads are done before going on
            cvpr_lst            = vpr_lst[(isub+1)*subsize:]
            MCINV               = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                                    monoc=monoc, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            pool                = multiprocessing.Pool(processes=nprocess)
            pool.map(MCINV, cvpr_lst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on
        else:
            MCINV               = partial(mc4mp, outdir=outdir, dispdtype=dispdtype, wdisp=wdisp, rffactor=rffactor,\
                                    monoc=monoc, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            pool                = multiprocessing.Pool(processes=nprocess)
            pool.map(MCINV, vpr_lst) #make our results with a map call
            pool.close() #we are not adding any more processes
            pool.join() #tell it to wait until all threads are done before going on

            # if staid != 'AK.MCK': continue
            # print '--- Joint MC inversion for station: '+staid+' '+str(ista)+'/'+str(Nsta)
            # vpr.mc_joint_inv_iso(outdir=outdir, pfx = staid, rffactor=5., wdisp=0.1)
            # vpr.mc_joint_inv_iso(outdir=outdir, pfx = staid)
            # if staid == 'AK.COLD':
            #     return vpr
        print 'End MC joint inversion, '+time.ctime()
        etime   = time.time()
        print 'Elapsed time: '+str(etime-stime)+' secs'
    

def mc4mp(invpr, outdir, dispdtype, wdisp, rffactor, monoc, verbose, step4uwalk, numbrun):
    print '--- Joint MC inversion for station: '+invpr.staid
    invpr.mc_joint_inv_iso(outdir=outdir, wdisp=wdisp, rffactor=rffactor,\
                   monoc=monoc, pfx=invpr.staid, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
    return 
    