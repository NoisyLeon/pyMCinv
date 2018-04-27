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
                
        
    
    def mc_inv_iso(self, instafname=None, ref=True, phase=True, group=False):
        if instafname is None:
            stalst  = self.waveforms.list()
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
            if staid == 'AK.WRH':
                return vpr
                
            # vpr.readdisp(infname='./old_code/TEST/Q22A.com.txt')
            # vpr.readrf(infname='./old_code/TEST/in.rf')
            # vpr.readmod(infname='./old_code/TEST/Q22A.mod1')
            # # vpr.readpara(infname='../old_code/TEST/in.para')
            # vpr.getpara()
    
    
        
            
        
        
        