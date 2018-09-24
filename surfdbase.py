# -*- coding: utf-8 -*-
"""
A python module for inversion with only dispersion data based on hdf5 database

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
import h5py
import numpy as np
import matplotlib.pyplot as plt
import obspy
import warnings
import copy
import os, shutil
from functools import partial
import multiprocessing
from subprocess import call
from mpl_toolkits.basemap import Basemap, shiftgrid, cm, interp
import obspy
import vprofile, mcpost, vmodel
import time
import numpy.ma as ma
import field2d_earth
from pyproj import Geod
import colormaps, pycpt
import numba
import time

def _get_vs_2d(z0, z1, zArr, vs_3d):
    Nlat, Nlon, Nz  = vs_3d.shape
    vs_out          = np.zeros((Nlat, Nlon))
    for ilat in range(Nlat):
        for ilon in range(Nlon):
            ind     = np.where((zArr > z0[ilat, ilon])*(zArr < z1[ilat, ilon]))[0]
            vs_temp = vs_3d[ilat, ilon, ind].mean()
            vs_out[ilat, ilon]\
                    = vs_temp
    return vs_out
    
class invhdf5(h5py.File):
    """ An object to for MCMC inversion based on HDF5 database
    ===================================================================================================================
    version history:
           - first version
    
    --- NOTES: mask data ---
    self[grd_id].attrs['mask_ph']   - existence of phase dispersion data, bool
    self[grd_id].attrs['mask_gr']   - existence of group dispersion data, bool
    self[grd_id].attrs['mask']      - existence of inversion, bool
    self.attrs['mask_inv']          - mask array for inversion, bool array
                                        this array is typically the mask_LD in the original ray tomography database
    self.attrs['mask_interp']       - mask array for interpolated finalized results, bool array
                                        this array is typically the "mask_inv" in the original ray tomography database
    ===================================================================================================================
    """
    def print_info(self):
        """
        print information of the database
        """
        outstr  = '================================================= Marcov Chain Monte Carlo Inversion Database ===============================================\n'
        outstr  += self.__str__()+'\n'
        outstr  += '--------------------------------------------------------------------------------------------------------------------------------------------\n'
        try:
            minlon          = self.attrs['minlon']
            maxlon          = self.attrs['maxlon']
            minlat          = self.attrs['minlat']
            maxlat          = self.attrs['maxlat']
            dlon            = self.attrs['dlon']
            dlat            = self.attrs['dlat']
            is_interp       = self.attrs['is_interp']
            if is_interp:
                dlon_interp = self.attrs['dlon_interp']
                dlat_interp = self.attrs['dlat_interp']
        except:
            print 'Empty Database!'
            return
        outstr      += 'minlon/maxlon           - '+str(minlon)+'/'+str(maxlon)+'\n'
        outstr      += 'minlat/maxlat           - '+str(minlat)+'/'+str(maxlat)+'\n'
        outstr      += 'dlon/dlat               - '+str(dlon)+'/'+str(dlat)+'\n'
        if is_interp:
            outstr  += 'dlon_interp/dlat_interp - '+str(dlon_interp)+'/'+str(dlat_interp)+'\n'
        outstr  += '--------------------------------------------------------------------------------------------------------------------------------------------\n'
        subgrps     = self.keys()
        if 'mask' in subgrps:
            outstr  += '--- mask array detected    \n'
        if 'topo' in subgrps:
            outstr  += '--- topo array(topography data for dlon/dlat) detected    \n'
        if 'topo_interp' in subgrps:
            outstr  += '--- topo_interp array(topography data for dlon_interp/dlat_interp) detected    \n'
        # average model
        if 'avg_paraval' in subgrps:
            outstr  += '=== average model \n'
            subgrp  = self['avg_paraval']
            if '0_org' in subgrp.keys():
                outstr\
                    += '--- original model parameters detected \n'
            if 'vs_org' in subgrp.keys():
                outstr\
                    += '--- original 3D model detected \n'
            if '0_smooth' in subgrp.keys():
                outstr\
                    += '--- smooth model parameters detected \n'
            if '3d_smooth' in subgrp.keys():
                outstr\
                    += '--- smooth 3D model detected \n'
        # minimum misfit model
        if 'min_paraval' in subgrps:
            outstr  += '=== minimum misfit model \n'
            subgrp  = self['min_paraval']
            if '0_org' in subgrp.keys():
                outstr\
                    += '--- original model parameters detected \n'
            if 'vs_org' in subgrp.keys():
                outstr\
                    += '--- original 3D model detected \n'
            if '0_smooth' in subgrp.keys():
                outstr\
                    += '--- smooth model parameters detected \n'
            if '3d_smooth' in subgrp.keys():
                outstr\
                    += '--- smooth 3D model detected \n'
        outstr += '============================================================================================================================================\n'
        print outstr
        return
    
    def _get_lon_lat_arr(self, is_interp=False):
        """Get longitude/latitude array
        """
        minlon          = self.attrs['minlon']
        maxlon          = self.attrs['maxlon']
        minlat          = self.attrs['minlat']
        maxlat          = self.attrs['maxlat']
        if is_interp:
            dlon        = self.attrs['dlon_interp']
            dlat        = self.attrs['dlat_interp']
        else:
            dlon        = self.attrs['dlon']
            dlat        = self.attrs['dlat']
        self.lons       = np.arange(int((maxlon-minlon)/dlon)+1)*dlon+minlon
        self.lats       = np.arange(int((maxlat-minlat)/dlat)+1)*dlat+minlat
        self.Nlon       = self.lons.size
        self.Nlat       = self.lats.size
        self.lonArr, self.latArr \
                        = np.meshgrid(self.lons, self.lats)
        return
    
    def _set_interp(self, dlon, dlat):
        self.attrs.create(name = 'dlon_interp', data=dlon)
        self.attrs.create(name = 'dlat_interp', data=dlat)
        return
    
    #==================================================================
    # functions before MC inversion runs
    #==================================================================
    def read_raytomo_dbase(self, inh5fname, runid, dtype='ph', wtype='ray', create_header=True, \
                           Tmin=-999, Tmax=999, verbose=False, res='LD', semfactor=2.):
        """
        read ray tomography database
        =================================================================================
        ::: input :::
        inh5fname   - input hdf5 file name
        runid       - id of run for the ray tomography
        dtype       - data type (ph or gr)
        wtype       - wave type (ray or lov)
        Tmin, Tmax  - minimum and maximum period to extract from the tomographic results
        res         - resolution for grid points, default is LD, low-definition
        semfactor   - factor to multiply for standard error of the mean (sem)
                        suggested by Fan-chi Lin
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
        if org_grp.attrs['datatype'] != dtype:
            raise ValueError('incompatible data type, input = '+dtype+', database = '+org_grp.attrs['datatype'])
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
            mask        = indset[dataid+'/mask_inv']
        else:
            mask        = indset[dataid+'/mask'+sfx]
        if create_header:
            self.attrs.create(name = 'minlon', data=minlon, dtype='f')
            self.attrs.create(name = 'maxlon', data=maxlon, dtype='f')
            self.attrs.create(name = 'minlat', data=minlat, dtype='f')
            self.attrs.create(name = 'maxlat', data=maxlat, dtype='f')
            self.attrs.create(name = 'dlon', data=dlon_interp)
            self.attrs.create(name = 'dlat', data=dlat_interp)
        self._get_lon_lat_arr()
        # create mask_inv array indicating inversions
        self.attrs.create(name='mask_inv', data = mask)
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
                data[2, :]          = disp_un[:] * semfactor
                group.create_dataset(name='disp_'+dtype+'_'+wtype, data=data)
                group.attrs.create(name='mask_'+dtype, data = mask[ilat, ilon])
        indset.close()
        return
    
    def read_raytomo_dbase_group(self, inh5fname, runid, wtype='ray',\
            create_grp = False, Tmin=-999, Tmax=999, verbose=False, res='LD', semfactor=2.):
        """
        read group dispersion data from ray tomography database
        =================================================================================
        ::: input :::
        inh5fname   - input hdf5 file name
        runid       - id of run for the ray tomography
        wtype       - wave type (ray or lov)
        Tmin, Tmax  - minimum and maximum period to extract from the tomographic results
        res         - resolution for grid points, default is LD, low-definition
        semfactor   - factor to multiply for standard error of the mean (sem)
        =================================================================================
        """
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
        if isotropic:
            print 'isotropic inversion results do not output gaussian std!'
            return
        if res == 'LD':
            sfx         = '_LD'
        elif res == 'HD':
            sfx         = '_HD'
        else:
            sfx         = ''
        if sfx == '':
            mask_new    = indset[dataid+'/mask_inv']
        else:
            mask_new    = indset[dataid+'/mask'+sfx]
        mask_org        = self.attrs['mask_inv']
        mask            = mask_org + mask_new
        self.attrs.create(name='mask_inv', data = mask)
        self._get_lon_lat_arr()
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                data_str    = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                try:
                    group   = self[data_str]
                except:
                    if create_grp:
                        group  \
                            = self.create_group( name = data_str )
                    else:
                        continue
                disp_v      = np.array([])
                T           = np.array([])
                for per in pers:
                    if per < Tmin or per > Tmax:
                        continue
                    try:
                        pergrp      = grp['%g_sec'%( per )]
                        vel         = pergrp['vel_iso'+sfx].value
                    except KeyError:
                        if verbose:
                            print 'No data for T = '+str(per)+' sec'
                        continue
                    T               = np.append(T, per)
                    disp_v          = np.append(disp_v, vel[ilat, ilon])
                # get sem from phase for group
                if not np.allclose(T, group['disp_ph_ray'].value[0, :]):
                    raise ValueError('Incompatible period for phase and group dispersion data!')
                disp_un             = group['disp_ph_ray'].value[2, :]    
                data                = np.zeros((3, T.size))
                data[0, :]          = T[:]
                data[1, :]          = disp_v[:]
                data[2, :]          = disp_un[:] * semfactor
                group.create_dataset(name='disp_gr_'+wtype, data=data)
                group.attrs.create(name='mask_gr', data = mask_new[ilat, ilon])
        indset.close()
        return
    
    def read_crust_thickness(self, infname='crsthk.xyz', source='crust_1.0'):
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
    
    def read_sediment_thickness(self, infname='sedthk.xyz', source='crust_1.0'):
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
            try:
                grd_lon = float(split_id[0])
            except ValueError:
                continue
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
        ============================================================================
        ::: input :::
        infname     - input file name
        download    - download the data or not, if the etopo file does not exist
        delete      - delete the downloaded etopo file or not
        source      - source name (default - etopo2)
        ============================================================================
        """
        from netCDF4 import Dataset
        try:
            etopodbase      = Dataset(infname)
        except IOError:
            if download:
                url         = 'https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2g/netCDF/ETOPO2v2g_f4_netCDF.zip'
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
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
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
    
    def get_raytomo_mask(self, inh5fname, runid):
        """
        get the mask array from ray tomography database
        """
        # mask array
        dataid      = 'reshaped_qc_run_'+str(runid)
        indset      = h5py.File(inh5fname)
        grp         = indset[dataid]
        mask        = grp['mask_inv'].value
        # dlon/dlat
        dataid      = 'qc_run_'+str(runid)
        grp         = indset[dataid]
        dlon        = grp.attrs['dlon']
        dlat        = grp.attrs['dlat']
        self._set_interp(dlon=dlon, dlat=dlat)
        self._get_lon_lat_arr(is_interp=True)
        if mask.shape == self.lonArr.shape:
            try:
                mask_org\
                    = self.attrs['mask_interp']
                mask+= mask_org
            except KeyError:
                self.attrs.create(name = 'mask_interp', data = mask)
        else:
            raise ValueError('Incompatible dlon/dlat with input mask array from ray tomography database')
        self.attrs.create(name = 'is_interp', data=True, dtype=bool)
        return
    
    #==================================================================
    # function inspection of the input data
    #==================================================================
    def plot_disp(self, lon, lat, wtype='ray', derivegr=True, ploterror=False, showfig=True):
        """
        plot dispersion data given location of the grid point
        ==========================================================================================
        ::: input :::
        lon/lat     - location of the grid point
        wtype       - type of waves (ray or lov)
        derivegr    - compute and plot the group velocities derived from phase velocities or not
        ploterror   - plot uncertainties or not
        showfig     - show the figure or not
        ==========================================================================================
        """
        if lon < 0.:
            lon     += 360.
        data_str    = str(lon)+'_'+str(lat)
        try:
            grp     = self[data_str]
        except:
            print 'No data at longitude =',lon,' lattitude =',lat
            return
        plt.figure()
        ax  = plt.subplot()
        try:
            disp_ph = grp['disp_ph_'+wtype]
            if ploterror:
                plt.errorbar(disp_ph[0, :], disp_ph[1, :], yerr=disp_ph[2, :], color='b', lw=3, label='phase')
            else:
                plt.plot(disp_ph[0, :], disp_ph[1, :], 'bo-', lw=3, ms=10, label='phase')
        except:
            pass
        # compute and plot the derived group velocities
        if derivegr:
            import scipy.interpolate
            CubicSpl= scipy.interpolate.CubicSpline(disp_ph[0, :], disp_ph[1, :])
            Tmin    = disp_ph[0, 0]
            Tmax    = disp_ph[0, -1]
            Tinterp = np.mgrid[Tmin:Tmax:0.1]
            Cinterp = CubicSpl(Tinterp)
            diffC   = Cinterp[2:] - Cinterp[:-2]
            dCdTinterp    \
                    = diffC/0.2
            # dCdT    = np.zeros(disp_ph[0, :].size)
            # for i in range(dCdT.size):
            #     if i == 0:
            #         dCdT[i] = dCdTinterp[0]
            #         continue
            #     if i == dCdT.size-1:
            #         dCdT[i] = dCdTinterp[-1]
            #         continue
            #     ind = np.where(abs(Tinterp[1:-1] - disp_ph[0, i])<0.01)[0]
            #     # print Tinterp[1:-1], disp_ph[0, i]
            #     dCdT[i]\
            #         = dCdTinterp[ind]
            # sU      = 1./disp_ph[1, :] + (disp_ph[0, :]/(disp_ph[1, :])**2)*dCdT
            # derived_U\
            #         = 1./sU
            # plt.plot(disp_ph[0, :], derived_U, 'k--', lw=1, ms=10, label='derived group')
            
            sU      = 1./Cinterp[1:-1] + (Tinterp[1:-1]/(Cinterp[1:-1])**2)*dCdTinterp
            derived_U\
                    = 1./sU
            plt.plot(Tinterp[1:-1], derived_U, 'k--', lw=3, label='derived group')
        try:
            disp_gr = grp['disp_gr_'+wtype]
            if ploterror:
                plt.errorbar(disp_gr[0, :], disp_gr[1, :], yerr=disp_gr[2, :], color='r', lw=3, label='group')
            else:
                plt.plot(disp_gr[0, :], disp_gr[1, :], 'ro-', lw=3, ms=10, label='group')
        except:
            pass
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        plt.xlabel('Period (sec)', fontsize=30)
        plt.ylabel('Velocity (km/sec)', fontsize=30)
        if lon > 180.:
            lon     -= 360.
        plt.title('longitude = '+str(lon)+' latitude = '+str(lat), fontsize=30)
        plt.legend(loc=0, fontsize=20)
        if showfig:
            plt.show()
        return
    #==================================================================
    # function for MC inversion runs
    #==================================================================
    
    def mc_inv_iso(self, ingrdfname=None, phase=True, group=False, outdir='./workingdir', vp_water=1.5, monoc=True,
            verbose=False, step4uwalk=1500, numbrun=15000, subsize=1000, nprocess=None, parallel=True, skipmask=True):
        """
        Bayesian Monte Carlo inversion of surface wave data for an isotropic model
        ==================================================================================================================
        ::: input :::
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        phase/group - include phase/group velocity dispersion data or not
        outdir      - output directory
        vp_water    - P wave velocity in water layer (default - 1.5 km/s)
        monoc       - require monotonical increase in the crust or not
        step4uwalk  - step interval for uniform random walk in the parameter space
        numbrun     - total number of runs
        subsize     - size of subsets, used if the number of elements in the parallel list is too large to avoid deadlock
        nprocess    - number of process
        parallel    - run the inversion in parallel or not
        skipmask    - skip masked grid points or not
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
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat = float(split_id[1])
            igrd    += 1
            if (phase and self[grd_id].attrs['mask_ph'] ) and skipmask:
                print '--- Skip MC inversion for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                continue
            if (group and self[grd_id].attrs['mask_gr'] ) and skipmask:
                print '--- Skip MC inversion for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                continue
            #-----------------------------
            # get data
            #-----------------------------
            vpr                 = vprofile.vprofile1d()
            if phase:
                try:
                    indisp      = self[grd_id+'/disp_ph_ray'].value
                    vpr.get_disp(indata=indisp, dtype='ph', wtype='ray')
                except KeyError:
                    print 'WARNING: No phase dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)
            if group:
                try:
                    indisp      = self[grd_id+'/disp_gr_ray'].value
                    vpr.get_disp(indata=indisp, dtype='gr', wtype='ray')
                except KeyError:
                    print 'WARNING: No group dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)
            if vpr.data.dispR.npper == 0 and vpr.data.dispR.ngper == 0:
                print 'WARNING: No dispersion data for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)
                continue
            #-----------------------------
            # initial model parameters
            #-----------------------------
            vsdata              = self[grd_id+'/reference_vs'].value
            crtthk              = self[grd_id].attrs['crust_thk']
            sedthk              = self[grd_id].attrs['sedi_thk']
            topovalue           = self[grd_id].attrs['topo']
            vpr.model.isomod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], crtthk=crtthk, sedthk=sedthk,\
                            topovalue=topovalue, maxdepth=200., vp_water=vp_water)
            vpr.getpara()
            # # # if np.random.rand() > 0.9 and topovalue<0.:
            # # #     print grd_id
            # # #     return vpr, vsdata
            # # # else:
            # # #     continue
            # # # if not (np.random.rand() > 0.9 and topovalue<0.):
            # # #     continue
            # if grd_lon != -155. or grd_lat != 68.:
            #     continue
            # return vpr
            print '--- MC inversion for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
            if parallel:
                vpr.mc_joint_inv_iso_mp(outdir=outdir, dispdtype=dispdtype, wdisp=1.,\
                   monoc=monoc, pfx=grd_id, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun, subsize=subsize, nprocess=nprocess)
            else:
                vpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=1., \
                   monoc=monoc, pfx=grd_id, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
        return
    
    #==================================================================
    # function to read MC inversion results
    #==================================================================
    def read_inv(self, datadir, ingrdfname=None, factor=1., thresh=0.5, stdfactor=2, skipmask=True, avgqc=True, Nmin=500):
        """
        read the inversion results in to data base
        ==================================================================================================================
        ::: input :::
        datadir     - data directory
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        factor      - factor to determine the threshhold value for selectingthe finalized model
        thresh      - threshhold value for selecting the finalized model
                        misfit < min_misfit*factor + thresh
        skipmask    - skip masked grid points or not
        avgqc       - turn on quality control for average model or not
        ::: NOTE :::
        mask_inv array will be updated according to the existence of inversion results
        ==================================================================================================================
        """
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
        igrd        = 0
        Ngrd        = len(grdlst)
        temp_mask   = self.attrs['mask_inv']
        self._get_lon_lat_arr(is_interp=False)
        for grd_id in grdlst:
            split_id= grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            if grd_lon > 180.:
                grd_lon     -= 360.
            grd_lat = float(split_id[1])
            igrd    += 1
            grp     = self[grd_id]
            ilat    = np.where(grd_lat == self.lats)[0]
            ilon    = np.where(grd_lon == self.lons)[0]
            if (grp.attrs['mask_ph'] and grp.attrs['mask_gr']) and skipmask:
                print '--- Skipping inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                grp.attrs.create(name='mask', data = True)
                temp_mask[ilat, ilon]\
                    = True
                continue
            invfname    = datadir+'/mc_inv.'+ grd_id+'.npz'
            datafname   = datadir+'/mc_data.'+grd_id+'.npz'
            if not (os.path.isfile(invfname) and os.path.isfile(datafname)):
                print '--- No inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                grp.attrs.create(name='mask', data = True)
                temp_mask[ilat, ilon]\
                        = True
                continue
                # # # raise ValueError('No inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
            print '--- Reading inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
            temp_mask[ilat, ilon]\
                        = False
            grp.attrs.create(name='mask', data = False)
            topovalue   = grp.attrs['topo']
            vpr         = mcpost.postvpr(waterdepth=-topovalue, factor=factor, thresh=thresh, stdfactor=stdfactor)
            vpr.read_data(infname = datafname)
            vpr.read_inv_data(infname = invfname, verbose=False)
            # change stdfactor is necessary to make sure enough models are finally accepted
            # future plan: adaptively change thresh and stdfactor to make accpeted model around a specified value
            # --- added Sep 7th, 2018
            N0          = vpr.ind_thresh.size
            if N0 < Nmin:
                vpr_temp    = copy.deepcopy(vpr)
                vpr_temp.stdfactor\
                            = None
                vpr_temp.get_thresh_model()
                N1          = vpr_temp.ind_thresh.size
                temp_factor = stdfactor
                if N1 > Nmin:
                    # loop to modify stdfactor
                    while(N0 < 0.9*N1 and temp_factor<10.):
                        temp_factor     += 0.5
                        vpr.stdfactor   = temp_factor
                        vpr.get_thresh_model()
                        N0              = vpr.ind_thresh.size
                    if temp_factor == 10.:
                        vpr.stdfactor   = None
                        vpr.ind_thresh  = vpr_temp.ind_thresh
                else:
                    print '--- NOTE: number of final accpected model =',N1
                    vpr.stdfactor       = None
                    vpr.ind_thresh      = vpr_temp.ind_thresh
            # --- added Sep 7th, 2018
            vpr.get_paraval()
            vpr.run_avg_fwrd(wdisp=1.)
            if avgqc:
                if vpr.avg_misfit > (vpr.min_misfit*vpr.factor + vpr.thresh)*3.:
                    print '--- Unstable inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                    continue
            #------------------------------------------
            # store inversion results in the database
            #------------------------------------------
            grp.create_dataset(name = 'avg_paraval', data = vpr.avg_paraval)
            grp.create_dataset(name = 'min_paraval', data = vpr.min_paraval)
            grp.create_dataset(name = 'sem_paraval', data = vpr.sem_paraval)
            grp.attrs.create(name = 'avg_misfit', data = vpr.vprfwrd.data.misfit)
            grp.attrs.create(name = 'min_misfit', data = vpr.min_misfit)
        # set the is_interp as False (default)
        self.attrs.create(name = 'is_interp', data=False, dtype=bool)
        self.attrs.create(name='mask_inv', data = temp_mask)
        return
    
    def get_vpr(self, datadir, lon, lat, factor=1., thresh=0.5):
        if lon < 0.:
            lon     += 360.
        grd_id      = str(lon)+'_'+str(lat)
        try:
            grp     = self[grd_id]
        except:
            print 'No data at longitude =',lon,' lattitude =',lat
            return
        invfname    = datadir+'/mc_inv.'+ grd_id+'.npz'
        datafname   = datadir+'/mc_data.'+grd_id+'.npz'
        topovalue   = grp.attrs['topo']
        vpr         = mcpost.postvpr(waterdepth=-topovalue, factor=factor, thresh=thresh)
        vpr.read_inv_data(infname = invfname, verbose=True)
        vpr.read_data(infname = datafname)
        vpr.get_paraval()
        vpr.run_avg_fwrd(wdisp=1.)
        if vpr.avg_misfit > (vpr.min_misfit*vpr.factor + vpr.thresh)*2.:
            print '--- Unstable inversion results for grid: lon = '+str(lon)+', lat = '+str(lat)
        if lon > 0.:
            lon     -= 360.
        vpr.code    = str(lon)+'_'+str(lat)
        return vpr
        
    #==================================================================
    # postprocessing, functions maniplulating paraval arrays
    #==================================================================
    
    def get_paraval(self, pindex, dtype='min', ingrdfname=None, isthk=False):
        """
        get the data for the model parameter
        ==================================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        ==================================================================================================================
        """
        self._get_lon_lat_arr(is_interp=False)
        data        = np.zeros(self.latArr.shape)
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
        igrd            = 0
        Ngrd            = len(grdlst)
        for grd_id in grdlst:
            split_id    = grd_id.split('_')
            try:
                grd_lon     = float(split_id[0])
            except ValueError:
                continue
            grd_lat     = float(split_id[1])
            igrd        += 1
            grp         = self[grd_id]
            try:
                ind_lon = np.where(grd_lon==self.lons)[0][0]
                ind_lat = np.where(grd_lat==self.lats)[0][0]
            except IndexError:
                # print 'WARNING: grid data N/A at: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                continue
            try:
                paraval             = grp[dtype+'_paraval'].value
            except KeyError:
                # print 'WARNING: no data at grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                continue
            data[ind_lat, ind_lon]  = paraval[pindex]
            if isthk:
                topovalue               = grp.attrs['topo']
                data[ind_lat, ind_lon]  = data[ind_lat, ind_lon] - topovalue
        return data
    
    def get_filled_paraval(self, pindex, dtype='min', ingrdfname=None, isthk=False, do_interp=False, workingdir='working_interpolation'):
        """
        get the filled data array for the model parameter
        ==================================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        do_interp   - perform interpolation or not
        workingdir  - working directory for interpolation
        ==================================================================================================================
        """
        minlon      = self.attrs['minlon']
        maxlon      = self.attrs['maxlon']
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']
        data        = self.get_paraval(pindex=pindex, dtype=dtype, ingrdfname=ingrdfname, isthk=isthk)
        mask_inv    = self.attrs['mask_inv']
        ind_valid   = np.logical_not(mask_inv)
        data_out    = data.copy()
        g           = Geod(ellps='WGS84')
        vlonArr     = self.lonArr[ind_valid]
        vlatArr     = self.latArr[ind_valid]
        vdata       = data[ind_valid]
        L           = vlonArr.size
        #------------------------------
        # filling the data_out array
        #------------------------------
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                if not mask_inv[ilat, ilon]:
                    continue
                clonArr         = np.ones(L, dtype=float)*self.lons[ilon]
                clatArr         = np.ones(L, dtype=float)*self.lats[ilat]
                az, baz, dist   = g.inv(clonArr, clatArr, vlonArr, vlatArr)
                ind_min         = dist.argmin()
                data_out[ilat, ilon] \
                                = vdata[ind_min]
        if do_interp:
            #----------------------------------------------------
            # interpolation for data to dlon_interp/dlat_interp
            #----------------------------------------------------
            dlon                = self.attrs['dlon_interp']
            dlat                = self.attrs['dlat_interp']
            field2d             = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                    minlat=minlat, maxlat=maxlat, dlat=dlat, period=10., evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            field2d.read_array(lonArr = vlonArr, latArr = vlatArr, ZarrIn = vdata)
            outfname            = 'interp_data.lst'
            field2d.interp_surface(workingdir=workingdir, outfname=outfname)
            data_out            = field2d.Zarr
        return data_out
    
    def get_smooth_paraval(self, pindex, sigma=1., smooth_type = 'gauss', dtype='min',\
            workingdir = 'working_gauss_smooth', width=50., ingrdfname=None, isthk=False, do_interp=False):
        """
        get smooth data array for the model parameter
        ==================================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
        sigma       - total number of smooth iterations
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        width       - width for Gaussian smoothing (unit - km)
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        ==================================================================================================================
        """
        data            = self.get_filled_paraval(pindex=pindex, dtype=dtype, ingrdfname=ingrdfname, isthk=isthk, do_interp=do_interp)
        if smooth_type is 'nearneighbor':
            data_smooth = data.copy()
            #- Smoothing by averaging over neighbouring cells. ----------------------
            for iteration in range(int(sigma)):
                for ilat in range(1, self.Nlat-1):
                    for ilon in range(1, self.Nlon-1):
                        data_smooth[ilat, ilon] = (data[ilat, ilon] + data[ilat+1, ilon] \
                                                   + data[ilat-1, ilon] + data[ilat, ilon+1] + data[ilat, ilon-1])/5.0
        elif smooth_type is 'gauss':
            minlon          = self.attrs['minlon']
            maxlon          = self.attrs['maxlon']
            minlat          = self.attrs['minlat']
            maxlat          = self.attrs['maxlat']
            if do_interp:
                dlon        = self.attrs['dlon_interp']
                dlat        = self.attrs['dlat_interp']
                self._get_lon_lat_arr(is_interp=True)
                # change mask array if interpolation is performed
                mask        = self.attrs['mask_interp']
            else:
                dlon        = self.attrs['dlon']
                dlat        = self.attrs['dlat']
                mask        = self.attrs['mask_inv']
            field           = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=dlon,
                                    minlat=minlat, maxlat=maxlat, dlat=dlat, period=10., evlo=(minlon+maxlon)/2., evla=(minlat+maxlat)/2.)
            index           = np.logical_not(mask)
            field.read_array(lonArr = self.lonArr[index], latArr = self.latArr[index], ZarrIn = data[index])
            outfname        = 'smooth_paraval.lst'
            field.gauss_smoothing(workingdir=workingdir, outfname=outfname, width=width)
            data_smooth     = field.Zarr
        return data, data_smooth
    
    def paraval_arrays(self, dtype='min', sigma=1, width = 50., verbose=False):
        """
        get the paraval arrays and store them in the database
        =============================================================================
        ::: input :::
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        sigma       - total number of smooth iterations
        width       - width for Gaussian smoothing (unit - km)
        dlon/dlat   - longitude/latitude interval for interpolation
        -----------------------------------------------------------------------------
        ::: procedures :::
        1.  get_paraval
                    - get the paraval for each grid point in the inversion
        2.  get_filled_paraval
                    - a. fill the grid points that are NOT included in the inversion
                      b. perform interpolation if needed
        3.  get_smooth_paraval
                    - perform spatial smoothing of the paraval in each grid point
        =============================================================================
        """
        grp                 = self.require_group( name = dtype+'_paraval' )
        do_interp           = self.attrs['is_interp']
        for pindex in range(13):
            if pindex == 11 or pindex == 12:
                data, data_smooth   = self.get_smooth_paraval(pindex=pindex, dtype=dtype, \
                        sigma=sigma, width = width, isthk=True, do_interp=do_interp)
            else:
                data, data_smooth   = self.get_smooth_paraval(pindex=pindex, dtype=dtype, \
                        sigma=sigma, width = width, isthk=False, do_interp=do_interp)
            grp.create_dataset(name = str(pindex)+'_org', data = data)
            grp.create_dataset(name = str(pindex)+'_smooth', data = data_smooth)
        return
    
    #==================================================================
    # postprocessing, functions for 3D model
    #==================================================================
    
    def construct_3d(self, dtype='min', is_smooth=False, maxdepth=200., dz=0.1):
        """
        construct 3D vs array
        =================================================================
        ::: input :::
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        is_smooth   - use the smoothed array or not
        maxdepth    - maximum depth (default - 200 km)
        dz          - depth interval (default - 0.1 km)
        =================================================================
        """
        is_interp   = self.attrs['is_interp']
        grp         = self[dtype+'_paraval']
        self._get_lon_lat_arr(is_interp=is_interp)
        if self.latArr.shape != grp['0_org'].value.shape:
            raise ValueError('incompatible paraval data with lonArr/latArr !')
        Nz          = int(maxdepth/dz) + 1
        zArr        = np.arange(Nz)*dz
        vs3d        = np.zeros((self.latArr.shape[0], self.latArr.shape[1], Nz))
        Ntotal      = self.Nlat*self.Nlon
        N0          = int(Ntotal/100.)
        i           = 0
        j           = 0
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                i                   += 1
                if np.floor(i/N0) > j:
                    print 'Constructing 3d model:',j,' % finished'
                    j               += 1
                paraval             = np.zeros(13, dtype=np.float64)
                if is_interp:
                    topovalue       = self['topo_interp'].value[ilat, ilon]
                else:
                    grd_id          = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                    topovalue       = self[grd_id].attrs['topo']
                for pindex in range(13):
                    if is_smooth:
                        data        = grp[str(pindex)+'_smooth'].value
                    else:
                        data        = grp[str(pindex)+'_org'].value
                    paraval[pindex] = data[ilat, ilon]
                    if pindex == 11 or pindex == 12:
                        paraval[pindex] \
                                    = paraval[pindex] + topovalue
                vel_mod             = vmodel.model1d()
                if topovalue < 0.:
                    vel_mod.get_para_model(paraval = paraval, waterdepth=-topovalue, vpwater=1.5, nmod=4, \
                        numbp=np.array([1, 2, 4, 5]), mtype = np.array([5, 4, 2, 2]), vpvs = np.array([0, 2., 1.75, 1.75]), maxdepth=200.)
                else:
                    vel_mod.get_para_model(paraval = paraval)
                zArr_in, VsvArr_in  = vel_mod.get_grid_mod()
                if topovalue > 0.:
                    zArr_in         = zArr_in - topovalue
                # # interpolation
                vs_interp           = np.interp(zArr, xp = zArr_in, fp = VsvArr_in)
                vs3d[ilat, ilon, :] = vs_interp[:]                
        if is_smooth:
            grp.create_dataset(name = 'vs_smooth', data = vs3d)
            grp.create_dataset(name = 'z_smooth', data = zArr)
        else:
            grp.create_dataset(name = 'vs_org', data = vs3d)
            grp.create_dataset(name = 'z_org', data = zArr)
        return
        
    def get_topo_arr(self, infname='../ETOPO2v2g_f4.nc'):
        """
        get the topography array
        """
        is_interp   = self.attrs['is_interp']
        self._get_lon_lat_arr(is_interp=is_interp)
        topoarr     = np.zeros(self.lonArr.shape)
        if is_interp:
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
            for ilat in range(self.Nlat):
                for ilon in range(self.Nlon):
                    grd_lon             = self.lons[ilon]
                    grd_lat             = self.lats[ilat]
                    if grd_lon > 180.:
                        grd_lon         -= 360.
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
                    topoarr[ilat, ilon] = z
            self.create_dataset(name='topo_interp', data = topoarr)
        else:
            for ilat in range(self.Nlat):
                for ilon in range(self.Nlon):
                    grd_id              = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                    topovalue           = self[grd_id].attrs['topo']
                    topoarr[ilat, ilon] = topovalue
            self.create_dataset(name='topo', data = topoarr)
        return
    
    def convert_to_vts(self, outdir, dtype='min', is_smooth=False, pfx='', verbose=False, unit=True):
        """ Convert Vs model to vts format for plotting with Paraview, VisIt
        ========================================================================================
        ::: input :::
        outdir      - output directory
        modelname   - modelname ('dvsv', 'dvsh', 'dvp', 'drho')
        pfx         - prefix of output files
        unit        - output unit sphere(radius=1) or not
        ========================================================================================
        """
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
            data_str= dtype + '_smooth'
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
            data_str= dtype + '_org'
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        from tvtk.api import tvtk, write_data
        if unit:
            Rref=6471.
        else:
            Rref=1.
        is_interp   = self.attrs['is_interp']
        self._get_lon_lat_arr(is_interp=is_interp)
        # convert geographycal coordinate to spherichal coordinate
        theta       = (90.0 - self.lats)*np.pi/180.
        phi         = self.lons*np.pi/180.
        radius      = Rref - zArr
        theta, phi, radius \
                    = np.meshgrid(theta, phi, radius, indexing='ij')
        # convert spherichal coordinate to 3D Cartesian coordinate
        x           = radius * np.sin(theta) * np.cos(phi)/Rref
        y           = radius * np.sin(theta) * np.sin(phi)/Rref
        z           = radius * np.cos(theta)/Rref
        dims        = vs3d.shape
        pts         = np.empty(z.shape + (3,), dtype=float)
        pts[..., 0] = x
        pts[..., 1] = y
        pts[..., 2] = z
        pts         = pts.transpose(2, 1, 0, 3).copy()
        pts.shape   = pts.size / 3, 3
        sgrid       = tvtk.StructuredGrid(dimensions=dims, points=pts)
        sgrid.point_data.scalars \
                    = (vs3d).ravel(order='F')
        sgrid.point_data.scalars.name \
                    = 'Vs'
        outfname    = outdir+'/'+pfx+'Vs_'+data_str+'.vts'
        write_data(sgrid, outfname)
        return

    #==================================================================
    # plotting functions 
    #==================================================================
    
    def _get_basemap(self, projection='lambert', geopolygons=None, resolution='i'):
        """Get basemap for plotting results
        """
        # fig=plt.figure(num=None, figsize=(12, 12), dpi=80, facecolor='w', edgecolor='k')
        plt.figure()
        minlon      = self.attrs['minlon']
        maxlon      = self.attrs['maxlon']
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']
        lat_centre  = (maxlat+minlat)/2.0
        lon_centre  = (maxlon+minlon)/2.0
        if projection=='merc':
            m       = Basemap(projection='merc', llcrnrlat=minlat-5., urcrnrlat=maxlat+5., llcrnrlon=minlon-5.,
                      urcrnrlon=maxlon+5., lat_ts=20, resolution=resolution, epsg = 4269)
            # m.drawparallels(np.arange(minlat,maxlat,dlat), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(minlon,maxlon,dlon), labels=[1,0,0,1])
            m.drawparallels(np.arange(-80.0,80.0,5.0), labels=[1,0,0,1])
            m.drawmeridians(np.arange(-170.0,170.0,5.0), labels=[1,0,0,1])
            m.drawstates(color='g', linewidth=2.)
        elif projection=='global':
            m       = Basemap(projection='ortho',lon_0=lon_centre, lat_0=lat_centre, resolution=resolution)
            # m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,1])
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), labels=[1,0,0,1])
        elif projection=='regional_ortho':
            m1      = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution='l')
            m       = Basemap(projection='ortho', lon_0=minlon, lat_0=minlat, resolution=resolution,\
                        llcrnrx=0., llcrnry=0., urcrnrx=m1.urcrnrx/mapfactor, urcrnry=m1.urcrnry/3.5)
            m.drawparallels(np.arange(-80.0,80.0,10.0), labels=[1,0,0,0],  linewidth=2,  fontsize=20)
            # m.drawparallels(np.arange(-90.0,90.0,30.0),labels=[1,0,0,0], dashes=[10, 5], linewidth=2,  fontsize=20)
            # m.drawmeridians(np.arange(10,180.0,30.0), dashes=[10, 5], linewidth=2)
            m.drawmeridians(np.arange(-170.0,170.0,10.0),  linewidth=2)
        elif projection=='lambert':
            distEW, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, minlat, maxlon) # distance is in m
            distNS, az, baz = obspy.geodetics.gps2dist_azimuth(minlat, minlon, maxlat+2., minlon) # distance is in m
            m       = Basemap(width=distEW, height=distNS, rsphere=(6378137.00,6356752.3142), resolution='h', projection='lcc',\
                        lat_1=minlat, lat_2=maxlat, lon_0=lon_centre, lat_0=lat_centre+1)
            m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=1, dashes=[2,2], labels=[1,1,0,0], fontsize=20)
            m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=1, dashes=[2,2], labels=[0,0,1,0], fontsize=20)
            # m.drawparallels(np.arange(-80.0,80.0,10.0), linewidth=0.5, dashes=[2,2], labels=[1,0,0,0], fontsize=5)
            # m.drawmeridians(np.arange(-170.0,170.0,10.0), linewidth=0.5, dashes=[2,2], labels=[0,0,0,1], fontsize=5)
        m.drawcoastlines(linewidth=1.0)
        m.drawcountries(linewidth=1.)
        # # m.drawmapboundary(fill_color=[1.0,1.0,1.0])
        # m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        # # m.drawlsmask(land_color='0.8', ocean_color='#99ffff')
        # m.drawmapboundary(fill_color="white")
        # m.shadedrelief(scale=1., origin='lower')
        try:
            geopolygons.PlotPolygon(inbasemap=m)
        except:
            pass
        return m
         
    def plot_paraval(self, pindex, is_smooth=True, dtype='avg', sigma=1, width = 50., \
            ingrdfname=None, isthk=False, shpfx=None, outfname=None, clabel='', cmap='cv', \
                projection='lambert', hillshade=False, geopolygons=None, vmin=None, vmax=None, showfig=True):
        """
        plot the one given parameter in the paraval array
        ===================================================================================================
        ::: input :::
        pindex      - parameter index in the paraval array
        org_mask    - use the original mask in the database or not
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        ===================================================================================================
        """
        is_interp       = self.attrs['is_interp']
        data, data_smooth\
                        = self.get_smooth_paraval(pindex=pindex, dtype=dtype,\
                            sigma=sigma, width = width, isthk=isthk, do_interp=is_interp)
        if is_interp:
            mask        = self.attrs['mask_interp']
        else:
            mask        = self.attrs['mask_inv']
        if is_smooth:
            mdata       = ma.masked_array(data_smooth, mask=mask )
        else:
            mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m               = self._get_basemap(projection=projection)
        x, y            = m(self.lonArr, self.latArr)
        shapefname      = '/home/leon/geological_maps/qfaults'
        m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        shapefname      = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./cv.cpt')
        elif cmap == 'gmtseis':
            import pycpt
            cmap        = pycpt.load.gmtColormap('./GMTseis.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap= pycpt.load.gmtColormap(cmap)
            except:
                pass
        ################################3
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
        
            etopodata   = Dataset('/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1       = pycpt.load.gmtColormap('/home/leon/station_map/etopo1.cpt')
            mycm2       = pycpt.load.gmtColormap('/home/leon/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        ###################################################################
        # if hillshade:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        # else:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        if hillshade:
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=.5)
        else:
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(clabel, fontsize=20, rotation=0)
        cb.ax.tick_params(labelsize=15)
        cb.set_alpha(1)
        cb.draw_all()
        # # cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        if outfname is not None:
            ind_valid   = np.logical_not(mask)
            outlon      = self.lonArr[ind_valid]
            outlat      = self.latArr[ind_valid]
            outZ        = data[ind_valid]
            OutArr      = np.append(outlon, outlat)
            OutArr      = np.append(OutArr, outZ)
            OutArr      = OutArr.reshape(3, outZ.size)
            OutArr      = OutArr.T
            np.savetxt(outfname, OutArr, '%g')
        return 
    
    def plot_horizontal(self, depth, depthb=None, depthavg=None, dtype='avg', is_smooth=True, shpfx=None, clabel='', title='',\
            cmap='cv', projection='lambert', hillshade=False, geopolygons=None, vmin=None, vmax=None, \
            lonplt=[], latplt=[], incat=None, plotevents=False, showfig=True):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        depth       - depth of the slice for plotting
        depthb      - depth of bottom grid for plotting (default: None)
        depthavg    - depth range for average, vs will be averaged for depth +/- depthavg
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        is_smooth   - use the data that has been smoothed or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        =================================================================================================================
        """
        is_interp   = self.attrs['is_interp']
        self._get_lon_lat_arr(is_interp=is_interp)
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        if depthb is not None:
            if depthb < depth:
                raise ValueError('depthb should be larger than depth!')
            index   = np.where((zArr >= depth)*(zArr <= depthb) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        elif depthavg is not None:
            depth0  = max(0., depth-depthavg)
            depth1  = depth+depthavg
            index   = np.where((zArr >= depth0)*(zArr <= depth1) )[0]
            vs_plt  = (vs3d[:, :, index]).mean(axis=2)
        else:
            try:
                index   = np.where(zArr >= depth )[0][0]
            except IndexError:
                print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
                return
            depth       = zArr[index]
            vs_plt      = vs3d[:, :, index]
        if is_interp:
            mask    = self.attrs['mask_interp']
        else:
            mask    = self.attrs['mask_inv']
        mvs         = ma.masked_array(vs_plt, mask=mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr, self.latArr)
        shapefname  = '/home/leon/geological_maps/qfaults'
        m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        shapefname  = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        
        # shapefname  = '/home/leon/sediments_US/Sedimentary_Basins_of_the_United_States'
        # m.readshapefile(shapefname, 'sediments', linewidth=2, color='grey')
        # shapefname  = '/home/leon/AK_sediments/AK_Sedimentary_Basins'
        # m.readshapefile(shapefname, 'sediments', linewidth=2, color='grey')
        # shapefname  = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        ################################3
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
        
            etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1=pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
            mycm2=pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        ###################################################################
        # if hillshade:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        # else:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        
        im          = m.pcolormesh(x, y, mvs, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(clabel, fontsize=20, rotation=0)
        cb.ax.tick_params(labelsize=15)
        cb.set_alpha(1)
        cb.draw_all()
        #
        if len(lonplt) > 0 and len(lonplt) == len(latplt): 
            xc, yc      = m(lonplt, latplt)
            m.plot(xc, yc,'go', lw = 3)
        ############################################################
        if plotevents or incat is not None:
            evlons  = np.array([])
            evlats  = np.array([])
            values  = np.array([])
            valuetype = 'depth'
            if incat is None:
                print 'Loading catalog'
                cat     = obspy.read_events('alaska_events.xml')
                print 'Catalog loaded!'
            else:
                cat     = incat
            for event in cat:
                event_id    = event.resource_id.id.split('=')[-1]
                porigin     = event.preferred_origin()
                pmag        = event.preferred_magnitude()
                magnitude   = pmag.mag
                Mtype       = pmag.magnitude_type
                otime       = porigin.time
                try:
                    evlo        = porigin.longitude
                    evla        = porigin.latitude
                    evdp        = porigin.depth/1000.
                except:
                    continue
                evlons      = np.append(evlons, evlo)
                evlats      = np.append(evlats, evla);
                if valuetype=='depth':
                    values  = np.append(values, evdp)
                elif valuetype=='mag':
                    values  = np.append(values, magnitude)
            ind             = (values >= depth - 5.)*(values<=depth+5.)
            x, y            = m(evlons[ind], evlats[ind])
            m.plot(x, y, 'o', mfc='white', mec='k', ms=3, alpha=0.5)
        # # # 
        # # # if vmax==None and vmin==None:
        # # #     vmax        = values.max()
        # # #     vmin        = values.min()
        # # # if gcmt:
        # # #     for i in xrange(len(focmecs)):
        # # #         value   = values[i]
        # # #         rgbcolor= cmap( (value-vmin)/(vmax-vmin) )
        # # #         b       = beach(focmecs[i], xy=(x[i], y[i]), width=100000, linewidth=1, facecolor=rgbcolor)
        # # #         b.set_zorder(10)
        # # #         ax.add_collection(b)
        # # #         # ax.annotate(str(i), (x[i]+50000, y[i]+50000))
        # # #     im          = m.scatter(x, y, marker='o', s=1, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        # # #     cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        # # #     cb.set_label(valuetype, fontsize=20)
        # # # else:
        # # #     if values.size!=0:
        # # #         im      = m.scatter(x, y, marker='o', s=300, c=values, cmap=cmap, vmin=vmin, vmax=vmax)
        # # #         cb      = m.colorbar(im, "bottom", size="3%", pad='2%')
        # # #     else:
        # # #         m.plot(x,y,'o')
        # # # if gcmt:
        # # #     stime       = self.events[0].origins[0].time
        # # #     etime       = self.events[-1].origins[0].time
        # # # else:
        # # #     etime       = self.events[0].origins[0].time
        # # #     stime       = self.events[-1].origins[0].time
        # # # plt.suptitle('Number of event: '+str(len(self.events))+' time range: '+str(stime)+' - '+str(etime), fontsize=20 )
        # # # if showfig:
        # # #     plt.show()
        #########################################################################
        
        # 
        # xc, yc      = m(np.array([-160, -145]), np.array([64, 60]))
        # m.plot(xc, yc,'g', lw = 3)
        # 
        # xc, yc      = m(np.array([-160, -147]), np.array([62, 59]))
        # m.plot(xc, yc,'k', lw = 3)
        # 
        # xc, yc      = m(np.array([-164.5]), np.array([60.]))
        # m.plot(xc, yc,'x', lw = 3, ms=15)
        
        #
        # print 'plotting data from '+dataid
        # # cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        return
    
    def plot_horizontal_discontinuity(self, depthrange, distype='moho', dtype='avg', is_smooth=True, shpfx=None, clabel='', title='',\
            cmap='cv', projection='lambert', hillshade=False, geopolygons=None, vmin=None, vmax=None, \
            lonplt=[], latplt=[], showfig=True):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        depthrange  - depth range for average
        dtype       - data type:
                        avg - average model
                        min - minimum misfit model
                        sem - uncertainties (standard error of the mean)
        is_smooth   - use the data that has been smoothed or not
        clabel      - label of colorbar
        cmap        - colormap
        projection  - projection type
        geopolygons - geological polygons for plotting
        vmin, vmax  - min/max value of plotting
        showfig     - show figure or not
        =================================================================================================================
        """
        is_interp       = self.attrs['is_interp']
        if distype is 'moho':
            if is_smooth:
                disArr  = self[dtype+'_paraval/12_smooth'].value
            else:
                disArr  = self[dtype+'_paraval/12_org'].value
        elif distype is 'sedi':
            if is_smooth:
                disArr  = self[dtype+'_paraval/11_smooth'].value
            else:
                disArr  = self[dtype+'_paraval/11_org'].value
        else:
            raise ValueError('Unexpected type of discontinuity:'+distype)
        if is_interp:
            topoArr = self['topo_interp'].value
        else:
            topoArr = self['topo'].value
        self._get_lon_lat_arr(is_interp=is_interp)
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        if depthrange < 0.:
            depth0  = disArr + depthrange
            depth1  = disArr.copy()
        else:
            depth0  = disArr 
            depth1  = disArr + depthrange
        vs_plt      = _get_vs_2d(z0=depth0, z1=depth1, zArr=zArr, vs_3d=vs3d)
        if is_interp:
            mask    = self.attrs['mask_interp']
        else:
            mask    = self.attrs['mask_inv']
        mvs         = ma.masked_array(vs_plt, mask=mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr, self.latArr)
        shapefname  = '/home/leon/geological_maps/qfaults'
        m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        shapefname  = '/home/leon/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        ################################3
        if hillshade:
            from netCDF4 import Dataset
            from matplotlib.colors import LightSource
        
            etopodata   = Dataset('/projects/life9360/station_map/grd_dir/ETOPO2v2g_f4.nc')
            etopo       = etopodata.variables['z'][:]
            lons        = etopodata.variables['x'][:]
            lats        = etopodata.variables['y'][:]
            ls          = LightSource(azdeg=315, altdeg=45)
            # nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
            etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
            # topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
            ny, nx      = etopo.shape
            topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
            m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
            mycm1=pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
            mycm2=pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
            mycm2.set_over('w',0)
            m.imshow(ls.shade(topodat, cmap=mycm1, vert_exag=1., dx=1., dy=1., vmin=0, vmax=8000))
            m.imshow(ls.shade(topodat, cmap=mycm2, vert_exag=1., dx=1., dy=1., vmin=-11000, vmax=-0.5))
        ###################################################################
        # if hillshade:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        # else:
        #     m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        
        im          = m.pcolormesh(x, y, mvs, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(clabel, fontsize=20, rotation=0)
        cb.ax.tick_params(labelsize=15)
        cb.set_alpha(1)
        cb.draw_all()
        #
        # xc, yc      = m(np.array([-150, -170]), np.array([57, 64]))
        # m.plot(xc, yc,'k', lw = 3)
        if len(lonplt) > 0 and len(lonplt) == len(latplt): 
            xc, yc      = m(lonplt, latplt)
            m.plot(xc, yc,'ko', lw = 3)
        ############################################################
        # evlons  = np.array([])
        # evlats  = np.array([])
        # values  = np.array([])
        # valuetype = 'depth'
        # cat     = obspy.read_events('alaska_events.xml')
        # for event in cat:
        #     event_id    = event.resource_id.id.split('=')[-1]
        #     porigin     = event.preferred_origin()
        #     pmag        = event.preferred_magnitude()
        #     magnitude   = pmag.mag
        #     Mtype       = pmag.magnitude_type
        #     otime       = porigin.time
        #     try:
        #         evlo        = porigin.longitude
        #         evla        = porigin.latitude
        #         evdp        = porigin.depth/1000.
        #     except:
        #         continue
        #     evlons      = np.append(evlons, evlo)
        #     evlats      = np.append(evlats, evla);
        #     if valuetype=='depth':
        #         values  = np.append(values, evdp)
        #     elif valuetype=='mag':
        #         values  = np.append(values, magnitude)
        # ind             = (values >= depth - 5.)*(values<=depth+5.)
        # x, y            = m(evlons[ind], evlats[ind])
        # m.plot(x, y, 'ko', ms=3, alpha=0.5)
        #########################################################################
        
        # 
        # xc, yc      = m(np.array([-155, -170]), np.array([56, 60]))
        # m.plot(xc, yc,'k', lw = 3)
        # 
        # xc, yc      = m(np.array([-164]), np.array([59.5]))
        # m.plot(xc, yc,'x', lw = 3, ms=15)
        # 
        # xc, yc      = m(np.array([-164.5]), np.array([60.]))
        # m.plot(xc, yc,'x', lw = 3, ms=15)
        
        #
        # print 'plotting data from '+dataid
        # # cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        return
    
    def plot_vertical_rel(self, lon1, lat1, lon2, lat2, maxdepth, vs_mantle=4.4, plottype = 0, d = 10., dtype='min', is_smooth=True,\
                      clabel='', cmap='cv', vmin1=3.0, vmax1=4.2, vmin2=-10., vmax2=10., incat=None, dist_thresh=20., showfig=True):
        is_interp   = self.attrs['is_interp']
        if is_smooth:
            mohoArr = self[dtype+'_paraval/12_smooth'].value
        else:
            mohoArr = self[dtype+'_paraval/12_org'].value
        if is_interp:
            topoArr = self['topo_interp'].value
        else:
            topoArr = self['topo'].value
        if lon1 == lon2 and lat1 == lat2:
            raise ValueError('The start and end points are the same!')
        self._get_lon_lat_arr(is_interp=is_interp)
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        ind_z       = np.where(zArr <= maxdepth )[0]
        zplot       = zArr[ind_z]
        if lon1 == lon2 or lat1 == lat2:
            if lon1 == lon2:    
                ind_lon = np.where(self.lons == lon1)[0]
                ind_lat = np.where((self.lats<=max(lat1, lat2))*(self.lats>=min(lat1, lat2)))[0]
                # data    = np.zeros((len(ind_lat), ind_z.size))
            else:
                ind_lon = np.where((self.lons<=max(lon1, lon2))*(self.lons>=min(lon1, lon2)))[0]
                ind_lat = np.where(self.lats == lat1)[0]
                # data    = np.zeros((len(ind_lon), ind_z.size))
            data_temp   = vs3d[ind_lat, ind_lon, :]
            data        = data_temp[:, ind_z]
            if lon1 == lon2:
                xplot       = self.lats[ind_lat]
                xlabel      = 'latitude (deg)'
            if lat1 == lat2:
                xplot       = self.lons[ind_lon]
                xlabel      = 'longitude (deg)'
            # 
            topo1d          = topoArr[ind_lat, ind_lon]
            moho1d          = mohoArr[ind_lat, ind_lon]
            #
            data_moho       = data.copy()
            mask_moho       = np.ones(data.shape, dtype=bool)
            data_mantle     = data.copy()
            mask_mantle     = np.ones(data.shape, dtype=bool)
            for ix in range(data.shape[0]):
                ind_moho    = zplot <= moho1d[ix]
                ind_mantle  = np.logical_not(ind_moho)
                mask_moho[ix, ind_moho] \
                            = False
                mask_mantle[ix, ind_mantle] \
                            = False
                data_mantle[ix, :] \
                            = (data_mantle[ix, :] - vs_mantle)/vs_mantle*100.
        else:
            g               = Geod(ellps='WGS84')
            az, baz, dist   = g.inv(lon1, lat1, lon2, lat2)
            dist            = dist/1000.
            d               = dist/float(int(dist/d))
            Nd              = int(dist/d)
            lonlats         = g.npts(lon1, lat1, lon2, lat2, npts=Nd-1)
            lonlats         = [(lon1, lat1)] + lonlats
            lonlats.append((lon2, lat2))
            data            = np.zeros((len(lonlats), ind_z.size))
            L               = self.lonArr.size
            vlonArr         = self.lonArr.reshape(L)
            vlatArr         = self.latArr.reshape(L)
            ind_data        = 0
            plons           = np.zeros(len(lonlats))
            plats           = np.zeros(len(lonlats))
            topo1d          = np.zeros(len(lonlats))
            moho1d          = np.zeros(len(lonlats))
            for lon,lat in lonlats:
                if lon < 0.:
                    lon     += 360.
                clonArr         = np.ones(L, dtype=float)*lon
                clatArr         = np.ones(L, dtype=float)*lat
                az, baz, dist   = g.inv(clonArr, clatArr, vlonArr, vlatArr)
                ind_min         = dist.argmin()
                ind_lat         = int(np.floor(ind_min/self.Nlon))
                ind_lon         = ind_min - self.Nlon*ind_lat
                azmin, bazmin, distmin = g.inv(lon, lat, self.lons[ind_lon], self.lats[ind_lat])
                if distmin != dist[ind_min]:
                    raise ValueError('DEBUG!')
                data[ind_data, :]   \
                                = vs3d[ind_lat, ind_lon, ind_z]
                plons[ind_data] = lon
                plats[ind_data] = lat
                topo1d[ind_data]= topoArr[ind_lat, ind_lon]
                moho1d[ind_data]= mohoArr[ind_lat, ind_lon]
                ind_data        += 1
            data_moho           = data.copy()
            mask_moho           = np.ones(data.shape, dtype=bool)
            data_mantle         = data.copy()
            mask_mantle         = np.ones(data.shape, dtype=bool)
            for ix in range(data.shape[0]):
                ind_moho        = zplot <= moho1d[ix]
                ind_mantle      = np.logical_not(ind_moho)
                mask_moho[ix, ind_moho] \
                                = False
                mask_mantle[ix, ind_mantle] \
                                = False
                data_mantle[ix, :] \
                                = (data_mantle[ix, :] - vs_mantle)/vs_mantle*100.
            if plottype == 0:
                xplot   = plons
                xlabel  = 'longitude (deg)'
            else:
                xplot   = plats
                xlabel  = 'latitude (deg)'
        cmap1           = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        cmap2           = pycpt.load.gmtColormap('./cv.cpt')
        f, (ax1, ax2)   = plt.subplots(2, sharex=True, sharey=False, gridspec_kw={'height_ratios':[1,4]})
        topo1d[topo1d<0.]   \
                        = 0.
        ax1.plot(xplot, topo1d*1000., 'k', lw=3)
        ax1.fill_between(xplot, 0, topo1d*1000., facecolor='grey')
        ax1.set_ylabel('Elevation (m)', fontsize=20)
        ax1.set_ylim(0, topo1d.max()*1000.+10.)
        mdata_moho      = ma.masked_array(data_moho, mask=mask_moho )
        mdata_mantle    = ma.masked_array(data_mantle, mask=mask_mantle )
        m1              = ax2.pcolormesh(xplot, zplot, mdata_mantle.T, shading='gouraud', vmax=vmax2, vmin=vmin2, cmap=cmap1)
        cb1             = f.colorbar(m1, orientation='horizontal', fraction=0.05)
        cb1.set_label('Mantle Vs perturbation relative to '+str(vs_mantle)+' km/s (%)', fontsize=20)
        cb1.ax.tick_params(labelsize=10) 
        m2              = ax2.pcolormesh(xplot, zplot, mdata_moho.T, shading='gouraud', vmax=vmax1, vmin=vmin1, cmap=cmap2)
        cb2             = f.colorbar(m2, orientation='horizontal', fraction=0.06)
        cb2.set_label('Crustal Vs (km/s)', fontsize=20)
        cb2.ax.tick_params(labelsize=10) 
        #
        ax2.plot(xplot, moho1d, 'r', lw=3)
        #
        ax2.set_xlabel(xlabel, fontsize=20)
        ax2.set_ylabel('Depth (km)', fontsize=20)
        f.subplots_adjust(hspace=0)
        # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        # # plt.axis([self.xgrid[0], self.xgrid[-1], self.ygrid[0], self.ygrid[-1]], 'scaled')
        # cb      = plt.colorbar()
        # cb.set_label('Vs (km/s)', fontsize=30)
        ############################################################
        lonlats_arr \
                = np.asarray(lonlats)
        lons_arr= lonlats_arr[:, 0]
        lats_arr= lonlats_arr[:, 1]
        evlons  = np.array([])
        evlats  = np.array([])
        values  = np.array([])
        valuetype = 'depth'
        if incat is None:
            print 'Loading catalog'
            cat     = obspy.read_events('alaska_events.xml')
            print 'Catalog loaded!'
        else:
            cat     = incat
        Nevent      = 0
        for event in cat:
            event_id    = event.resource_id.id.split('=')[-1]
            porigin     = event.preferred_origin()
            pmag        = event.preferred_magnitude()
            magnitude   = pmag.mag
            Mtype       = pmag.magnitude_type
            otime       = porigin.time
            try:
                evlo        = porigin.longitude
                evla        = porigin.latitude
                evdp        = porigin.depth/1000.
            except:
                continue
            # # # Nevent      += 1
            # # # print Nevent
            az, baz, dist \
                            = g.inv(lons_arr, lats_arr, np.ones(lons_arr.size)*evlo, np.ones(lons_arr.size)*evla)
            # print dist.min()/1000.
            if evlo < 0.:
                evlo        += 360.
            if dist.min()/1000. < dist_thresh:
                evlons      = np.append(evlons, evlo)
                evlats      = np.append(evlats, evla)
                if valuetype=='depth':
                    values  = np.append(values, evdp)
                elif valuetype=='mag':
                    values  = np.append(values, magnitude)
            # 
            # for lon,lat in lonlats:
            #     if lon < 0.:
            #         lon     += 360.
            #     dist, az, baz \
            #                 = obspy.geodetics.gps2dist_azimuth(lat, lon, evla, evlo)
            #     # az, baz, dist \
            #     #             = g.inv(lon, lat, evlo, evla)
            #     if dist/1000. < 10.:
            #         evlons      = np.append(evlons, evlo)
            #         evlats      = np.append(evlats, evla)
            #     if valuetype=='depth':
            #         values  = np.append(values, evdp)
            #     elif valuetype=='mag':
            #         values  = np.append(values, magnitude)
            #         break
 
        # # # for lon,lat in lonlats:
        # # #     if lon < 0.:
        # # #         lon     += 360.
        # # #     for event in cat:
        # # #         event_id    = event.resource_id.id.split('=')[-1]
        # # #         porigin     = event.preferred_origin()
        # # #         pmag        = event.preferred_magnitude()
        # # #         magnitude   = pmag.mag
        # # #         Mtype       = pmag.magnitude_type
        # # #         otime       = porigin.time
        # # #         try:
        # # #             evlo        = porigin.longitude
        # # #             evla        = porigin.latitude
        # # #             evdp        = porigin.depth/1000.
        # # #         except:
        # # #             continue
        # # #         if evlo < 0.:
        # # #             evlo    += 360.
        # # #         if abs(evlo-lon)<0.1 and abs(evla-lat)<0.1:
        # # #             evlons      = np.append(evlons, evlo)
        # # #             evlats      = np.append(evlats, evla)
        # # #             if valuetype=='depth':
        # # #                 values  = np.append(values, evdp)
        # # #             elif valuetype=='mag':
        # # #                 values  = np.append(values, magnitude)
        # print evlons.size
        if plottype == 0:
            ax2.plot(evlons, values, 'o', mfc='white', mec='k', ms=5, alpha=0.8)
        else:
            ax2.plot(evlats, values, 'o', mfc='white', mec='k', ms=5, alpha=0.8)
        #########################################################################
        ax1.tick_params(axis='y', labelsize=20)
        ax2.tick_params(axis='x', labelsize=20)
        ax2.tick_params(axis='y', labelsize=20)
        ax2.set_ylim([zplot[0], zplot[-1]])
        ax2.set_xlim([xplot[0], xplot[-1]])
        plt.gca().invert_yaxis()
        if showfig:
            plt.show()
        return
                    
    def plot_vertical_abs(self, lon1, lat1, lon2, lat2, maxdepth, plottype = 0, d = 10., dtype='min', is_smooth=False,\
                      clabel='', cmap='cv', vmin=None, vmax=None, showfig=True):        
        if lon1 == lon2 and lat1 == lat2:
            raise ValueError('The start and end points are the same!')
        self._get_lon_lat_arr()
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        ind_z       = np.where(zArr <= maxdepth )[0]
        zplot       = zArr[ind_z]
        if lon1 == lon2 or lat1 == lat2:
            if lon1 == lon2:    
                ind_lon = np.where(self.lons == lon1)[0]
                ind_lat = np.where((self.lats<=max(lat1, lat2))*(self.lats>=min(lat1, lat2)))[0]
                # data    = np.zeros((len(ind_lat), ind_z.size))
            else:
                ind_lon = np.where((self.lons<=max(lon1, lon2))*(self.lons>=min(lon1, lon2)))[0]
                ind_lat = np.where(self.lats == lat1)[0]
                # data    = np.zeros((len(ind_lon), ind_z.size))
            data_temp   = vs3d[ind_lat, ind_lon, :]
            data        = data_temp[:, ind_z]
            # return data, data_temp
            if lon1 == lon2:
                xplot       = self.lats[ind_lat]
                xlabel      = 'latitude (deg)'
            if lat1 == lat2:
                xplot       = self.lons[ind_lon]
                xlabel      = 'longitude (deg)'            
        else:
            g               = Geod(ellps='WGS84')
            az, baz, dist   = g.inv(lon1, lat1, lon2, lat2)
            dist            = dist/1000.
            d               = dist/float(int(dist/d))
            Nd              = int(dist/d)
            lonlats         = g.npts(lon1, lat1, lon2, lat2, npts=Nd-1)
            lonlats         = [(lon1, lat1)] + lonlats
            lonlats.append((lon2, lat2))
            data            = np.zeros((len(lonlats), ind_z.size))
            L               = self.lonArr.size
            vlonArr         = self.lonArr.reshape(L)
            vlatArr         = self.latArr.reshape(L)
            ind_data        = 0
            plons           = np.zeros(len(lonlats))
            plats           = np.zeros(len(lonlats))
            for lon,lat in lonlats:
                if lon < 0.:
                    lon     += 360.
                # if lat <
                # print lon, lat
                clonArr         = np.ones(L, dtype=float)*lon
                clatArr         = np.ones(L, dtype=float)*lat
                az, baz, dist   = g.inv(clonArr, clatArr, vlonArr, vlatArr)
                ind_min         = dist.argmin()
                ind_lat         = int(np.floor(ind_min/self.Nlon))
                ind_lon         = ind_min - self.Nlon*ind_lat
                # 
                azmin, bazmin, distmin = g.inv(lon, lat, self.lons[ind_lon], self.lats[ind_lat])
                if distmin != dist[ind_min]:
                    raise ValueError('DEBUG!')
                #
                data[ind_data, :]   \
                                = vs3d[ind_lat, ind_lon, ind_z]
                plons[ind_data] = lon
                plats[ind_data] = lat
                ind_data        += 1
            # data[0, :]          = 
            if plottype == 0:
                xplot   = plons
                xlabel  = 'longitude (deg)'
            else:
                xplot   = plats
                xlabel  = 'latitude (deg)'
                
        if cmap == 'ses3d':
            cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
                            0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        elif cmap == 'cv':
            import pycpt
            cmap    = pycpt.load.gmtColormap('./cv.cpt')
        else:
            try:
                if os.path.isfile(cmap):
                    import pycpt
                    cmap    = pycpt.load.gmtColormap(cmap)
            except:
                pass
        ax      = plt.subplot()
        plt.pcolormesh(xplot, zplot, data.T, shading='gouraud', vmax=vmax, vmin=vmin, cmap=cmap)
        plt.xlabel(xlabel, fontsize=30)
        plt.ylabel('depth (km)', fontsize=30)
        plt.gca().invert_yaxis()
        # plt.axis([self.xgrid[0], self.xgrid[-1], self.ygrid[0], self.ygrid[-1]], 'scaled')
        cb=plt.colorbar()
        cb.set_label('Vs (km/s)', fontsize=30)
        ax.tick_params(axis='x', labelsize=20)
        ax.tick_params(axis='y', labelsize=20)
        if showfig:
            plt.show()


    