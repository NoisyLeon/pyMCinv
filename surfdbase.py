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
        self.lonArr, self.latArr \
                        = np.meshgrid(self.lons, self.lats)
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
                group.attrs.create(name='mask', data = mask[ilat, ilon])
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
            if self[grd_id].attrs['mask'] and skipmask:
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
            print '--- MC inversion for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
            if parallel:
                vpr.mc_joint_inv_iso_mp(outdir=outdir, dispdtype=dispdtype, wdisp=1.,\
                   monoc=monoc, pfx=grd_id, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun, subsize=subsize, nprocess=nprocess)
            else:
                vpr.mc_joint_inv_iso(outdir=outdir, dispdtype=dispdtype, wdisp=1., \
                   monoc=monoc, pfx=grd_id, verbose=verbose, step4uwalk=step4uwalk, numbrun=numbrun)
            # # # return
        return
    
    def read_inv(self, datadir, ingrdfname=None, factor=1., thresh=0.5, skipmask=True):
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
            if grp.attrs['mask'] and skipmask:
                print '--- Skipping inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
                continue
            print '--- Reading inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd)
            invfname    = datadir+'/mc_inv.'+ grd_id+'.npz'
            datafname   = datadir+'/mc_data.'+grd_id+'.npz'
            if not (os.path.isfile(invfname) and os.path.isfile(datafname)):
                raise ValueError('No inversion results for grid: lon = '+str(grd_lon)+', lat = '+str(grd_lat)+', '+str(igrd)+'/'+str(Ngrd))
            topovalue   = grp.attrs['topo']
            vpr         = mcpost.postvpr(waterdepth=-topovalue, factor=factor, thresh=thresh)
            vpr.read_inv_data(infname = invfname, verbose=False)
            vpr.read_data(infname = datafname)
            vpr.get_paraval()
            vpr.run_avg_fwrd(wdisp=1.)
            #------------------------------------------
            # store inversion results in the database
            #------------------------------------------
            grp.create_dataset(name = 'avg_paraval', data = vpr.avg_paraval)
            grp.create_dataset(name = 'min_paraval', data = vpr.min_paraval)
            grp.create_dataset(name = 'sem_paraval', data = vpr.sem_paraval)
            grp.attrs.create(name = 'avg_misfit', data = vpr.vprfwrd.data.misfit)
            grp.attrs.create(name = 'min_misfit', data = vpr.min_misfit)
        return
    
    def get_paraval(self, pindex, dtype='min', ingrdfname=None, isthk=False):
        """
        get the data and corresponding mask array for the model parameter
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
        self._get_lon_lat_arr()
        data        = np.zeros(self.latArr.shape)
        mask        = np.ones(self.latArr.shape, dtype=bool)
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
            mask[ind_lat, ind_lon]      = False
        if not np.allclose(mask, self.attrs['mask']):
            print 'WARNING: check the mask array!'
        return data, mask
    
    def get_filled_paraval(self, pindex, dtype='min', ingrdfname=None, isthk=False):
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
        ==================================================================================================================
        """
        minlon      = self.attrs['minlon']
        maxlon      = self.attrs['maxlon']
        minlat      = self.attrs['minlat']
        maxlat      = self.attrs['maxlat']
        data, mask  = self.get_paraval(pindex=pindex, dtype=dtype, ingrdfname=ingrdfname, isthk=isthk)
        ind_valid   = np.logical_not(mask)
        data_out    = data.copy()
        g           = Geod(ellps='WGS84')
        vlonArr     = self.lonArr[ind_valid]
        vlatArr     = self.latArr[ind_valid]
        vdata       = data[ind_valid]
        L           = vlonArr.size
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                if not mask[ilat, ilon]:
                    continue
                clonArr         = np.ones(L, dtype=float)*self.lons[ilon]
                clatArr         = np.ones(L, dtype=float)*self.lats[ilat]
                az, baz, dist   = g.inv(clonArr, clatArr, vlonArr, vlatArr)
                ind_min         = dist.argmin()
                data_out[ilat, ilon] \
                                = vdata[ind_min]
        return data_out, mask
    
    def get_smooth_paraval(self, pindex, sigma, dtype='min', ingrdfname=None, isthk=False): #, vmin=None, vmax=None, clabel=''):
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
        ingrdfname  - input grid point list file indicating the grid points for surface wave inversion
        isthk       - flag indicating if the parameter is thickness or not
        ==================================================================================================================
        """
        data, mask  = self.get_filled_paraval(pindex=pindex, dtype=dtype, ingrdfname=ingrdfname, isthk=isthk)
        data_smooth = data.copy()
        #- Smoothing by averaging over neighbouring cells. ----------------------
        for iteration in range(int(sigma)):
            for ilat in range(1, self.Nlat-1):
                for ilon in range(1, self.Nlon-1):
                    data_smooth[ilat, ilon] = (data[ilat, ilon] + data[ilat+1, ilon] \
                                               + data[ilat-1, ilon] + data[ilat, ilon+1] + data[ilat, ilon-1])/5.0
        return data, data_smooth, mask
        #-----------
        # plot data
        #-----------
        # m           = self._get_basemap(projection='lambert')
        # x, y        = m(self.lonArr, self.latArr)
        # cmap        = 'cv'
        # if cmap == 'ses3d':
        #     cmap        = colormaps.make_colormap({0.0:[0.1,0.0,0.0], 0.2:[0.8,0.0,0.0], 0.3:[1.0,0.7,0.0],0.48:[0.92,0.92,0.92],
        #                     0.5:[0.92,0.92,0.92], 0.52:[0.92,0.92,0.92], 0.7:[0.0,0.6,0.7], 0.8:[0.0,0.0,0.8], 1.0:[0.0,0.0,0.1]})
        # elif cmap == 'cv':
        #     import pycpt
        #     cmap    = pycpt.load.gmtColormap('./cv.cpt')
        # else:
        #     try:
        #         if os.path.isfile(cmap):
        #             import pycpt
        #             cmap    = pycpt.load.gmtColormap(cmap)
        #     except:
        #         pass
        # im          = m.pcolormesh(x, y, data_smooth, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        # cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        # cb.set_label(clabel, fontsize=12, rotation=0)
        # cb.ax.tick_params(labelsize=15)
        # cb.set_alpha(1)
        # cb.draw_all()
        # # # cb.solids.set_rasterized(True)
        # cb.solids.set_edgecolor("face")
        # # m.shadedrelief(scale=1., origin='lower')
        # # if showfig:
        # plt.show()
        # # plot 
        # return data_out
    
    def paraval_arrays(self, dtype='min', sigma=1):
        """
        get the paraval arrays and store them in the data base
        =================================================================
        ::: input :::
        sigma       - total number of smooth iterations
        dtype       - data type:
                    avg - average model
                    min - minimum misfit model
                    sem - uncertainties (standard error of the mean)
        =================================================================
        """
        grp                 = self.require_group( name = dtype+'_paraval' )
        for pindex in range(13):
            if pindex == 11 or pindex == 12:
                data, data_smooth, mask = self.get_smooth_paraval(pindex=pindex, dtype=dtype, sigma=sigma, isthk=True)
            else:
                data, data_smooth, mask = self.get_smooth_paraval(pindex=pindex, dtype=dtype, sigma=sigma, isthk=False)
            grp.create_dataset(name = str(pindex)+'_org', data = data)
            grp.create_dataset(name = str(pindex)+'_smooth', data = data_smooth)
        grp.create_dataset(name = 'mask', data = mask)
        return
    
    # def paraval_arrays_HD(self, dtype='min'):
    #     grp     = self[dtype+'_paraval']
    #     self._get_lon_lat_arr()
        
        
        
    
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
        grp     = self[dtype+'_paraval']
        self._get_lon_lat_arr()
        Nz      = int(maxdepth/dz) + 1
        zArr    = np.arange(Nz)*dz
        vs3d    = np.zeros((self.latArr.shape[0], self.latArr.shape[1], Nz))
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                paraval             = np.zeros(13, dtype=np.float64)
                grd_id              = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                topovalue           = self[grd_id].attrs['topo']
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
                    vel_mod.zArr    = vel_mod.zArr - topovalue
                # interpolation
                # vs_interp           = np.interp(zArr, xp = vel_mod.zArr, fp = vel_mod.VsvArr)
                # vs3d[ilat, ilon, :] = vs_interp[:]
                #
                return vel_mod
                from scipy import interpolate
                f                   = interpolate.interp1d(x = vel_mod.zArr, y = vel_mod.VsvArr, kind='cubic')
                vs3d[ilat, ilon, :] = f(vs_interp[:])   # use interpolation function returned by `interp1d`
                # plt.plot(x, y, 'o', xnew, ynew, '-')
                
        if is_smooth:
            grp.create_dataset(name = 'vs_smooth', data = vs3d)
            grp.create_dataset(name = 'z_smooth', data = zArr)
        else:
            grp.create_dataset(name = 'vs_org', data = vs3d)
            grp.create_dataset(name = 'z_org', data = zArr)
        return
    
        
    def get_topo_arr(self):
        """
        get the topography array
        """
        self._get_lon_lat_arr()
        topoarr     = np.zeros(self.lonArr.shape)
        for ilat in range(self.Nlat):
            for ilon in range(self.Nlon):
                grd_id              = str(self.lons[ilon])+'_'+str(self.lats[ilat])
                topovalue           = self[grd_id].attrs['topo']
                topoarr[ilat, ilon] = topovalue
        self.create_dataset(name='topo', data = topoarr)
        return
            
            
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
        
    
    def plot_paraval(self, pindex, org_mask=False, dtype='min', ingrdfname=None, isthk=False, shpfx=None,\
            clabel='', cmap='cv', projection='lambert', hillshade=False, geopolygons=None, vmin=None, vmax=None, showfig=True):
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
        # # # mask        = self.attrs['mask']
        data, mask  = self.get_paraval(pindex=pindex, dtype=dtype, ingrdfname=ingrdfname, isthk=isthk)
        if org_mask:
            mask    = self.attrs['mask']
        mdata       = ma.masked_array(data, mask=mask )
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection)
        x, y        = m(self.lonArr, self.latArr)
        shapefname  = '/projects/life9360/geological_maps/qfaults'
        m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        shapefname  = '/projects/life9360/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        # shapefname  = '../AKfaults/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        # shapefname  = '../AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        # shapefname  = '/projects/life9360/AK_sediments/Cook_Inlet_sediments_WGS84'
        # m.readshapefile(shapefname, 'faultline', linewidth=1, color='blue')
        
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
        if hillshade:
            m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        else:
            m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        if hillshade:
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax, alpha=.5)
        else:
            im          = m.pcolormesh(x, y, mdata, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(clabel, fontsize=12, rotation=0)
        cb.ax.tick_params(labelsize=15)
        cb.set_alpha(1)
        cb.draw_all()
        # # cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        return
    
    def plot_horizontal(self, depth, dtype='min', is_smooth=False, shpfx=None, clabel='', title='', cmap='cv', projection='lambert', hillshade=False,\
             geopolygons=None, vmin=None, vmax=None, showfig=True):
        """plot maps from the tomographic inversion
        =================================================================================================================
        ::: input parameters :::
        depth       - depth of the slice for plotting
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
        self._get_lon_lat_arr()
        grp         = self[dtype+'_paraval']
        if is_smooth:
            vs3d    = grp['vs_smooth'].value
            zArr    = grp['z_smooth'].value
        else:
            vs3d    = grp['vs_org'].value
            zArr    = grp['z_org'].value
        try:
            index   = np.where(zArr >= depth )[0][0]
        except IndexError:
            print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
            return
        depth       = zArr[index]
        mask        = grp['mask']
        mvs         = ma.masked_array(vs3d[:, :, index], mask=mask )
        
        #-----------
        # plot data
        #-----------
        m           = self._get_basemap(projection=projection, geopolygons=geopolygons)
        x, y        = m(self.lonArr, self.latArr)
        shapefname  = '/projects/life9360/geological_maps/qfaults'
        m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        shapefname  = '/projects/life9360/AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        # shapefname  = '../AKfaults/qfaults'
        # m.readshapefile(shapefname, 'faultline', linewidth=2, color='grey')
        # shapefname  = '../AKgeol_web_shp/AKStategeolarc_generalized_WGS84'
        # m.readshapefile(shapefname, 'geolarc', linewidth=1, color='grey')
        # shapefname  = '/projects/life9360/AK_sediments/Cook_Inlet_sediments_WGS84'
        # m.readshapefile(shapefname, 'faultline', linewidth=1, color='blue')
        
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
        if hillshade:
            m.fillcontinents(lake_color='#99ffff',zorder=0.2, alpha=0.2)
        else:
            m.fillcontinents(lake_color='#99ffff',zorder=0.2)
        
        im          = m.pcolormesh(x, y, mvs, cmap=cmap, shading='gouraud', vmin=vmin, vmax=vmax)
        cb          = m.colorbar(im, "bottom", size="3%", pad='2%')
        cb.set_label(clabel, fontsize=20, rotation=0)
        cb.ax.tick_params(labelsize=15)
        cb.set_alpha(1)
        cb.draw_all()
        # print 'plotting data from '+dataid
        # # cb.solids.set_rasterized(True)
        cb.solids.set_edgecolor("face")
        plt.suptitle(title, fontsize=30)
        # m.shadedrelief(scale=1., origin='lower')
        if showfig:
            plt.show()
        return

    
    def plot_vertical_rel(self, lon1, lat1, lon2, lat2, maxdepth, vs_mantle=4.4, plottype = 0, d = 10., dtype='min', is_smooth=False,\
                      clabel='', cmap='cv', vmin1=3.0, vmax1=4.2, vmin2=-10., vmax2=10., showfig=True):
        if is_smooth:
            mohoArr = self[dtype+'_paraval/12_smooth'].value
        else:
            mohoArr = self[dtype+'_paraval/12_org'].value
        topoArr     = self['topo'].value
        
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
                # 
                azmin, bazmin, distmin = g.inv(lon, lat, self.lons[ind_lon], self.lats[ind_lat])
                if distmin != dist[ind_min]:
                    raise ValueError('DEBUG!')
                #
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
        ax1.set_ylabel('Elevation (m)', fontsize=30)
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
        ax2.set_xlabel(xlabel, fontsize=30)
        ax2.set_ylabel('Depth (km)', fontsize=30)
        plt.gca().invert_yaxis()
        f.subplots_adjust(hspace=0)
        # plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
        # # plt.axis([self.xgrid[0], self.xgrid[-1], self.ygrid[0], self.ygrid[-1]], 'scaled')
        # cb      = plt.colorbar()
        # cb.set_label('Vs (km/s)', fontsize=30)
        ax1.tick_params(axis='y', labelsize=20)
        ax2.tick_params(axis='x', labelsize=20)
        ax2.tick_params(axis='y', labelsize=20)
        plt.xlim([xplot[0], xplot[-1]])
        if showfig:
            plt.show()
            
            
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
        
        
                 
        
        
        #     
        # 
        #     
        # try:
        #     index   = np.where(zArr >= depth )[0][0]
        # except IndexError:
        #     print 'depth slice required is out of bound, maximum depth = '+str(zArr.max())+' km'
        #     return
        # depth       = zArr[index]
        # mask        = grp['mask']
        # mvs         = ma.masked_array(vs3d[:, :, index], mask=mask )
        # if 
    
    
    
    # def plot_paraval(self, pindex):
        
    
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
# # #             crtthk           = self.auxiliary_data['MohoDepth'][staid_aux].parameters['moho_depth']
# # #             sedthk            = self.auxiliary_data['SediDepth'][staid_aux].parameters['sedi_depth']
# # #             vpr.model.isomod.parameterize_input(zarr=vsdata[:, 0], vsarr=vsdata[:, 1], crtthk=crtthk, sedthk=sedthk, maxdepth=200.)
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


    