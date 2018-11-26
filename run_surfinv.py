
import surfdbase
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180919_3d.h5')
import copy
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180902_3d.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180907.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180913.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180915.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180917.h5')
# 
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180919_single.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180919.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180919_3d.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180822_3d.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180928.h5')

dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20181105.h5')
#-------------------------
# before inversion
#-------------------------
# dset.read_raytomo_dbase(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_LD.h5', runid=2, Tmin=8., Tmax=50.)
# dset.read_hybridtomo_dbase(inh5fname='/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20181101.h5', runid=0)
# dset.read_crust_thickness()
# dset.read_sediment_thickness()
# # # dset.read_CU_model()
# dset.read_etopo(infname='/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
# # # # # 
# # # # group
# dset.read_raytomo_dbase_group(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_20180823_gr.h5', runid=2, Tmin=8., Tmax=50.)

# #-------------------------
# # inversion
# #-------------------------
# # dset.mc_inv_iso(outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20180822_150000',
# #                 numbrun=150000, nprocess=30, verbose=False)
# 
# # dset.mc_inv_iso(outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20180913_150000_both',
# #                 numbrun=150000, nprocess=35, verbose=False, group=True)
# 
# # dset.mc_inv_iso(outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20180917_150000_both',
# #                 numbrun=150000, nprocess=35, verbose=False, group=True)
# 
# dset.mc_inv_iso(use_ref=False, outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both',
#                 numbrun=150000, nprocess=35, verbose=False, group=True)

# dset.mc_inv_iso(outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20180915_150000_both',
#                 numbrun=150000, nprocess=35, verbose=False, group=False)



# vpr = dset.mc_inv_iso(outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20180919_150000_both',
#                 numbrun=150000, nprocess=30, verbose=False, group=True)

# vpr = dset.mc_inv_iso(use_ref=False, outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both',
#                 numbrun=150000, nprocess=35, verbose=False, group=True)
# 
# 
# # # # # vpr_ph          = copy.deepcopy(vpr)
# # # # # vpr_ph.data.dispR.isgroup  \
# # # # #                 = False
# # # # # 
# # # # # vpr_gr          = copy.deepcopy(vpr)
# # # # # vpr_gr.data.dispR.isphase  \
# # # # #                 = False
# # # # 
# vpr.mc_joint_inv_iso_mp(outdir='./workingdir', dispdtype='both', pfx='BOTH', wdisp=1., nprocess=35, verbose=True, numbrun=150000, step4uwalk=1500)
# vpr.mc_joint_inv_iso_mp(outdir='./workingdir', dispdtype='both', pfx='BOTH', wdisp=1., nprocess=8, verbose=True, numbrun=150000)
# vpr_gr.mc_joint_inv_iso_mp(outdir='./workingdir', dispdtype='both', pfx='GR', wdisp=1., nprocess=35, verbose=True, numbrun=150000)
# vpr_ph.mc_joint_inv_iso_mp(outdir='./workingdir', dispdtype='both', pfx='PH', wdisp=1., nprocess=35, verbose=True, numbrun=150000)
#-------------------------
# read inversion results
#-------------------------
# dset.read_inv(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both', avgqc=False)
# # 
# # #-------------------------
# # # interpolation/smoothingdset.get_raytomo_mask(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_LD.h5', runid=2)
# # #-------------------------
# dset.get_raytomo_mask(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_LD.h5', runid=2)
# dset.get_hybrid_mask(inh5fname='/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20181101.h5', runid=0)
# dset.get_topo_arr(infname='/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
# # 
# dset.paraval_arrays(dtype='min')
# dset.construct_3d(dtype='min')
# dset.construct_3d(dtype='min', is_smooth=True)
# 
# dset.paraval_arrays(dtype='avg')
# dset.construct_3d(dtype='avg')
# dset.construct_3d(dtype='avg', is_smooth=True)

# dset.paraval_arrays(dtype='avg', width=50.)
# dset.construct_3d(dtype='avg')
# dset.construct_3d(dtype='avg', is_smooth=True)
# # # 
# depth = 80.
# # dset.plot_horizontal(depth=depth, dtype='min', is_smooth=False, shpfx=None, clabel='Vs (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
# #              geopolygons=None, vmin=4.2, vmax=4.6, showfig=True)
# # 
# # dset.plot_horizontal(depth=depth, dtype='avg', is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
# #              geopolygons=None, vmin=None, vmax=None, showfig=True)
# # 
# dset.plot_horizontal(depth=depth, dtype='avg', is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
#              geopolygons=None, vmin=4.2, vmax=4.6, showfig=True)
# # # # 
# # # # dset.plot_horizontal(depth=10., dtype='avg', is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title='4 km', projection='lambert', hillshade=False,\
# # # #              geopolygons=None, vmin=None, vmax=None, showfig=True)
# # # 
# dset.plot_horizontal(depth=4., dtype='avg', is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title='4 km', projection='lambert', hillshade=False,\
#              geopolygons=None, vmin=2.5, vmax=3.5, showfig=True)
# dset.plot_vertical_rel(lon1=-165+360, lon2=-150+360, lat1=65, lat2=55, maxdepth=100., dtype='avg', is_smooth=True)

# dset.plot_vertical_rel(lon1=-170+360, lon2=-145+360, lat1=64, lat2=58, maxdepth=100., dtype='avg', is_smooth=True)

# dset.plot_vertical_rel(lon1=-153+360, lon2=-142+360, lat1=61, lat2=64, maxdepth=150., dtype='avg', is_smooth=True)


# vpr = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20180829_150000_both', lon=-142., lat=62.)
# vpr = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both', lon=-147., lat=66.5)

# vpr = dset.generate_disp_vs_figs(datadir ='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both',\
#                 outdir='/home/leon/ALASKA_disp_vs')

# dset.plot_paraval(pindex='min_misfit', is_smooth=False, cmap='jet', vmin=0.2, vmax=2.0, outfname='min_misfit.txt')
# dset.plot_paraval(pindex='avg_misfit', is_smooth=False, cmap='jet', vmin=0.2, vmax=2.0, outfname='avg_misfit.txt')
# dset.plot_paraval(pindex='fitratio', is_smooth=False, cmap='jet', vmin=0.3, vmax=1.0, outfname='fitratio.txt')


