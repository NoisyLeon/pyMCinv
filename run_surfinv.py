
import surfdbase
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180919_3d.h5')
import copy

# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_final.h5')
dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_ready4post.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20181203.h5')

# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20181128.h5')
# #-------------------------
# # before inversion
# #-------------------------
# dset.read_raytomo_dbase(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_LD.h5', runid=2, Tmin=8., Tmax=50.)
# OR
dset.read_hybridtomo_dbase(inh5fname='/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20181101.h5', runid=0, semfactor=2.)
dset.read_etopo(infname='/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
dset.read_crust_thickness(replace_moho=10., 
    infname_refine='/home/leon/miller_alaskamoho_srl2018-1.2.2/miller_alaskamoho_srl2018/Models/AlaskaMoHiErrs-AlaskaMohoFineGrid.npz')
dset.read_sediment_thickness()
# # # # dset.read_CU_model()
# # # # # # 
# # # # # group
dset.read_raytomo_dbase_group(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_20180823_gr.h5', runid=2, Tmin=8., Tmax=50.)
# # # # 
# # #-------------------------
# # # inversion
# # #-------------------------
# # 
# # # # vpr = dset.mc_inv_iso(outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20180919_150000_both',
# # # #                 numbrun=150000, nprocess=30, verbose=False, group=True)
# # # 
# # # vpr = dset.mc_inv_iso(use_ref=False, outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181202_150000_both_miller',
# # #                 numbrun=150000, nprocess=35, verbose=False, group=True, Ntotalruns=5)
# # # #-------------------------
# # # # read inversion results
# # # #-------------------------
# dset.read_inv(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both', avgqc=False)
# dset.read_inv(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181202_150000_both_miller', avgqc=False)
# 
# # # # # # 
# # # # # # #-------------------------
# # # # # # # interpolation/smoothing
# # # # # # #-------------------------
# # # # # dset.get_raytomo_mask(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_LD.h5', runid=2)
# # # # # OR
dset.get_hybrid_mask(inh5fname='/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20181101.h5', runid=0)
# 
dset.get_topo_arr(infname='/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
# # 
# # 
# # # dset.paraval_arrays(dtype='min')
# # # dset.construct_3d(dtype='min')
# # # dset.construct_3d(dtype='min', is_smooth=True)
# # # 
# dset.paraval_arrays(dtype='avg')
# # # dset.construct_3d(dtype='avg')
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


# vpr = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181202_150000_both_miller', lon=-142., lat=62.)
# vpr = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both', lon=-147., lat=66.5)

# vpr1 = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both', lon=-155., lat=68., thresh=0.1)
# vpr2 = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181202_150000_both_miller', lon=-155., lat=68., thresh=0.1)

# vpr = dset.generate_disp_vs_figs(datadir ='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both',\
#                 outdir='/home/leon/ALASKA_disp_vs')

# vpr = dset.generate_disp_vs_figs(datadir ='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181202_150000_both_miller',\
#                 outdir='/home/leon/ALASKA_disp_vs')

# dset.plot_paraval(pindex='min_misfit', is_smooth=False, cmap='jet', vmin=0.2, vmax=2.0, outfname='min_misfit.txt')
# dset.plot_paraval(pindex='avg_misfit', is_smooth=False, cmap='jet', vmin=0.2, vmax=2.0, outfname='avg_misfit.txt')
# dset.plot_paraval(pindex='fitratio', is_smooth=False, cmap='jet', vmin=0.3, vmax=1.0, outfname='fitratio.txt')

# dset.plot_paraval(pindex='moho', isthk=True, is_smooth=True, cmap='gist_ncar', vmin=20., vmax=60.0, clabel='Moho Depth (km)')

# import matplotlib.pyplot as plt
# import numpy as np
# cmap = plt.cm.RdYlBu
# # 
# # # Transparent colours
# # from matplotlib.colors import ListedColormap
# # 
# # colA = cmap(np.arange(cmap.N))
# # colA[:,-1] = 0.25 + 0.5 * np.linspace(-1.0, 1.0, cmap.N)**2.0
# # 
# # # Create new colormap
# # cmap = ListedColormap(colA)
# cmap = surfdbase.discrete_cmap(10, 'RdYlBu')
# # cmap = surfdbase.discrete_cmap(8, 'jet')
# dset.plot_paraval(pindex='moho', isthk=False, is_smooth=True, cmap=cmap, vmin=25., vmax=45.0, clabel='Crustal thickness (km)')
# dset.plot_paraval(pindex='moho', isthk=True,dtype='std', is_smooth=True, cmap=cmap, vmin=1., vmax=5.0, clabel='Uncertainties (km)')