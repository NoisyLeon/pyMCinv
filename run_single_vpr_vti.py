
# -*- coding: utf-8 -*-
import surfdbase
import copy
import matplotlib.pyplot as plt
import numpy as np

# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190501_no_osci_vti_sediment_15_crust_10_mantle_10.h5')
###
dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190501_no_osci_vti_sed_25_crt_10_mantle_0_col.h5')
# dset.read_hybridtomo_dbase(inh5fname='/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_Love_20190318.h5',\
#                            runid=0, semfactor=2., Tmin=8., Tmax=50., wtype='lov')


# vpr1 = dset.get_vpr_vti(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190501_150000_sed_25_crust_0_mantle_10_vti_col',\
#     lon=-155.+360., lat=69., thresh=0.5)
vpr = dset.get_vpr_vti(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190327_150000_crust_15_mantle_10_vti',\
    lon=-156.+360., lat=67.5, thresh=0.5)

# vpr2 = dset.get_vpr_vti(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190501_150000_sed_25_crust_0_mantle_10_vti_col',\
#     lon=-155.+360., lat=68.5, thresh=0.5)
# # 
# vpr = dset.get_vpr_vti(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190327_150000_crust_0_mantle_10_vti',\
#     lon=-155.+360., lat=68., thresh=0.5)

# 
# vpr1 = dset.mc_inv_vti(use_ref=True, outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190325_150000_vti',
#                 numbrun=150000, nprocess=35, verbose=False, group=False, outlon=-142., outlat = 60.)
# vpr = dset.mc_inv_vti(use_ref=True, outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190501_150000_vti',
#                 numbrun=150000, nprocess=35, verbose=False, group=False, outlon=-150., outlat = 68.)
# # 
# vpr.mc_joint_inv_vti_mp(outdir='./test_working_vti_sed_2', solver_type=1, numbrun=150000, nprocess=20, verbose=True)
    


# # 
# # # # # # #-------------------------
# # # # # # # read inversion results
# # # # # # #-------------------------
# dset.read_inv(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190322_150000_both_crust1_ocsi', avgqc=False)
# # # 
# 
# #-------------------------
# # interpolation/smoothing
# #-------------------------
# # # # dset.get_raytomo_mask(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_LD.h5', runid=2)
# # # # # # # OR
# dset.get_hybrid_mask(inh5fname='/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20190318.h5', runid=0)
# # # # # 
# dset.get_topo_arr(infname='/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
# # # # # # 
# # # # # # 
# # # # # # dset.paraval_arrays(dtype='min')
# # # # # # dset.construct_3d(dtype='min')
# # # # # # dset.construct_3d(dtype='min', is_smooth=True)
# # # # # # 
# dset.paraval_arrays(dtype='avg')
# # # dset.construct_3d(dtype='avg')
# dset.construct_3d(dtype='avg', is_smooth=True)

# # # 
# depth = 100.
# # # dset.plot_horizontal(depth=depth, dtype='min', is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
# # #              geopolygons=None, vmin=4.2, vmax=4.6, showfig=True)
# # # # 
# dset.plot_horizontal(depth=depth, dtype='avg', is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
#              geopolygons=None, vmin=None, vmax=None, showfig=True)
# # # 
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
# import obspy
# cat     = obspy.read_events('alaska_events.xml')
# dset.plot_vertical_rel(plottype=1, lon1=-160+360, lon2=-150+360, lat1=62, lat2=58, maxdepth=120.,\
#                         dtype='avg', is_smooth=True, incat = cat, vmin2=-5., vmax2=5., vs_mantle=4.35)
# dset.plot_vertical_rel(plottype=1, lon1=-151+360, lon2=-150+360, lat1=69, lat2=58, maxdepth=120.,\
#                        dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)
# dset.plot_vertical_rel(plottype=1, lon1=-130+360, lon2=-150+360, lat1=68, lat2=58, maxdepth=120.,\
#                        dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)




# vpr = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181202_150000_both_miller', lon=-142., lat=62.)
# vpr = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both', lon=-147., lat=66.5)

# vpr1 = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both', lon=-155., lat=68., thresh=0.1)
# vpr2 = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181202_150000_both_miller', lon=-155., lat=68., thresh=0.1)

# vpr = dset.generate_disp_vs_figs(datadir ='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both',\
#                 outdir='/home/leon/ALASKA_disp_vs')

# vpr = dset.generate_disp_vs_figs(datadir ='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181202_150000_both_miller',\
#                 outdir='/home/leon/ALASKA_disp_vs')

# dset.plot_paraval(pindex='min_misfit', is_smooth=False, cmap='jet', vmin=0.2, vmax=2.0, outfname='min_misfit.txt')
# dset.plot_paraval(pindex='avg_misfit', is_smooth=False, cmap='hot', vmin=0.2, vmax=2.0, outfname='avg_misfit.txt')
# dset.plot_paraval(pindex='fitratio', is_smooth=False, cmap='jet', vmin=0.3, vmax=1.0, outfname='fitratio.txt')

# dset.plot_paraval(pindex='moho', isthk=True, is_smooth=True, cmap='gist_ncar', vmin=20., vmax=60.0, clabel='Moho Depth (km)')
# 
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
# cmap = ListedColormap(colA)
# cmap = surfdbase.discrete_cmap(10, 'RdYlBu')
# # # cmap = surfdbase.discrete_cmap(8, 'hot_r')
# # # dset.plot_paraval(pindex='avg_misfit', is_smooth=False, cmap=cmap, vmin=0.0, vmax=2.0, outfname='avg_misfit.txt', clabel='Misfit')
# # dset.plot_paraval(pindex='moho', isthk=False, is_smooth=True, cmap=cmap, vmin=25., vmax=45.0, clabel='Crustal thickness (km)')
# # dset.plot_paraval(pindex='moho', isthk=True, dtype='std', is_smooth=True, cmap=cmap, vmin=0., vmax=10.0, clabel='Uncertainties of Crustal Thickness (km)')
# # # dset.plot_paraval(pindex='vs_std', is_smooth=False, depth=10., depthavg = 0.)
# # 
# dset.plot_crust1( infname='crsthk.xyz', vmin=25., vmax=45., clabel='Crustal thickness (km)', cmap=cmap)