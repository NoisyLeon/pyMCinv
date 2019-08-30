
import surfdbase
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180919_3d.h5')
import copy

# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_final.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_ready4post.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190322_osci.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190404_no_osci.h5')
dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190320_no_ocsi.h5')

# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190701_no_ocsi.h5')


#-------------------------
# before inversion
#-------------------------
# dset.read_raytomo_dbase(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_LD.h5', runid=2, Tmin=8., Tmax=50.)
# # OR
# dset.read_hybridtomo_dbase(inh5fname='/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20190318.h5', runid=0, semfactor=2., Tmin=8., Tmax=85.)
# dset.read_etopo(infname='/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
# # # # dset.read_crust_thickness(replace_moho=10., 
# # #     # infname_refine='/home/leon/miller_alaskamoho_srl2018-1.2.2/miller_alaskamoho_srl2018/Models/AlaskaMoHiErrs-AlaskaMohoFineGrid.npz')
# dset.read_crust_thickness(replace_moho=None)
# dset.read_sediment_thickness()
# # # # 
# # # # # # # 
# # # # # # group
# dset.read_raytomo_dbase_group(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_bks/ray_tomo_Alaska_20190318_gr.h5', runid=1, Tmin=8., Tmax=50.)
# # # #
# # # 
# # # # -------------------------
# # # # inversion
# # # # -------------------------
# # # 
# dset.mc_inv_iso(use_ref=False, outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190701_150000_both_crust1_no_ocsi',
#                 numbrun=150000, nprocess=30, verbose=False, group=True, Ntotalruns=2)
# # # 
# # # # # # # #-------------------------
# # # # # # # # read inversion results
# # # # # # # #-------------------------
# dset.read_inv(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190701_150000_both_crust1_no_ocsi', avgqc=False)
# # # 
# 
# #-------------------------
# # interpolation/smoothing
# #-------------------------
# # # # dset.get_raytomo_mask(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_LD.h5', runid=2)
# # # # # # # OR
# dset.get_hybrid_mask(inh5fname='/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20190318.h5', runid=0)
# # # # # # # 
# dset.get_topo_arr(infname='/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
# # # # # # # # 
# # # # # # # # 
# # # # # # # # 
# dset.paraval_arrays(dtype='avg')
# dset.construct_3d(dtype='avg')
# dset.construct_3d(dtype='avg', is_smooth=True)
# 
# # # # 
# depth = 100.
# dset.plot_horizontal(depth=depth, dtype='sem', is_smooth=False, shpfx=None, clabel='Uncertainties (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
#              geopolygons=None,  showfig=True)
# # dset.plot_horizontal(depth=depth, dtype='avg', is_smooth=False, shpfx=None, clabel='Vs (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
# #              geopolygons=None, vmin=4.1, vmax=4.6, showfig=True)
# # 
# dset.plot_horizontal_zoomin(depth=depth, dtype='avg', is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
#              geopolygons=None, vmin=4.1, vmax=4.6, showfig=True)
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
# 
# dset.plot_vertical_rel(lon1=-170+360, lon2=-145+360, lat1=64, lat2=58, maxdepth=100., dtype='avg', is_smooth=True)
# import obspy
# cat     = obspy.read_events('alaska_events.xml')
# dset.plot_vertical_rel(plottype=1, lon1=-160+360, lon2=-150+360, lat1=62, lat2=58, maxdepth=120.,\
#                         dtype='avg', is_smooth=True, incat = cat, vmin2=-5., vmax2=5., vs_mantle=4.35)
# dset.plot_vertical_rel(plottype=1, lon1=-151+360, lon2=-150+360, lat1=69, lat2=58, maxdepth=120.,\
#                        dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)
# dset.plot_vertical_rel(plottype=1, lon1=-130+360, lon2=-150+360, lat1=68, lat2=58, maxdepth=120.,\
#                        dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)
# 
# dset.plot_vertical_rel(plottype=1, lon1=-130+360, lon2=-150+360, lat1=68, lat2=58, maxdepth=120.,\
#                        dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)

## Jademec

# 
# 
# # # dset.plot_vertical_rel_2(plottype=1, lon1=-157.5+360, lon2=-146+360, lat1=62, lat2=59, maxdepth=120.,\
# # #                        dtype='avg', is_smooth=True, incat = None, vmin2=4.2, vmax2=4.5, vs_mantle=4.35)
# 
# dset.plot_vertical_rel(plottype=1, lon1=-150+360, lon2=-150.+360, lat1=58, lat2=70, maxdepth=120.,\
#                        dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)

# dset.plot_vertical_rel(plottype=1, lon1=-146+360, lon2=-159+360, lat1=59, lat2=62, maxdepth=120.,\
#                        dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)

# dset.plot_vertical_rel(plottype=1, lon2=-146+360, lon1=-159+360, lat2=59, lat1=62, maxdepth=120.,\
#                        dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)

# dset.plot_vertical_rel(plottype=1, lon1=-145+360, lon2=-142+360, lat1=59, lat2=64, maxdepth=120.,\
#                        dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)

# dset.plot_vertical_rel_2(plottype=1, lon1=-153.+360, lon2=-153+360, lat1=68., lat2=65., maxdepth=200.,\
                       # dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)


# dset.plot_vertical_rel_2(plottype=1, lon1=-156.+360, lon2=-143+360, lat1=64., lat2=60., maxdepth=120.,\
#                        dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)

# dset.plot_vertical_rel(plottype=0, lon1=-160+360, lon2=-136+360, lat1=60, lat2=60.5, maxdepth=120.,\
#               dtype='avg', is_smooth=True, incat = -1, vmin2=-5., vmax2=5., vs_mantle=4.35)


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
# # dset.plot_paraval(pindex='moho', isthk=False, is_smooth=True, cmap=cmap, vmin=25., vmax=45.0, clabel='Crustal thickness (km)')
# # colA = cmap(np.arange(cmap.N))
# # colA[:,-1] = 0.25 + 0.5 * np.linspace(-1.0, 1.0, cmap.N)**2.0
# # 
# # # Create new colormap
# cmap = ListedColormap(colA)
# cmap = surfdbase.discrete_cmap(10, 'RdYlBu')
# cmap = surfdbase.discrete_cmap(8, 'jet_r')
# # # # dset.plot_paraval(pindex='avg_misfit', is_smooth=False, cmap=cmap, vmin=0.0, vmax=2.0, outfname='avg_misfit.txt', clabel='Misfit')
# dset.plot_paraval(pindex='moho', isthk=False, is_smooth=True, cmap=cmap, vmin=25., vmax=45.0, clabel='Crustal thickness (km)')
# dset.plot_paraval(pindex='moho', isthk=True, dtype='std', is_smooth=True, cmap=cmap, vmin=0., vmax=10.0, clabel='Uncertainties of Crustal Thickness (km)')

# cmap = surfdbase.discrete_cmap(5, 'jet_r')
# dset.plot_paraval(pindex='rel_moho_std', isthk=True, dtype='std', is_smooth=True, cmap=cmap, vmin=0., vmax=0.25, clabel='Uncertainties of Crustal Thickness (km)')
# dset.plot_paraval(pindex='vs_std_ray', is_smooth=True, depth=100., depthavg = 3., cmap=cmap, vmin=0., vmax=0.16)
# # 
# dset.plot_crust1( infname='crsthk.xyz', vmin=25., vmax=45., clabel='Crustal thickness (km)', cmap=cmap)
# dset.plot_miller_moho_finer(vmin=25., vmax=45., clabel='Crustal thickness (km)', cmap=cmap)
# dset.convert_to_vts(outdir='outvts', depthavg=-1.)