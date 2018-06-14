
import surfdbase

dset = surfdbase.invhdf5('/scratch/summit/life9360/ALASKA_work/mc_inv_files/inversion_alaska_surf.h5')

# dset.paraval_arrays()
# dset = surfdbase.invhdf5('./inversion_alaska_surf.h5')
# dset = surfdbase.invhdf5('/work3/leon/mc_inv_files/inversion_alaska_surf.h5')
# dset = surfdbase.invASDF('../../inversion_alaska.h5')
# dset.read_ref_dbase(inasdfname='/scratch/summit/life9360/ALASKA_work/ASDF_data/ref_Alaska.h5')
# dset.read_raytomo_dbase(inh5fname='/work3/leon/ray_tomo_Alaska_20180410.h5', runid=2, Tmin=8., Tmax=50.)
# 
# dset.read_crust_thickness()
# dset.read_sediment_thickness()
# dset.read_CU_model()
# dset.read_etopo()
# vpr = dset.mc_inv_iso(instafname='ref_log')
# 
# dset.mc_inv_iso(outdir='/work3/leon/mc_inv_files/mc_alaska_surf',
#                 numbrun=45000, nprocess=12, verbose=False)

# dset.mc_inv_iso(instafname='ref_log', outdir='/home/lili/new_mc_results_Miller',
#                 numbrun=150000, nprocess=4, verbose=True)
# dset.mc_inv_iso_mp(instafname='ref_log', outdir='/work3/leon/mc_inv_files/mc_results_100000', nprocess=10, subsize=50, numbrun=100000)

# vpr, vsdata = dset.mc_inv_iso()
# import matplotlib.pyplot as plt
# 
# plt.plot(vsdata[:, 0], vsdata[:, 1])
# vpr.update_mod()
# vpr.get_vmodel()
# plt.plot(vpr.model.zArr, vpr.model.VsvArr)
# plt.show()

# dset.read_inv(datadir='/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_alaska_surf')
# dset.plot_paraval(pindex=-2, isthk=True, dtype='avg')

dset.paraval_arrays(dtype='avg', sigma=1)
v=dset.construct_3d(dtype='avg')
# dset.construct_3d(dtype='avg', is_smooth=True)

# depth = 100.
# dset.plot_horizontal(depth=depth, dtype='avg', is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
#              geopolygons=None, vmin=4.2, vmax=4.7, showfig=True)
# 
# dset.plot_horizontal(depth=depth, dtype='avg', is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
             # geopolygons=None, vmin=None, vmax=None, showfig=True)

# dset.plot_horizontal(depth=10., dtype='avg', is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title='4 km', projection='lambert', hillshade=False,\
             # geopolygons=None, vmin=None, vmax=None, showfig=True)
# dset.plot_vertical_rel(lon1=-165+360, lon2=-150+360, lat1=65, lat2=55, maxdepth=100., dtype='avg', is_smooth=True)

