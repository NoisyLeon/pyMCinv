
import surfdbase
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180919_3d.h5')
import copy


dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190320_no_ocsi.h5')


#-------------------------
# before inversion
#-------------------------

dset.read_hybridtomo_dbase(inh5fname='/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20190318.h5', runid=0, semfactor=2., Tmin=8., Tmax=85.)
dset.read_etopo(infname='/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
dset.read_crust_thickness(replace_moho=None)
dset.read_sediment_thickness()
# 
# # 
# group
dset.read_raytomo_dbase_group(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_20190318_gr.h5', runid=1, Tmin=8., Tmax=50.)
# # #
# # 
# # # -------------------------
# # # inversion
# # # -------------------------
# # 
dset.mc_inv_iso(use_ref=False, outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190404_150000_both_crust1_no_ocsi',
                numbrun=150000, nprocess=30, verbose=False, group=True, Ntotalruns=2)
# # 
# # # # # # #-------------------------
# # # # # # # read inversion results
# # # # # # #-------------------------
dset.read_inv(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190404_150000_both_crust1_no_ocsi', avgqc=False)
# # # 
# 
#-------------------------
# interpolation/smoothing
#-------------------------

dset.get_hybrid_mask(inh5fname='/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20190318.h5', runid=0)
dset.get_topo_arr(infname='/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')

dset.paraval_arrays(dtype='avg')
dset.construct_3d(dtype='avg', is_smooth=True)

# # # 