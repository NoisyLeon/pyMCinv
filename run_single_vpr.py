
import surfdbase
import copy

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

#-------------------------
# inversion
#-------------------------

# vpr = dset.mc_inv_iso(outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20180919_150000_both',
#                 numbrun=150000, nprocess=30, verbose=False, group=True)

vpr = dset.mc_inv_iso(use_ref=False, outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both',
                numbrun=150000, nprocess=35, verbose=False, group=True, outlon=-148., outlat = 66.)
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
vpr.mc_joint_inv_iso(outdir='./workingdir_prior', dispdtype='both', pfx='PH', wdisp=-1., verbose=True, numbrun=15000, step4uwalk=15)


