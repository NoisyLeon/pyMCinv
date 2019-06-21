
import surfdbase
import copy

# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20181203_single.h5')

# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190404_no_osci.h5')
dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190320_no_ocsi.h5')

# #-------------------------
# # before inversion
# #-------------------------
# dset.read_raytomo_dbase(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_LD.h5', runid=2, Tmin=8., Tmax=50.)
# OR
# dset.read_hybridtomo_dbase(inh5fname='/work1/leon/ALASKA_work/hdf5_files/eikonal_hybrid_20181101.h5', runid=0, semfactor=2.)
# dset.read_etopo(infname='/home/leon/station_map/grd_dir/ETOPO2v2g_f4.nc')
# dset.read_crust_thickness(replace_moho=10., 
#     infname_refine='/home/leon/miller_alaskamoho_srl2018-1.2.2/miller_alaskamoho_srl2018/Models/AlaskaMoHiErrs-AlaskaMohoFineGrid.npz')
# dset.read_sediment_thickness()
# dset.read_CU_model()
# # # 
# # group
# dset.read_raytomo_dbase_group(inh5fname='/work1/leon/ALASKA_work/hdf5_files/ray_tomo_Alaska_20180823_gr.h5', runid=2, Tmin=8., Tmax=50.)
# # # 
#-------------------------
# inversion
#-------------------------

# vpr = dset.mc_inv_iso(outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20180919_150000_both',
#                 numbrun=150000, nprocess=30, verbose=False, group=True)
# 
# vpr = dset.mc_inv_iso(use_ref=False, outdir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190213_150000_both_crust1',
#                 numbrun=150000, nprocess=35, verbose=False, group=True, outlon=-144., outlat = 60.)
# # # # # # # # vpr_ph          = copy.deepcopy(vpr)
# # # # # # # # vpr_ph.data.dispR.isgroup  \
# # # # # # # #                 = False
# # # # # # # # 
# # # # # # # # vpr_gr          = copy.deepcopy(vpr)
# # # # # # # # vpr_gr.data.dispR.isphase  \
# # # # # # # #                 = False
# # # # # # # 
# # # vpr.mc_joint_inv_iso_mp(outdir='./workingdir_crust_miller', dispdtype='both', pfx='BOTH', wdisp=1., nprocess=5, verbose=True, numbrun=150000, step4uwalk=1500)
# vpr.mc_joint_inv_iso_mp(outdir='./workingdir_highVcheck', dispdtype='both', pfx='BOTH', wdisp=1., nprocess=15, verbose=True, numbrun=150000)
# vpr_gr.mc_joint_inv_iso_mp(outdir='./workingdir', dispdtype='both', pfx='GR', wdisp=1., nprocess=35, verbose=True, numbrun=150000)
# vpr.mc_joint_inv_iso(outdir='./workingdir_prior', dispdtype='both', pfx='PH', wdisp=-1., verbose=True, numbrun=15000, step4uwalk=15)


# vpr1 = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both', lon=-155., lat=68., thresh=0.1)
# 
# vpr = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190320_150000_both_crust1_no_ocsi', lon=-155.+360., lat=69., thresh=0.5)
vpr = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190320_150000_both_crust1_no_ocsi', lon=-156.+360., lat=67.5, thresh=0.5)
# vpr = dset.get_vpr(datadir='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190320_150000_both_crust1_no_ocsi', lon=-148.+360., lat=64., thresh=0.5)
vpr.get_ensemble_2()
vpr.run_prior_fwrd(overwrite = False)
vpr.prior_vpr.get_ensemble() 
vpr.get_vs_std()





