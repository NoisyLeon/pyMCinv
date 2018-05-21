
import invdbase

# dset = invdbase.invASDF('/scratch/summit/life9360/ALASKA_work/mc_inv_files/inversion_alaska.h5')
# dset = invdbase.invhdf5('./inversion_alaska_surf.h5')
dset = invdbase.invhdf5('/work3/leon/mc_inv_files/inversion_alaska_surf.h5')
# dset = invdbase.invASDF('../../inversion_alaska.h5')
# dset.read_ref_dbase(inasdfname='/scratch/summit/life9360/ALASKA_work/ASDF_data/ref_Alaska.h5')
# dset.read_raytomo_dbase(inh5fname='/work3/leon/ray_tomo_Alaska_20180410.h5', runid=2, Tmin=8., Tmax=50.)
# 
# dset.read_crust_thickness()
# dset.read_sediment_thickness()
# dset.read_CU_model()
# dset.read_etopo()
# vpr = dset.mc_inv_iso(instafname='ref_log')
# dset.mc_inv_iso(instafname='ref_log_Miller', outdir='/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_results_Miller',
#                 numbrun=150000, nprocess=10, verbose=True)
# 
dset.mc_inv_iso(outdir='/work3/leon/mc_inv_files/mc_alaska_surf',
                numbrun=45000, nprocess=12, verbose=False)

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
