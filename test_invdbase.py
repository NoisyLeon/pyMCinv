
import invdbase

dset = invdbase.invASDF('/scratch/summit/life9360/ALASKA_work/mc_inv_files/inversion_alaska.h5')
# dset.read_ref_dbase(inasdfname='/scratch/summit/life9360/ALASKA_work/ASDF_data/ref_Alaska.h5')
dset.read_raytomo_dbase(inh5fname='/scratch/summit/life9360/ALASKA_work/hdf5_files/ray_tomo_Alaska_20180410.h5', runid=2)
