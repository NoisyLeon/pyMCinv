
import surfdbase4vts
import numpy as np
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20180919_3d.h5')
import copy

# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_final.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_ready4post.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190322_osci.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190404_no_osci.h5')
dset = surfdbase4vts.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190320_no_ocsi.h5')

# dlst    = np.array([3., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130.])
# for depth in dlst:
#     dset.convert_to_vts_slice(depth=depth)

# dset.convert_to_vts_slice(depth=3., factor=1.0005)

# dset.convert_to_vts_slice(depth=3., factor=1.0)

dset.convert_to_vts_slice(depth=3., factor=1.01, outdir='/home/leon/outvts/depth_slices_coarse')
