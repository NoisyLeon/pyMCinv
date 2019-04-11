import mcpost_vti
import numpy as np

# vpr = mcpost_vti.postvpr(thresh=0.5, factor=1., stdfactor=2.)
# vpr.read_inv_data('./test_working_vti/mc_inv.MC.npz')
# vpr.read_data('./test_working_vti/MC')
# vpr.get_vmodel()

# vpr = mcpost_vti.postvpr(thresh=0.5, factor=1., stdfactor=2.)
# vpr.read_inv_data('/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190327_150000_crust_15_mantle_10_vti/mc_inv.210.0_65.0.npz')
# vpr.read_data('/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190327_150000_crust_15_mantle_10_vti/210.0_65.0')
# # vpr.code = '206.0_64.0'
# vpr.get_vmodel()

vpr = mcpost_vti.postvpr(thresh=0.5, factor=1., stdfactor=2.)
vpr.read_inv_data('/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190327_150000_crust_0_mantle_10_vti/mc_inv.196.0_64.0.npz')
vpr.read_data('/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20190327_150000_crust_0_mantle_10_vti/196.0_64.0')
# vpr.code = '206.0_64.0'
vpr.get_vmodel()