import surfdbase

dsetvti = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190710_no_ocsi_crust_15_mantle_10_vti_gr_uppercrt.h5')

dsetvti.get_lov_data(fname = 'lov.txt', lon=-150., lat=65.)

dsethti = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190711_psi1_psi2_ucrt_man_twolay_15km.h5')

dsethti.get_azi_data(fname = 'ray.txt', lon=-150., lat=65.)

dsethti.get_refmod(fname='refmod.txt', lon=-150., lat=65.)
