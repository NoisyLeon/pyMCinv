
import surfdbase
import copy
import numpy as np


# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190711_psi1_psi2_crt_man_twolay.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190711_psi1_psi2_crt_man_twolay_ori_unamp.h5')

# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190711_psi1_psi2_ucrt_man_twolay_15km_ori_unamp.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190711_psi1_psi2_lcrt_man_twolay_ori_unamp.h5')


# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190711_psi1_psi2_ucrt_man_twolay.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190711_psi1_psi2_lcrt_man_twolay.h5')

# for SKS comparison
dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190711_psi1_psi2_ucrt_man_twolay_15km.h5')


# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190711_psi1_psi2.h5')

# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190715_padded_ucrt_man_twolay_15km.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190715_padded_crt_man_twolay.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190715_padded_lcrt_man_twolay.h5')



# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/before_20190710/azi_20190705_twolayer_moho.h5')

#-------------------------
# before inversion
#-------------------------
# dset.read_eik_azi_aniso(inh5fname='/work1/leon/ALASKA_work/hdf5_files/azi_2deg_0.05_20190617.h5')
# dset.read_eik_azi_aniso_2(inh5fname='/work1/leon/ALASKA_work/hdf5_files/azi_2deg_0.05_20190617_psi1.h5')
# 
# # -------------------------
# # computing kernels
# # -------------------------
# # dset.compute_kernels_hti(misfit_thresh=5.)
# 
# #-------------------------
# # LAB
# #-------------------------
# # dset.read_LAB()
# # dset.read_LAB_interp(extrapolate=True)
# dset.construct_LAB_miller()
# # # # # # # # # # # # # # dset.construct_dvs()
# # # # # # # # # # # # # # dset.construct_slab_edge()
# # # 
# # # # -------------------------
# # # # inversion
# # # # -------------------------
# # # 
# # # # two layer
# # # # -------------------------
# # dset.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=70., imoho=True, ilab=False)
# # # vpr = dset.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=70., imoho=True, ilab=False, outlon=-153.+360., outlat=66.1)
# # # # # # # # # # 
# d2d = np.zeros((2,2))
# # # # # upper crust
# # d2d[0, 0] = -1
# # d2d[0, 1] = 15.
# # d2d[1, 0] = -2
# # d2d[1, 1] = -3
# # # # # # # # # lower crust
# d2d[0, 0] = 15.
# d2d[0, 1] = -2.
# d2d[1, 0] = -2.
# d2d[1, 1] = -3.
# dset.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=70., imoho=True, ilab=False, depth2d=d2d)
# # # # # # # # # # # # dset.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=50., imoho=True, ilab=False, noasth=True)
# # # # # # # # # # # 
# # # # # # # # # # # v = dset.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=60., imoho=False, ilab=True)
# # # # # # # # # # # # 
# dset.construct_hti_model()
# # # # # # # # -------------------------
# # # # # # # 
# # # # # # # 
# # # # # # # 
# # # # # # # # 
# # # # # # # # # d = dset.plot_hti_diff_misfit(inh5fname='/work1/leon/ALASKA_work/azi_inv_files/azi_20190701_fourlay.h5')
# # # # # # # # # 
#cmap = surfdbase.discrete_cmap(6, 'hot_r')
#dset.plot_hti(gindex=-1, datatype='misfit', plot_data=True, plot_axis=False, cmap=cmap, vmin=0.5, vmax=2.0, ticks=[0.5, 1., 1.5, 2.])

#dset.plot_hti(gindex=-1, datatype='amp_misfit', plot_data=True, plot_axis=False, cmap=cmap, vmin=0.5, vmax=2.0, ticks=[0.5, 1., 1.5, 2.])
#dset.plot_hti(gindex=-1, datatype='psi_misfit', plot_data=True, plot_axis=False, cmap=cmap, vmin=0.5, vmax=2.0, ticks=[0.5, 1., 1.5, 2.])

# dset.plot_hti_vel(depth=100., gindex=2, scaled=True, factor=5, ampref=2., normv=1., vmin=4.1, vmax=4.6)
# dset.plot_hti_vel(depth=20., gindex=1, scaled=True, factor=5, ampref=2., normv=1., vmin=3.4, vmax=3.8)
# dset.plot_hti_vel(depth=20., gindex=0, scaled=True, factor=5, ampref=2., normv=1., vmin=3.4, vmax=3.8, ticks=[3.4, 3.5, 3.6, 3.7, 3.8])

# dset.plot_hti_vel(depth=60., gindex=2, scaled=True, factor=5, ampref=2., normv=1., vmin=4.15, vmax=4.55)

# dset.plot_hti_vel(depth=100., gindex=1, scaled=True, factor=5, ampref=1., normv=1., vmin=4.1, vmax=4.6)

# dset.plot_hti_vel(depth=10., gindex=0, scaled=True, factor=5, ampref=3., normv=1., vmin=3., vmax=3.7)

##
# LAB
#

# dset.plot_hti(gindex=-1, datatype='labarr', plot_data=True, plot_axis=False)
# dset.plot_paraval(pindex='moho', isthk=True, is_smooth=True, cmap='gist_ncar', vmin=20., vmax=60.0, clabel='Moho Depth (km)')
# 
# import matplotlib.pyplot as plt
# import numpy as np
# cmap = plt.cm.RdYlBu
# # 
# # # Transparent colours
# # from matplotlib.colors import ListedColormap
# # 
# # colA = cmap(np.arange(cmap.N))
# # colA[:,-1] = 0.25 + 0.5 * np.linspace(-1.0, 1.0, cmap.N)**2.0
# # 
# # # Create new colormap
# cmap = ListedColormap(colA)
# cmap = surfdbase.discrete_cmap(10, 'RdYlBu')
# cmap = surfdbase.discrete_cmap(10, 'hot_r')

# dset.plot_hti_doublelay(depth=100., vmin=4.1, vmax=4.6, plot_data=False, gindex=1)

