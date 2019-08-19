
import surfdbase
import copy




dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190705_fourlayer.h5')

#-------------------------
# before inversion
#-------------------------
# dset.read_eik_azi_aniso(inh5fname='/work1/leon/ALASKA_work/hdf5_files/azi_2deg_0.05_20190617.h5')

# -------------------------
# computing kernels
# -------------------------
# dset.compute_kernels_hti(misfit_thresh=5.)

#-------------------------
# LAB
#-------------------------
# dset.read_LAB()
dset.read_LAB_interp(extrapolate=True)
# # # # dset.construct_dvs()
# # # # dset.construct_slab_edge()
# # # 
# # # # -------------------------
# # # # inversion
# # # # -------------------------
# # # 
# # # # three layer
# # # # -------------------------
dset.linear_inv_hti_adaptive(misfit_thresh=5., labthresh=70., depth_mid_crust=15., imoho=True, ilab=True)
dset.construct_hti_model_four_lay()
# -------------------------



# 
# # d = dset.plot_hti_diff_misfit(inh5fname='/work1/leon/ALASKA_work/azi_inv_files/azi_20190701_fourlay.h5')
# # 
cmap = surfdbase.discrete_cmap(6, 'hot_r')
dset.plot_hti(gindex=-1, datatype='misfit', plot_data=True, plot_axis=False, cmap=cmap, vmin=0.5, vmax=2.0, ticks=[0.5, 1., 1.5, 2.])

# dset.plot_hti_vel(depth=100., gindex=2, scaled=True, factor=5, ampref=2., normv=1., vmin=4.1, vmax=4.6)
# dset.plot_hti_vel(depth=20., gindex=1, scaled=True, factor=5, ampref=2., normv=1., vmin=3.4, vmax=3.8)
# dset.plot_hti_vel(depth=20., gindex=1, scaled=True, factor=5, ampref=2., normv=2., vmin=3.4, vmax=3.8, ticks=[3.4, 3.5, 3.6, 3.7, 3.8])

# dset.plot_hti_vel(depth=60., gindex=2, scaled=True, factor=5, ampref=2., normv=1., vmin=4.15, vmax=4.55)

# dset.plot_hti_vel(depth=120., gindex=3, scaled=True, factor=5, ampref=2., normv=1., vmin=4.1, vmax=4.6)

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



