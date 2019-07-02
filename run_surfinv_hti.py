
import surfdbase
import copy


# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190624_useref.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190626.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190624_unamp_4.h5')
# dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190701_fourlay.h5')
dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/azi_inv_files/azi_20190701.h5')

#-------------------------
# before inversion
#-------------------------
# dset.read_eik_azi_aniso(inh5fname='/work1/leon/ALASKA_work/hdf5_files/azi_2deg_0.05_20190617.h5')


# -------------------------
# inversion
# -------------------------
# dset.compute_kernels_hti(misfit_thresh=5.)
# vpr = dset.compute_kernels_hti(misfit_thresh=5.)
# vpr = dset.compute_kernels_hti(outlon=209., outlat=63.1)
# vpr = dset.linear_inv_hti(outlon=-150.+360., outlat = 65., depth_mid_mantle=80.)
# vpr = dset.linear_inv_hti(outlon=-155.+360., outlat = 63.)
# vpr = dset.linear_inv_hti(misfit_thresh=10.)
# dset.linear_inv_hti(misfit_thresh=5.)
# # 
# # 
# dset.construct_hti_model()

# vpr = dset.linear_inv_hti(outlon=-150.+360., outlat = 65., depth_mid_mantle=80.)


# dset.linear_inv_hti(misfit_thresh=5., depth_mid_mantle=80.)
# dset.construct_hti_model_four_lay()

# 
# cmap = surfdbase.discrete_cmap(6, 'hot_r')
# dset.plot_hti(scaled=True, normv=5.,factor=5, gindex=2, datatype='misfit', ampref=.3, plot_data=True, plot_axis=False, cmap=cmap, vmin=0.5, vmax=2.0)

# dset.plot_hti_vel(depth=100., gindex=2, scaled=True, factor=5, ampref=2., normv=1., vmin=4.1, vmax=4.6)
# dset.plot_hti_vel(depth=20., gindex=1, scaled=True, factor=5, ampref=2., normv=1., vmin=3.4, vmax=3.8)
# dset.plot_hti_vel(depth=20., gindex=1, scaled=True, factor=5, ampref=2., normv=2., vmin=3.4, vmax=3.8, ticks=[3.4, 3.5, 3.6, 3.7, 3.8])

# vpr = dset.generate_disp_vs_figs(datadir ='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181105_150000_both',\
#                 outdir='/home/leon/ALASKA_disp_vs')

# vpr = dset.generate_disp_vs_figs(datadir ='/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20181202_150000_both_miller',\
#                 outdir='/home/leon/ALASKA_disp_vs')

# dset.plot_paraval(pindex='min_misfit', is_smooth=False, cmap='jet', vmin=0.2, vmax=2.0, outfname='min_misfit.txt')
# dset.plot_paraval(pindex='avg_misfit', is_smooth=False, cmap='hot', vmin=0.2, vmax=2.0, outfname='avg_misfit.txt')
# dset.plot_paraval(pindex='fitratio', is_smooth=False, cmap='jet', vmin=0.3, vmax=1.0, outfname='fitratio.txt')

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
# # # # dset.plot_paraval(pindex='avg_misfit', is_smooth=False, cmap=cmap, vmin=0.0, vmax=2.0, outfname='avg_misfit.txt', clabel='Misfit')
# dset.plot_paraval(pindex='moho', isthk=False, is_smooth=True, cmap=cmap, vmin=25., vmax=45.0, clabel='Crustal thickness (km)')
# dset.plot_paraval(pindex='moho', isthk=True, dtype='std', is_smooth=True, cmap=cmap, vmin=0., vmax=10.0, clabel='Uncertainties of Crustal Thickness (km)')
# # # dset.plot_paraval(pindex='vs_std', is_smooth=False, depth=10., depthavg = 0.)
# # 
# dset.plot_crust1( infname='crsthk.xyz', vmin=25., vmax=45., clabel='Crustal thickness (km)', cmap=cmap)
# dset.plot_miller_moho_finer(vmin=25., vmax=45., clabel='Crustal thickness (km)', cmap=cmap)
# dset.convert_to_vts(outdir='outvts', depthavg=-1.)


# # # 
# # # import copy
# # # vpr     = dset.compute_kernels_hti(misfit_thresh=5., outlon=-155.+360., outlat = 61.3)
# # # vpr2    = copy.deepcopy(vpr)
# # # 
# # # vpr.compute_reference_vti(wtype='ray')
# # # vpr2.compute_reference_vti_2(wtype='ray')
# # # v0 = vpr.data.dispR.pvelref.copy()
# # # 
# # # vpr.eigkR.bveti[-10:]    *= 1.02
# # # vpr2.eigkR.bveti[-10:]   *= 1.02
# # # vpr.model.vsv[-10:]    *= 1.02
# # # vpr2.model.vsv[-10:]    *= 1.02
# # # 
# # # v1 = v0 + vpr.eigkR.eti_perturb_vel()
# # # v2 = v0 + vpr2.eigkR.eti_perturb_vel()
# # # 
# # # vpr.model.vel2love()
# # # vpr.compute_reference_vti(wtype='ray')
# # # v3 = vpr.data.dispR.pvelref.copy()

