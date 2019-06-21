import surfdbase
import os
import numpy as np
import matplotlib.pyplot as plt

dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190404_no_osci.h5')


# dlst    = np.array([3., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130.])
# dlst    = np.array([3., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100.])
# dlst    = np.append(3., np.arange(24.)*5. + 5.)
# 
# # 
# i       = 0
# outdir  = '/home/leon/ALASKA_figs_un_no_osci'
# if not os.path.isdir(outdir):
#     os.makedirs(outdir)
# cmap    = surfdbase.discrete_cmap(8, 'hot_r')
# stdarr  = np.zeros(dlst.size)
# for depth in dlst:
#     vmin    = 0.
#     vmax    = 0.16
#     outfname= outdir+'/un_'+str(int(depth))+'km.jpg'
#     print 'plotting: '+outfname
#     data, data_smooth\
#                         = dset.get_smooth_paraval(pindex='vs_std_ray', depth=depth, depthavg=0.)
#     stdarr[i] = data.mean()
#     # dset.plot_paraval(pindex='vs_std_ray', depth=depth, depthavg=0., is_smooth=True, clabel='Uncertainties (km/s)', cmap=cmap,\
#     #         title=str(int(depth))+' km', projection='lambert',  vmin=vmin, vmax=vmax, showfig=False, outimg = outfname)
#     # break
#     i   += 1
# 
# ax  = plt.subplot()
# plt.plot(dlst, stdarr*1000., 'o', ms=15, label='observed')
# ax.tick_params(axis='x', labelsize=40)
# ax.tick_params(axis='y', labelsize=40)
# plt.xlabel('Depth (km)', fontsize=60)
# plt.ylabel('Standard deviation (m/s)', fontsize=60)
# # plt.legend(fontsize=30)
# plt.show()

mask        = dset.attrs['mask_inv']
data, data_smooth\
                        = dset.get_smooth_paraval(pindex='moho', dtype='std')
print data[np.logical_not(mask)].mean()

dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190327_no_ocsi_crust_15_mantle_10_vti_gr.h5')

# data, data_smooth\
#                         = dset.get_smooth_paraval(pindex='moho', dtype='std')

un, un_smooth\
                        = dset.get_smooth_paraval(pindex=-2, dtype='std', itype='vti', )
print un[np.logical_not(mask)].mean()


un, un_smooth\
                        = dset.get_smooth_paraval(pindex=-1, dtype='std', itype='vti', )
print un[np.logical_not(mask)].mean()