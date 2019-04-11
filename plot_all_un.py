import surfdbase
import os
import numpy as np
dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190404_no_osci.h5')


dlst    = np.array([3., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130.])

# 
i       = 0
outdir  = '/home/leon/ALASKA_figs_un_no_osci'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
cmap    = surfdbase.discrete_cmap(8, 'hot_r')
for depth in dlst:
    vmin    = 0.
    vmax    = 0.16
    outfname= outdir+'/un_'+str(int(depth))+'km.jpg'
    print 'plotting: '+outfname
    dset.plot_paraval(pindex='vs_std_ray', depth=depth, depthavg=0., is_smooth=False, clabel='Uncertainties (km/s)', cmap=cmap,\
            title=str(int(depth))+' km', projection='lambert',  vmin=vmin, vmax=vmax, showfig=False, outimg = outfname)
    # break
    i   += 1
   