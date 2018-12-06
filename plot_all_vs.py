import surfdbase
import os
import numpy as np
dset = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20181202.h5')


dlst    = np.array([3., 10., 20., 30., 40., 50., 60., 70., 80., 90., 100., 110., 120., 130.])
vlst    = [[2.5, 3.5], [3.0, 3.6], [3.4, 3.8], [3.7, 4.2], [3.8, 4.5], [4.1, 4.5], [4.2, 4.5]]
# 
i       = 0
outdir  = '/home/leon/ALASKA_figs_vs_new'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
for depth in dlst:
    if i <= 6:
        v   = vlst[i]
    else:
        v   = vlst[-1]
    vmin    = v[0]
    vmax    = v[1]
    outfname= outdir+'/vs_'+str(int(depth))+'km.jpg'
    print 'plotting: '+outfname
    dset.plot_horizontal(depth=depth, dtype='avg', depthavg=3., is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
             geopolygons=None, vmin=vmin, vmax=vmax, showfig=False, outfname = outfname)
    i   += 1
    
import obspy
cat     = obspy.read_events('alaska_events.xml')
i       = 0
outdir  = '/home/leon/ALASKA_figs_vs_new'
if not os.path.isdir(outdir):
    os.makedirs(outdir)
for depth in dlst:
    if i <= 6:
        v   = vlst[i]
    else:
        v   = vlst[-1]
    vmin    = v[0]
    vmax    = v[1]
    outfname= outdir+'/vs_'+str(int(depth))+'km_events.jpg'
    print 'plotting: '+outfname
    dset.plot_horizontal(depth=depth, dtype='avg', depthavg=3., is_smooth=True, shpfx=None, clabel='Vs (km/s)', cmap='cv', title=str(int(depth))+' km', projection='lambert', hillshade=False,\
             geopolygons=None, vmin=vmin, vmax=vmax, showfig=False, outfname = outfname, plotevents=True, incat=cat)
    i   += 1