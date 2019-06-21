import obspy
import numpy as np
evlons  = np.array([])
evlats  = np.array([])
zArr    = np.array([])
magarr  = np.array([])
outdir  = '/home/leon/outvts/slab2.0'

def read_slab_contour(depth, infname='/home/leon/Slab2Distribute_Mar2018/Slab2_CONTOURS/alu_slab2_dep_02.23.18_contours.in'):
    ctrlst  = []
    lonlst  = []
    latlst  = []
    with open(infname, 'rb') as fio:
        newctr  = False
        for line in fio.readlines():
            if line.split()[0] is '>':
                newctr  = True
                if len(lonlst) != 0:
                    ctrlst.append([lonlst, latlst])
                lonlst  = []
                latlst  = []
                z       = -float(line.split()[1])
                if z == depth:
                    skipflag    = False
                else:
                    skipflag    = True
                continue
            if skipflag:
                continue
            lonlst.append(float(line.split()[0]))
            latlst.append(float(line.split()[1]))
    return ctrlst

from tvtk.api import tvtk, write_data
from sympy.ntheory import primefactors
dlst    = np.array([20., 40., 60., 80., 100., 120.])
for depth in dlst:
    slb_ctrlst      = read_slab_contour(depth=depth)
    lons            = np.array([])
    lats            = np.array([])
    for slbctr in slb_ctrlst:
        lons1, lats1= np.array(slbctr[0])-360., np.array(slbctr[1])
        lons        = np.append(lons, lons1)
        lats        = np.append(lats, lats1)
        
    zArr        = np.ones(lons.size)*depth
    val         = np.ones(lons.size)*depth
    Rref        = 6371.
    # convert geographycal coordinate to spherichal coordinate
    theta       = (90.0 - lats)*np.pi/180.
    phi         = lons*np.pi/180.
    radius      = Rref - zArr
    # convert spherichal coordinate to 3D Cartesian coordinate
    x           = radius * np.sin(theta) * np.cos(phi)/Rref
    y           = radius * np.sin(theta) * np.sin(phi)/Rref
    z           = radius * np.cos(theta)/Rref
    
    least_prime = primefactors(val.size)[0]
    dims        = (val.size/least_prime, least_prime, 1)
    pts         = np.empty(z.shape + (3,), dtype=float)
    pts[..., 0] = x; pts[..., 1] = y; pts[..., 2] = z
    sgrid = tvtk.StructuredGrid(dimensions=dims, points=pts)
    sgrid.point_data.scalars = (val).ravel(order='F')
    sgrid.point_data.scalars.name = 'slab2_depth'
    outfname    = outdir+'/slab2_'+str(depth)+'.vts'
    write_data(sgrid, outfname)