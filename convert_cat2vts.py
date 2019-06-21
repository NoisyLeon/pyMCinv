import obspy
import numpy as np
evlons  = np.array([])
evlats  = np.array([])
zArr    = np.array([])
magarr  = np.array([])
outdir  = '/home/leon/outvts'

cat     = obspy.read_events('alaska_events.xml')
# cat     = cat.filter('magnitude > 5.0')
for event in cat:
    event_id    = event.resource_id.id.split('=')[-1]
    porigin     = event.preferred_origin()
    pmag        = event.preferred_magnitude()
    magnitude   = pmag.mag
    Mtype       = pmag.magnitude_type
    otime       = porigin.time
    try:
        evlo        = porigin.longitude
        evla        = porigin.latitude
        evdp        = porigin.depth/1000.
    except:
        continue
    evlons      = np.append(evlons, evlo)
    evlats      = np.append(evlats, evla);
    zArr  = np.append(zArr, evdp)
    magarr  = np.append(magarr, magnitude)

from tvtk.api import tvtk, write_data
from sympy.ntheory import primefactors
Rref        = 6371.
# convert geographycal coordinate to spherichal coordinate
theta       = (90.0 - evlats)*np.pi/180.
phi         = evlons*np.pi/180.
radius      = Rref - zArr
# theta, phi, radius \
#             = np.meshgrid(theta, phi, radius, indexing='ij')
# convert spherichal coordinate to 3D Cartesian coordinate
x           = radius * np.sin(theta) * np.cos(phi)/Rref
y           = radius * np.sin(theta) * np.sin(phi)/Rref
z           = radius * np.cos(theta)/Rref

least_prime = primefactors(magarr.size)[0]
dims        = (magarr.size/least_prime, least_prime, 1)
pts         = np.empty(z.shape + (3,), dtype=float)
pts[..., 0] = x; pts[..., 1] = y; pts[..., 2] = z
sgrid = tvtk.StructuredGrid(dimensions=dims, points=pts)
sgrid.point_data.scalars = (magarr).ravel(order='F')
sgrid.point_data.scalars.name = 'Mw'
outfname    = outdir+'/earthquakes.vts'
write_data(sgrid, outfname)