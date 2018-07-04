import field2d_earth
import numpy as np

minlat  = 52.
maxlat  = 72.
minlon  = 188.
maxlon  = 238.


field       = field2d_earth.Field2d(minlon=minlon, maxlon=maxlon, dlon=0.5, minlat=minlat, maxlat=maxlat, dlat=0.5, period=10.)
field.read(fname='./crthick_min.txt')
lat0        = (minlat + maxlat)/2.
lon0        = (minlon + maxlon)/2.

workingdir  = './field_gauss_filtering'
field.interp_surface(workingdir=workingdir, outfname='crthick_gauss')
field.gauss_smoothing(workingdir=workingdir, outfname='crthick_gauss', width=100.)
field.fieldtype = 'crstthk'



