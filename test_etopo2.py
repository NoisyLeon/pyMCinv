from netCDF4 import Dataset
from matplotlib.colors import LightSource
etopo2_link = 'https://www.ngdc.noaa.gov/mgg/global/relief/ETOPO2/ETOPO2v2-2006/ETOPO2v2g/netCDF/ETOPO2v2g_f4_netCDF.zip'
etopodata   = Dataset('../ETOPO2v2g_f4.nc')
etopo       = etopodata.variables['z'][:]
lons        = etopodata.variables['x'][:]
lats        = etopodata.variables['y'][:]
# ls          = LightSource(azdeg=315, altdeg=45)
# nx          = int((m.xmax-m.xmin)/40000.)+1; ny = int((m.ymax-m.ymin)/40000.)+1
# etopo,lons  = shiftgrid(180.,etopo,lons,start=False)
# topodat,x,y = m.transform_scalar(etopo,lons,lats,nx,ny,returnxy=True)
# ny, nx      = etopo.shape
# topodat,xtopo,ytopo = m.transform_scalar(etopo,lons,lats,nx, ny, returnxy=True)
# m.imshow(ls.hillshade(topodat, vert_exag=1., dx=1., dy=1.), cmap='gray')
# mycm1=pycpt.load.gmtColormap('/projects/life9360/station_map/etopo1.cpt')
# mycm2=pycpt.load.gmtColormap('/projects/life9360/station_map/bathy1.cpt')
# mycm2.set_over('w',0)
