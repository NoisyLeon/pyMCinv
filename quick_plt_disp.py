import surfdbase
import matplotlib.pyplot as plt

dset    = surfdbase.invhdf5('/work1/leon/ALASKA_work/mc_inv_files/inversion_alaska_surf_20190213.h5')


plt.figure()
ax      = plt.subplot()
        
lon     = -144.
lat     = 60.
disp_ph, disp_gr    = dset.get_disp(lon=lon, lat=lat)
plt.errorbar(disp_ph[0, :], disp_ph[1, :], yerr=disp_ph[2, :], fmt='o-',color='b', lw=3, label=str(lon)+' '+str(lat))
# plt.errorbar(disp_gr[0, :], disp_gr[1, :], yerr=disp_gr[2, :], color='b', lw=3, label=str(lon)+' '+str(lat))

lon     = -156.
lat     = 60.
disp_ph, disp_gr    = dset.get_disp(lon=lon, lat=lat)
plt.errorbar(disp_ph[0, :], disp_ph[1, :], yerr=disp_ph[2, :], fmt='o-.',color='r', lw=3, label=str(lon)+' '+str(lat))
# plt.errorbar(disp_gr[0, :], disp_gr[1, :], yerr=disp_gr[2, :], color='r', lw=3, label=str(lon)+' '+str(lat))

lon     = -152.
lat     = 60.
disp_ph, disp_gr    = dset.get_disp(lon=lon, lat=lat)
plt.errorbar(disp_ph[0, :], disp_ph[1, :], yerr=disp_ph[2, :], fmt='o--',color='k', lw=3, label=str(lon)+' '+str(lat))
# plt.errorbar(disp_gr[0, :], disp_gr[1, :], yerr=disp_gr[2, :], color='k', lw=3, label=str(lon)+' '+str(lat))

lon     = -144.
lat     = 62.
disp_ph, disp_gr    = dset.get_disp(lon=lon, lat=lat)
plt.errorbar(disp_ph[0, :], disp_ph[1, :], yerr=disp_ph[2, :], fmt='o-.',color='g', lw=3, label=str(lon)+' '+str(lat))
# plt.errorbar(disp_gr[0, :], disp_gr[1, :], yerr=disp_gr[2, :], color='g', lw=3, label=str(lon)+' '+str(lat))


ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.xlabel('Period (sec)', fontsize=30)
plt.ylabel('Phase elocity (km/sec)', fontsize=30)
plt.legend()
# plt.show()

plt.figure()
ax      = plt.subplot()
        
lon     = -144.
lat     = 60.
disp_ph, disp_gr    = dset.get_disp(lon=lon, lat=lat)
# plt.errorbar(disp_ph[0, :], disp_ph[1, :], yerr=disp_ph[2, :], color='b', lw=3, label=str(lon)+' '+str(lat))
plt.errorbar(disp_gr[0, :], disp_gr[1, :], yerr=disp_gr[2, :], fmt='o-', color='b', lw=3, label=str(lon)+' '+str(lat))

lon     = -156.
lat     = 60.
disp_ph, disp_gr    = dset.get_disp(lon=lon, lat=lat)
# plt.errorbar(disp_ph[0, :], disp_ph[1, :], yerr=disp_ph[2, :], color='r', lw=3, label=str(lon)+' '+str(lat))
plt.errorbar(disp_gr[0, :], disp_gr[1, :], yerr=disp_gr[2, :], fmt='o--',color='r', lw=3, label=str(lon)+' '+str(lat))

lon     = -152.
lat     = 60.
disp_ph, disp_gr    = dset.get_disp(lon=lon, lat=lat)
# plt.errorbar(disp_ph[0, :], disp_ph[1, :], yerr=disp_ph[2, :], color='k', lw=3, label=str(lon)+' '+str(lat))
plt.errorbar(disp_gr[0, :], disp_gr[1, :], yerr=disp_gr[2, :], fmt='o-.',color='k', lw=3, label=str(lon)+' '+str(lat))

lon     = -144.
lat     = 62.
disp_ph, disp_gr    = dset.get_disp(lon=lon, lat=lat)
# plt.errorbar(disp_ph[0, :], disp_ph[1, :], yerr=disp_ph[2, :], color='g', lw=3, label=str(lon)+' '+str(lat))
plt.errorbar(disp_gr[0, :], disp_gr[1, :], yerr=disp_gr[2, :], fmt='o-.',color='g', lw=3, label=str(lon)+' '+str(lat))


ax.tick_params(axis='x', labelsize=20)
ax.tick_params(axis='y', labelsize=20)
plt.xlabel('Period (sec)', fontsize=30)
plt.ylabel('Group elocity (km/sec)', fontsize=30)
plt.legend()
plt.show()