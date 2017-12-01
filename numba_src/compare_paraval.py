import vprofile
import modparam
import vmodel
import matplotlib.pyplot as plt

outdir = './synthetic_inv'

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='./synthetic_inv/disp_lov.txt', wtype = 'l')
vpr.readdisp(infname='./synthetic_inv/disp_ray.txt', wtype = 'r')
vpr.readaziamp(infname='./synthetic_inv/aziamp.ray.txt', wtype = 'r')
vpr.readaziphi(infname='./synthetic_inv/aziphi.ray.txt', wtype = 'r')
vpr.readmod(infname='mod_-112.0.36.0.mod', mtype='tti')
vpr.getpara(mtype='tti')

vpr.read_paraval(infname='./synthetic_inv/paraval.txt')


vpr.compute_tcps(wtype='ray')
vpr.compute_tcps(wtype='love')
vpr.perturb_from_kernel(wtype='ray')
vpr.perturb_from_kernel(wtype='love')



fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
ax = axs[0,0]
ax.errorbar(vpr.indata.dispR.pper, vpr.indata.dispR.pvelo, yerr=vpr.indata.dispR.stdpvelo, fmt='o-')
ax.plot(vpr.indata.dispR.pper, vpr.indata.dispR.pvelp, 'x')
ax.set_title('Rayleigh wave dispersion')

# With 4 subplots, reduce the number of axis ticks to avoid crowding.
ax.locator_params(nbins=4)

ax = axs[0,1]
ax.errorbar(vpr.indata.dispR.pper, vpr.indata.dispR.pampo, yerr=vpr.indata.dispR.stdpampo, fmt='o-')
ax.plot(vpr.indata.dispR.pper, vpr.indata.dispR.pampp, 'x')
ax.set_title('Rayleigh wave azimuthal amplitude')

ax = axs[1,0]
ax.errorbar(vpr.indata.dispR.pper, vpr.indata.dispR.pphio, yerr=vpr.indata.dispR.stdpphio, fmt='o-')
ax.plot(vpr.indata.dispR.pper, vpr.indata.dispR.pphip, 'x')
ax.set_title('Rayleigh wave fast-axis azimuth')

ax = axs[1,1]
ax.errorbar(vpr.indata.dispL.pper, vpr.indata.dispL.pvelo, yerr=vpr.indata.dispL.stdpvelo, fmt='o-')
ax.plot(vpr.indata.dispL.pper, vpr.indata.dispL.pvelp, 'x')
ax.set_title('Love wave dispersion')

# fig.suptitle('Variable errorbars')

plt.show()
