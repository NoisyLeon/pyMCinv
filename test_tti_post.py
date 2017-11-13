import vprofile
import modparam
import matplotlib.pyplot as plt

vpr = vprofile.vprofile1d()


# vpr.readdisp(infname='./disp_-112.2_36.4_lov.txt', wtype = 'l')
# vpr.readdisp(infname='./disp_-112.2_36.4_ray.txt', wtype = 'r')
# vpr.readaziamp(infname='./aziamp_-112.2_36.4.txt', wtype = 'r')
# vpr.readaziphi(infname='./aziphi_-112.2_36.4.txt', wtype = 'r')
# vpr.readmod(infname='mod_-112.2.36.4.mod', mtype='tti')


# vpr.readdisp(infname='./disp_-112.0_36.0_lov.txt', wtype = 'l')
# vpr.readdisp(infname='./disp_-112.0_36.0_ray.txt', wtype = 'r')
# vpr.readaziamp(infname='./aziamp_-112.0_36.0.txt', wtype = 'r')
# vpr.readaziphi(infname='./aziphi_-112.0_36.0.txt', wtype = 'r')

vpr.readdisp(infname='./synthetic_inv/disp_lov.txt', wtype = 'l')
vpr.readdisp(infname='./synthetic_inv/disp_ray.txt', wtype = 'r')
vpr.readaziamp(infname='./synthetic_inv/aziamp.ray.txt', wtype = 'r')
vpr.readaziphi(infname='./synthetic_inv/aziphi.ray.txt', wtype = 'r')

vpr.readmod(infname='mod_-112.0.36.0.mod', mtype='tti')
vpr.get_period()

vpr.getpara(mtype='tti')
# vpr.mc_inv_tti()
vpr.update_mod(mtype='tti')
vpr.model.ttimod.get_rho()
# vpr.read_tti_inv(indir='workingdir_tti')

vpr.read_tti_inv(indir='synthetic_inv_result')

minmisfit = vpr.misfit.min()
ind = (vpr.isacc == 1)*(vpr.misfit < 2.0*minmisfit)
strike = vpr.paraval[:, -3]

ind2 = (vpr.isacc == 1)*(vpr.misfit < 2.0*minmisfit)*(strike > 90.)

vpr.get_vmodel('tti')
vpr.compute_tcps(wtype='ray')
vpr.compute_tcps(wtype='love')

# plt.hist(vpr.paraval[:, i], bins=1000)
# plt.hist(np.repeat(vpr.model.ttimod.para.paraval[i], 100), alpha=0.3, bins=100 )
# plt.show()

# vpr.perturb_from_kernel(wtype='ray')
# vpr.perturb_from_kernel(wtype='love')
# 
# 
# fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True)
# ax = axs[0,0]
# ax.errorbar(vpr.indata.dispR.pper, vpr.indata.dispR.pvelo, yerr=vpr.indata.dispR.stdpvelo, fmt='o-')
# ax.plot(vpr.indata.dispR.pper, vpr.indata.dispR.pvelp, 'x')
# ax.set_title('Rayleigh wave dispersion')
# 
# # With 4 subplots, reduce the number of axis ticks to avoid crowding.
# ax.locator_params(nbins=4)
# 
# ax = axs[0,1]
# ax.errorbar(vpr.indata.dispR.pper, vpr.indata.dispR.pampo, yerr=vpr.indata.dispR.stdpampo, fmt='o-')
# ax.plot(vpr.indata.dispR.pper, vpr.indata.dispR.pampp, 'x')
# ax.set_title('Rayleigh wave azimuthal amplitude')
# 
# ax = axs[1,0]
# ax.errorbar(vpr.indata.dispR.pper, vpr.indata.dispR.pphio, yerr=vpr.indata.dispR.stdpphio, fmt='o-')
# ax.plot(vpr.indata.dispR.pper, vpr.indata.dispR.pphip, 'x')
# ax.set_title('Rayleigh wave fast-axis azimuth')
# 
# ax = axs[1,1]
# ax.errorbar(vpr.indata.dispL.pper, vpr.indata.dispL.pvelo, yerr=vpr.indata.dispL.stdpvelo, fmt='o-')
# ax.plot(vpr.indata.dispL.pper, vpr.indata.dispL.pvelp, 'x')
# ax.set_title('Love wave dispersion')
# 
# # fig.suptitle('Variable errorbars')
# 
# plt.show()

