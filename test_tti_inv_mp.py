import vprofile
import modparam

vpr = vprofile.vprofile1d()


# vpr.readdisp(infname='./disp_-112.2_36.4_lov.txt', wtype = 'l')
# vpr.readdisp(infname='./disp_-112.2_36.4_ray.txt', wtype = 'r')
# vpr.readaziamp(infname='./aziamp_-112.2_36.4.txt', wtype = 'r')
# vpr.readaziphi(infname='./aziphi_-112.2_36.4.txt', wtype = 'r')
# vpr.readmod(infname='mod_-112.2.36.4.mod', mtype='tti')


vpr.readdisp(infname='./disp_-112.0_36.0_lov.txt', wtype = 'l')
vpr.readdisp(infname='./disp_-112.0_36.0_ray.txt', wtype = 'r')
vpr.readaziamp(infname='./aziamp_-112.0_36.0.txt', wtype = 'r')
vpr.readaziphi(infname='./aziphi_-112.0_36.0.txt', wtype = 'r')

# vpr.readdisp(infname='./synthetic_tti_inv/disp_lov.txt', wtype = 'l')
# vpr.readdisp(infname='./synthetic_tti_inv/disp_ray.txt', wtype = 'r')
# vpr.readaziamp(infname='./synthetic_tti_inv/aziamp.ray.txt', wtype = 'r')
# vpr.readaziphi(infname='./synthetic_tti_inv/aziphi.ray.txt', wtype = 'r')

vpr.readmod(infname='mod_-112.0.36.0.mod', mtype='tti')
vpr.getpara(mtype='tti')
# vpr.mc_inv_tti(outdir='/work3/leon/mc_inv/synthetic_tti_inv_result_vsv_vsh_eta')

# vpr.mc_inv_tti(outdir='/work3/leon/mc_inv/mod_-112.0.36.0_result')
# vpr.mc_inv_tti()





