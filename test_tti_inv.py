import vprofile
import modparam

vpr = vprofile.vprofile1d()
# vpr.readdisp(infname='/work3/leon/US_data/phvel_Lov/disp_-112.2_36.4.txt', wtype = 'l')
# vpr.readdisp(infname='/work3/leon/US_data/phvel_Ray/disp_-112.2_36.4.txt', wtype = 'r')
# vpr.readaziamp(infname='/work3/leon/US_data/phvel_Ray/aziamp_-112.2_36.4.txt', wtype = 'r')
# vpr.readaziphi(infname='/work3/leon/US_data/phvel_Ray/aziphi_-112.2_36.4.txt', wtype = 'r')

vpr.readdisp(infname='./disp_-112.2_36.4_lov.txt', wtype = 'l')
vpr.readdisp(infname='./disp_-112.2_36.4_ray.txt', wtype = 'r')
vpr.readaziamp(infname='./aziamp_-112.2_36.4.txt', wtype = 'r')
vpr.readaziphi(infname='./aziphi_-112.2_36.4.txt', wtype = 'r')
vpr.readmod(infname='mod_-112.2.36.4.mod', mtype='tti')
vpr.getpara(mtype='tti')

# vpr.readrf(infname='old_code/TEST/in.rf')
# vpr.readmod(infname='old_code/TEST/Q22A.mod1')
# vpr.readpara(infname='old_code/TEST/in.para')
# vpr.mc_inv_iso()
# 
# vpr.read_iso_inv(indir='./workingdir')
# 
# 
# # m2 = modparam.isomod()
# 
# 
# 
# # vpr.get_period()
# # vpr.update_mod()
# # vpr.get_rf_param()
# # vpr.get_vmodel()
# # vpr.compute_fsurf()
# # vpr.compute_rftheo()
# 
# 
# vpr2 = vprofile.vprofile1d()
# vpr2.readdisp(infname='old_code/TEST/Q22A.com.txt')
# vpr2.readrf(infname='old_code/TEST/in.rf')
# vpr2.readmod(infname='old_code/TEST/Q22A.mod1')
# vpr2.readpara(infname='old_code/TEST/in.para')
# # vpr.mc_inv_iso()
# 
# vpr2.read_iso_inv(indir='./workingdir')