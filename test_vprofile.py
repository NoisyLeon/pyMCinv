import vprofile
import modparam

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='./old_code/TEST/Q22A.com.txt')
vpr.readrf(infname='./old_code/TEST/in.rf')
vpr.readmod(infname='./old_code/TEST/Q22A.mod1')
# vpr.readpara(infname='../old_code/TEST/in.para')
vpr.getpara()
# # vpr.mc_inv_iso()
# # 
# # vpr.read_iso_inv(indir='./workingdir')
# # 
# # 
# # # m2 = modparam.isomod()
# # 
# # 
# # 
# vpr.get_period()
# vpr.update_mod()
# vpr.get_rf_param()
# vpr.get_vmodel()
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