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
vpr.readmod(infname='mod_-112.0.36.0.mod', mtype='tti')
vpr.getpara(mtype='tti')
vpr.mc_inv_tti()



# vpr.update_mod(mtype='tti')
# vpr.model.ttimod.get_rho()
# 
# newmod  = vpr.model.ttimod.copy()
# for i in xrange(1000000):
#     newmod.mod2para()
#     newmod.para.new_paraval(1)
#     newmod.para2mod()
#     newmod.update()
#     print i
#     if newmod.isgood(0, 1, 1, 0):
#         break

# vpr.get_vmodel(mtype='tti')
# vpr.get_period()
# vpr.compute_tcps(wtype='ray')
# vpr.compute_tcps(wtype='love')
# 
# vpr.model.ttimod.mod2para()
# # newmod.isgood(0, 1, 1, 0)
# tmodo = vpr.model.ttimod.copy()
# # 
# # 
# # vpr.model.ttimod.para.new_paraval(1)
# # vpr.model.ttimod.para2mod()
# # vpr.model.ttimod.update()
# # vpr.get_vmodel(mtype='tti')
# 
# vpr.model.dipArr[:]=90.
# vpr.model.strikeArr[vpr.model.strikeArr!=0.]=0.
# vpr.model.init_etensor()
# vpr.model.rot_dip_strike()
# vpr.model.decompose()
# # vpr.compute_tcps()


