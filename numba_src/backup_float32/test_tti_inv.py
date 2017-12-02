import vprofile
import modparam

vpr = vprofile.vprofile1d()


vpr.readdisp(infname='./disp_-112.2_36.4_lov.txt', wtype = 'l')
vpr.readdisp(infname='./disp_-112.2_36.4_ray.txt', wtype = 'r')
vpr.readaziamp(infname='./aziamp_-112.2_36.4.txt', wtype = 'r')
vpr.readaziphi(infname='./aziphi_-112.2_36.4.txt', wtype = 'r')
vpr.readmod(infname='mod_-112.2.36.4.mod', mtype='tti')
vpr.getpara(mtype='tti')
vpr.update_mod(mtype='tti')
vpr.model.ttimod.get_rho()
vpr.get_vmodel(mtype='tti')
vpr.get_period()
vpr.compute_tcps(wtype='ray')
vpr.compute_tcps(wtype='love')

vpr.model.ttimod.mod2para()
# 
tmodo = vpr.model.ttimod.copy()
# 
# 
vpr.model.ttimod.para.new_paraval(1)
vpr.model.ttimod.para2mod()
vpr.model.ttimod.update()
vpr.get_vmodel(mtype='tti')

vpr.model.dipArr[vpr.model.dipArr!=0.]=90.
vpr.model.strikeArr[vpr.model.strikeArr!=0.]=20.
vpr.model.rot_dip_strike()
vpr.model.decompose()
# vpr.compute_tcps()

# 
# import tcps
# import numpy as np
# 
# tcpsR1 = tcps.tcps_solver(vpr.model)
# tcpsR1.init_default()
# # tcpsR1.dArr = vpr.model.get_dArr()
# tcpsR1.solve_PSV()
# 
# CR1  = []
# azArr = np.arange(360)*1.
# for az in np.arange(360)*1.:
#     tcpsR1.psv_azi_perturb(az, True)
#     CR1.append(tcpsR1.CA[1])
#     
# CR1= np.array(CR1)


