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
# 
vpr.model.ttimod.para.new_paraval(1)
vpr.model.ttimod.para2mod()
vpr.model.ttimod.update()
vpr.get_vmodel(mtype='tti')

vpr.model.dipArr[vpr.model.dipArr!=0.]=90.
vpr.model.strikeArr[vpr.model.strikeArr!=0.]=33.
vpr.model.init_etensor()
vpr.model.rot_dip_strike()
vpr.model.decompose()
vpr.compute_tcps(wtype='ray')
vpr.compute_tcps(wtype='love')

# 
import tcps
import numpy as np

tcpsR1 = tcps.tcps_solver(vpr.model)
vpr.model.flat=0
tcpsR1.init_default_4()
tcpsR1.dArr = vpr.model.get_dArr()
tcpsR1.solve_PSV()

CR1  = []
azArr = np.arange(360)*1.
for az in np.arange(360)*1.:
    tcpsR1.psv_azi_perturb(az, False)
    CR1.append(tcpsR1.CA[1])
    
CR1= np.array(CR1)

tcpsL = tcps.tcps_solver(vpr.model)
vpr.model.flat=0
tcpsL.init_default_3()
tcpsL.dArr = vpr.model.get_dArr()
tcpsL.solve_SH()


