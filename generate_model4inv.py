import vprofile
import modparam
import vmodel

outdir = './synthetic_inv'

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='./disp_-112.0_36.0_lov.txt', wtype = 'l')
vpr.readdisp(infname='./disp_-112.0_36.0_ray.txt', wtype = 'r')
vpr.readaziamp(infname='./aziamp_-112.0_36.0.txt', wtype = 'r')
vpr.readaziphi(infname='./aziphi_-112.0_36.0.txt', wtype = 'r')
vpr.readmod(infname='mod_-112.0.36.0.mod', mtype='tti')
vpr.getpara(mtype='tti')
vpr.update_mod(mtype='tti')
vpr.get_vmodel(mtype='tti')
vpr.model.ttimod.new_paraval(0, 1, 1, 0, 1)
vpr.get_vmodel(mtype='tti')

outfname = outdir+'/paraval.txt'
vpr.model.ttimod.mod2para()
modparam.write_paraval_txt(outfname, vpr.model.ttimod.para)

import tcps
import numpy as np

tcpsR = tcps.tcps_solver(vpr.model)
vpr.model.flat=0
tcpsR.init_default_4()
tcpsR.dArr = vpr.model.get_dArr()
tcpsR.solve_PSV()


tcpsL = tcps.tcps_solver(vpr.model)
vpr.model.flat=0
tcpsL.init_default_3()
tcpsL.dArr = vpr.model.get_dArr()
tcpsL.solve_SH()

# write model
outfname = outdir+'/synthetic.mod'
vmodel.write_model(model=vpr.model, outfname=outfname, isotropic=False)


outfname = outdir+'/disp_ray.txt'
outArr  = np.append(tcpsR.T, tcpsR.C)
outArr  = np.append(outArr, np.ones(tcpsR.T.size)*0.002)
outArr  = outArr.reshape((3, tcpsR.T.size))
outArr  = outArr.T
np.savetxt(outfname, outArr, fmt='%g')


outfname = outdir+'/disp_lov.txt'
outArr  = np.append(tcpsL.T, tcpsL.C)
outArr  = np.append(outArr, np.ones(tcpsL.T.size)*0.002)
outArr  = outArr.reshape((3, tcpsL.T.size))
outArr  = outArr.T
np.savetxt(outfname, outArr, fmt='%g')


ampArr = []
phiArr = []
for i in xrange(tcpsR.T.size):
    CR  = []
    azArr = np.arange(360)*1.
    for az in azArr:
        tcpsR.psv_azi_perturb(az, False)
        CR.append(tcpsR.CA[i])
        
    CR= np.array(CR)
    ampArr.append((CR.max() - CR.min())/2.)
    phi = azArr[CR.argmax()]
    if phi > 180.:
        phi -= 180.
    phiArr.append(phi)

ampArr= np.array(ampArr)
phiArr= np.array(phiArr)


outfname = outdir+'/aziamp.ray.txt'
outArr  = np.append(tcpsR.T, ampArr)
outArr  = np.append(outArr, np.ones(tcpsR.T.size)*0.005)
outArr  = outArr.reshape((3, tcpsR.T.size))
outArr  = outArr.T
np.savetxt(outfname, outArr, fmt='%g')


outfname = outdir+'/aziphi.ray.txt'
outArr  = np.append(tcpsR.T, phiArr)
outArr  = np.append(outArr, np.ones(tcpsR.T.size)*2.)
outArr  = outArr.reshape((3, tcpsR.T.size))
outArr  = outArr.T
np.savetxt(outfname, outArr, fmt='%g')


