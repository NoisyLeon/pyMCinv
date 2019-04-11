import vprofile
import modparam
import vmodel
import numpy as np
outdir = './synthetic_tti_inv_gn'

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='./disp_-112.0_36.0_lov.txt', wtype = 'l')
vpr.readdisp(infname='./disp_-112.0_36.0_ray.txt', wtype = 'r')
vpr.readaziamp(infname='./aziamp_-112.0_36.0.txt', wtype = 'r')
vpr.readaziphi(infname='./aziphi_-112.0_36.0.txt', wtype = 'r')
vpr.readmod(infname='mod_-112.0.36.0.mod', mtype='tti')
vpr.getpara(mtype='tti')
vpr.update_mod(mtype='tti')
vpr.get_vmodel(mtype='tti')

vpr.model.ttimod.mod2para()

outfname = 'synthetic_tti_inv_gn/paraval_ref.txt'
modparam.write_paraval_txt(outfname, vpr.model.ttimod.para)
outfname = outdir+'/synthetic_ref.mod'
vmodel.write_model(model=vpr.model, outfname=outfname, isotropic=False)

vpr.get_period()
vpr.compute_tcps(wtype='ray')
vpr.compute_tcps(wtype='love')

vpr.model.ttimod.new_paraval(0, 1, 1, 0, 0)
vpr.get_vmodel(mtype='tti')

vpr.perturb_from_kernel()
vpr.perturb_from_kernel(wtype='love')

outfname = outdir+'/paraval.txt'
vpr.model.ttimod.mod2para()
modparam.write_paraval_txt(outfname, vpr.model.ttimod.para)

# write model
outfname = outdir+'/synthetic.mod'
vmodel.write_model(model=vpr.model, outfname=outfname, isotropic=False)


outfname = outdir+'/disp_ray.txt'
outArr  = np.append(vpr.indata.dispR.pper, vpr.indata.dispR.pvelp)
outArr  = np.append(outArr, np.ones(vpr.indata.dispR.npper)*0.002)
outArr  = outArr.reshape((3, vpr.indata.dispR.npper))
outArr  = outArr.T
np.savetxt(outfname, outArr, fmt='%g')


outfname = outdir+'/disp_lov.txt'
outArr  = np.append(vpr.indata.dispL.pper, vpr.indata.dispL.pvelp)
outArr  = np.append(outArr, np.ones(vpr.indata.dispL.npper)*0.002)
outArr  = outArr.reshape((3, vpr.indata.dispL.npper))
outArr  = outArr.T
np.savetxt(outfname, outArr, fmt='%g')




outfname = outdir+'/aziamp.ray.txt'
outArr  = np.append(vpr.indata.dispR.pper, vpr.indata.dispR.pampp)
outArr  = np.append(outArr, np.ones(vpr.indata.dispR.npper)*0.005)
outArr  = outArr.reshape((3, vpr.indata.dispR.npper))
outArr  = outArr.T
np.savetxt(outfname, outArr, fmt='%g')


outfname = outdir+'/aziphi.ray.txt'
outArr  = np.append(vpr.indata.dispR.pper, vpr.indata.dispR.pphip)
outArr  = np.append(outArr, np.ones(vpr.indata.dispR.npper)*2.)
outArr  = outArr.reshape((3, vpr.indata.dispR.npper))
outArr  = outArr.T
np.savetxt(outfname, outArr, fmt='%g')


