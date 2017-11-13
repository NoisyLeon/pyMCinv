import vprofile
import modparam
import vmodel

outdir = './synthetic_iso_inv'

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='old_code/TEST/Q22A.com.txt')
vpr.readrf(infname='old_code/TEST/in.rf')
vpr.readmod(infname='old_code/TEST/Q22A.mod1')
vpr.readpara(infname='old_code/TEST/in.para')

vpr.get_period()

vpr.get_rf_param()
vpr.model.isomod.mod2para()
vpr.model.isomod.para.new_paraval(0)
vpr.model.isomod.para2mod()
vpr.model.isomod.update()
vpr.get_vmodel(mtype = 'isotropic')
vpr.compute_fsurf()
vpr.compute_rftheo()

outfname = outdir+'/paraval.txt'
vpr.model.isomod.mod2para()
modparam.write_paraval_txt(outfname, vpr.model.isomod.para)

# write model
outfname = outdir+'/synthetic.mod'
vmodel.write_model(model=vpr.model, outfname=outfname, isotropic=True)


outfname = outdir+'/disp_ray.txt'
outArr  = np.append(vpr.indata.dispR.pper, vpr.indata.dispR.pvelp)
outArr  = np.append(outArr, vpr.indata.dispR.stdpvelo)
outArr  = outArr.reshape((3, vpr.indata.dispR.npper))
outArr  = outArr.T
np.savetxt(outfname, outArr, fmt='%g')


outfname = outdir+'/rf.txt'
outArr  = np.append(vpr.indata.rfr.tp, vpr.indata.rfr.rfp)
outArr  = np.append(outArr, vpr.indata.rfr.stdrfo)
outArr  = outArr.reshape((3, vpr.indata.rfr.npts))
outArr  = outArr.T
np.savetxt(outfname, outArr, fmt='%g')




