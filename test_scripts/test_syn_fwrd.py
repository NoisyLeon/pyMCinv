import vprofile
import modparam, copy

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='./old_code/weisen_old_code/TEST/Q22A.com.txt')
vpr.readrf(infname='./old_code/weisen_old_code/TEST/in.rf')
vpr.readmod(infname='./old_code/weisen_old_code/TEST/Q22A.mod1')
vpr.getpara()


vpr.get_period()
vpr.update_mod()
vpr.get_vmodel()
vpr.model.isomod.mod2para()

newmod      = copy.deepcopy(vpr.model.isomod)
newmod.para.new_paraval(0)
newmod.para2mod()
newmod.update()
# loop to find the "good" model,
# satisfying the constraint (3), (4) and (5) in Shen et al., 2012 
igood       = 0
while ( not newmod.isgood(0, 1, 1, 0)):
    igood   += igood + 1
    newmod  = copy.deepcopy(vpr.model.isomod)
    newmod.para.new_paraval(0)
    newmod.para2mod()
    newmod.update()
    
vpr.model.isomod = newmod

vpr.get_vmodel()


vpr.compute_fsurf()
vpr.compute_rftheo()

vpr.data.rfr.rfo        = vpr.data.rfr.rfp.copy()
vpr.data.rfr.writerftxt('synthetic_iso_inv/in.rf', tf=16.475, prerf=False)
vpr.data.dispR.pvelo    = vpr.data.dispR.pvelp.copy()
vpr.data.dispR.writedisptxt('synthetic_iso_inv/in.disp', predisp=False)

vpr.model.isomod.para.write_paraval_txt('synthetic_iso_inv/real_para.txt')




