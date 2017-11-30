import invsolver
import vprofile


vpr = vprofile.vprofile1d()
vpr.solver.readdisp('../synthetic_iso_inv/disp_ray.txt')
vpr.solver.readrf('../synthetic_iso_inv/rf.txt')

vpr.solver.readmod('../old_code/TEST/Q22A.mod1')

# solver.readpara('../old_code/TEST/in.para')
vpr.solver.getpara()
vpr.solver.update_mod_interface()
vpr.solver.get_vmodel_interface()
vpr.solver.get_period_interface()
vpr.mc_inv_iso_mp('workingdir_iso_mp/')
