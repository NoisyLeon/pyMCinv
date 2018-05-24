import invsolver

solver = invsolver.invsolver1d()
solver.readdisp('../synthetic_iso_inv/disp_ray.txt')
solver.readrf('../synthetic_iso_inv/rf.txt')

solver.readmod('../old_code/TEST/Q22A.mod1')

# solver.readpara('../old_code/TEST/in.para')
solver.getpara()
solver.update_mod_interface()
solver.get_vmodel_interface()
solver.get_period_interface()