import vmodel

m=vmodel.model1d()
m.isomod.readmodtxt('../old_code/TEST/Q22A.mod1')
m.isomod.update_interface()