import modparam
import vmodel

m=vmodel.model1d()



# modparam.readtimodtxt('demo_7.mod', inmod=tmod)
modparam.readtimodtxt('mod_-112.2.36.4.mod', inmod=m.ttimod)

m.ttimod.update()
m.ttimod.get_rho()
