import sampler_test_tool
import numpy as np
from scipy import optimize
import vprofile, vmodel
# 
# 
# p0   = np.loadtxt('synthetic_iso_inv/real_para.txt')+np.random.rand(13)/10. 
# 
# res = optimize.basinhopping(sampler_test_tool.logp_anneal, p0,  niter=10)
# 

vpr = vprofile.vprofile1d()
vpr.readdisp(infname='synthetic_iso_inv/in.disp')
vpr.readrf(infname='synthetic_iso_inv/in.rf')
vpr.readmod(infname='./old_code/weisen_old_code/TEST/Q22A.mod1')
vpr.getpara()

vpr.get_period()
vpr.update_mod(mtype = 'iso')
vpr.get_vmodel(mtype = 'iso')

vpr.model.isomod.mod2para()

space = vpr.model.isomod.para.space.copy()

bounds  = []
for igr in range(space.shape[1]):
    bounds.append((space[0, igr], space[1, igr]))

ranges  = []
for igr in range(space.shape[1]):
    ranges  += [slice(space[0, igr], space[1, igr], space[2, igr])]
    
pmin    = space[0, :]
pmax    = space[1, :]




# result  = optimize.differential_evolution(sampler_test_tool.misfit_opt, bounds=bounds, args=(), strategy='best1bin',\
#     maxiter=1000, popsize=20, tol=0.01, mutation=(0.5, 1), recombination=0.7, \
#     seed=None, callback=None, disp=True, polish=True, init='latinhypercube')


# result  = optimize.brute(sampler_test_tool.misfit_opt, ranges=ranges,  Ns=13, disp=True)

def constraint(**kwargs):
    x           = kwargs["x_new"]
    tmax        = bool(np.all(x <= pmax))
    tmin        = bool(np.all(x >= pmin))
    # model       = vmodel.model1d()
    # model.get_para_model(paraval=x)
    # model.isomod.mod2para()
    # is_good     = model.isomod.isgood(0, 1, 1, 0)
    return tmax and tmin #and is_good
    

# 
# bounds = [(0,2), (0, 2), (0, 2), (0, 2), (0, 2)]
# result = optimize.differential_evolution(rosen, bounds)
# result.x, result.fun
# 
# minimizer_kwargs = {"method":"L-BFGS-B", "jac":True}
# def print_fun(x, f, accepted):
#     print("at minimum %.4f accepted %d" % (f, int(accepted)))
#         
# ret = optimize.basinhopping(sampler_test_tool.misfit_opt, x0=vpr.model.isomod.para.paraval,\
#                    niter=10, accept_test=constraint, callback=print_fun, disp=True)

