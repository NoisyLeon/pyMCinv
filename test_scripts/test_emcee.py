import sampler_test_tool
import numpy as np
import emcee
import time
import vprofile

# def lnprob(x):
#     return -0.5 * np.sum(1 * x ** 2)
# 
# ndim, nwalkers = 10, 100
# ivar = 1. / np.random.rand(ndim)
# p0 = [np.random.rand(ndim) for i in range(nwalkers)]
# # p0 = np.random.rand(ndim)
# sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob)
# sampler.run_mcmc(p0, 1000)


# p0   = np.loadtxt('synthetic_iso_inv/real_para.txt')
ndim, nwalkers = 13, 30
ntemps = 10
p0   = [np.loadtxt('synthetic_iso_inv/real_para.txt')+np.random.rand(ndim)/10. for i in range(nwalkers)]
# ivar = 1. / np.random.rand(ndim)
# p0 = [np.random.rand(ndim) for i in range(nwalkers)]

# sampler = emcee.EnsembleSampler(nwalkers, ndim, sampler_test_tool.lnpostfn_emcee, threads=12, a = 1.)


p0  = np.zeros(((ntemps, nwalkers, ndim)))

vpr = vprofile.vprofile1d()
# vpr.readdisp(infname='synthetic_iso_inv/in.disp')
# vpr.readrf(infname='synthetic_iso_inv/in.rf')
vpr.readmod(infname='./old_code/weisen_old_code/TEST/Q22A.mod1')
vpr.getpara()

vpr.get_period()
vpr.update_mod(mtype = 'iso')
vpr.get_vmodel(mtype = 'iso')
vpr.model.isomod.mod2para()
newmod          = vpr.model.isomod
for i in range(nwalkers):
    # print i
    for j in range(ntemps):
        # newmod      = copy.deepcopy(oldmod)
        newmod.para.new_paraval(0)
        newmod.para2mod()
        newmod.update()
        m0  = 0
        m1  = 1
        if newmod.mtype[0] == 5: # water layer, added May 16th, 2018
            m0  += 1
            m1  += 1
        igood       = 0
        while ( not newmod.isgood(m0, m1, 1, 0)):
            # # newmod      = copy.deepcopy(oldmod)
            newmod.para.new_paraval(0)
            newmod.para2mod()
            newmod.update()
        p0[j, i, :]     = newmod.para.paraval[:]
    
        
    
    # space = vpr.model.isomod.para.space.copy()
    
# 
# 
# p0      = np.array([np.loadtxt('synthetic_iso_inv/real_para.txt')+np.random.rand(ndim)/10. for i in range(nwalkers*ntemps)])
# p0      = p0.reshape(ntemps, nwalkers, ndim)
# # p0   = p0.reshape
sampler = emcee.PTSampler(ntemps=ntemps, nwalkers=nwalkers, dim=ndim, logp=sampler_test_tool.logp_emcee, logl=sampler_test_tool.logl_emcee, threads=5, a = 2.)

start       = time.time()
out         = sampler.run_mcmc(p0, 2000)
end         = time.time()
print 'elasped time =',end-start
