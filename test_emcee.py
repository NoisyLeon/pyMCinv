import sampler_test_tool
import numpy as np
import emcee

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
ndim, nwalkers = 13, 26
p0   = [np.loadtxt('synthetic_iso_inv/real_para.txt')+np.random.rand(ndim)/10. for i in range(nwalkers)]
# ivar = 1. / np.random.rand(ndim)
# p0 = [np.random.rand(ndim) for i in range(nwalkers)]

sampler = emcee.EnsembleSampler(nwalkers, ndim, sampler_test_tool.logp_emcee, threads=12, a = 1.)
sampler.run_mcmc(p0, 100)


