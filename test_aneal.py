import sampler_test_tool
import numpy as np
from scipy import optimize


p0   = np.loadtxt('synthetic_iso_inv/real_para.txt')+np.random.rand(13)/10. 

res = optimize.basinhopping(sampler_test_tool.logp_anneal, p0,  niter=10)

