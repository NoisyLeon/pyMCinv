import sampler_test_tool
import pymc3 as pm
import numpy as np

p   = np.loadtxt('synthetic_iso_inv/real_para.txt')

with pm.Model(model=p) as model:

    disp_inv = pm.DensityDist('disp_inv', sampler_test_tool.logp_sampyl)#,\
        # observed={u'x1': p[0], u'x2': p[1], u'x3': p[2], u'x4': p[3], u'x5': p[4], u'x6': p[5], u'x7': p[6],\
         # u'x8': p[7], u'x9': p[8], u'x10': p[9],u'x11': p[10], u'x12': p[11], u'x13': p[12]})
    # sigma = pm.DensityDist('sigma', loglike2, testval=1)
    # # Create likelihood
    # like = pm.Normal('y_est', mu=alpha + beta *
    #                     xdata, sd=sigma, observed=ydata)

    # trace = pm.sample(2000, cores=1)