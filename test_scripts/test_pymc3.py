import sampler_test_tool
import pymc3 as pm
import numpy as np
import vmodel, modparam, data, vprofile
import fast_surf, theo


refvaluein    = np.loadtxt('synthetic_iso_inv/real_para.txt')

class disp(pm.Continuous):
    def __init__(self, disp_data, refvalue, *args, **kwargs):
        super(disp, self).__init__(*args, **kwargs)
        self.disp_data  = disp_data
        # # self.T          = T
        self.mean = refvalue
        
    def logp(self, value):
        T       = self.disp_data[:, 0]
        pvelo   = self.disp_data[:, 1]
        stdpvelo= self.disp_data[:, 2]
        model   = vmodel.model1d()
        model.get_para_model(paraval=value)
        model.isomod.mod2para()
        ilvry                   = 2
        nper                    = T.size
        per                     = np.zeros(200, dtype=np.float64)
        per[:nper]              = T[:]
        qsinv                   = 1./model.qs
        # print model.qs
        (ur0,ul0,cr0,cl0)       = fast_surf.fast_surf(model.nlay, ilvry, \
                                    model.vpv, model.vsv, model.rho, model.h, qsinv, per, nper)
        pvelp                   = cr0[:nper]
        gvelp                   = ur0[:nper]
        
        # misfit
        misfit                  = ((pvelo - pvelp)**2/stdpvelo**2).sum()
        L                       = np.exp(-0.5 * np.sqrt(misfit))
        return np.log(L)
    
indisp_data   = np.loadtxt('synthetic_iso_inv/in.disp')


value_in       = np.loadtxt('synthetic_iso_inv/real_para.txt')


with pm.Model() as Model:
    dispinv     = disp(name='disp', disp_data=indisp_data, refvalue=refvaluein)
    # print dispinv.logp(value=value_in)
# 
#     disp_inv = pm.DensityDist('disp_inv', sampler_test_tool.logp_sampyl)#,\
        # observed={u'x1': p[0], u'x2': p[1], u'x3': p[2], u'x4': p[3], u'x5': p[4], u'x6': p[5], u'x7': p[6],\
         # u'x8': p[7], u'x9': p[8], u'x10': p[9],u'x11': p[10], u'x12': p[11], u'x13': p[12]})
    # sigma = pm.DensityDist('sigma', loglike2, testval=1)
    # # Create likelihood
    # like = pm.Normal('y_est', mu=alpha + beta *
    #                     xdata, sd=sigma, observed=ydata)

    # trace = pm.sample(2000, cores=1)