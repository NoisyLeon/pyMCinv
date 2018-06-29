import vmodel, modparam, data
import fast_surf, theo
import numpy as np


disp_data   = np.loadtxt('synthetic_iso_inv/in.disp')

def logp_sampyl(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    T       = disp_data[:, 0]
    pvelo   = disp_data[:, 1]
    stdpvelo= disp_data[:, 2]
    # p       = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])
    # paraval = np.zeros(13)
    # try:
    #     paraval[0]  = x1
    # except:
    #     paraval[0]  = x1._value
    
    
    model   = vmodel.model1d()
    model.get_para_model(paraval=np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13]))
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
    L                       = np.exp(-0.5 * misfit)
    return L
    # return paraval.sum()


def logp_emcee(p):
    T       = disp_data[:, 0]
    pvelo   = disp_data[:, 1]
    stdpvelo= disp_data[:, 2]
    # p       = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])
    # paraval = np.zeros(13)
    # try:
    #     paraval[0]  = x1
    # except:
    #     paraval[0]  = x1._value
    
    # print p
    model   = vmodel.model1d()
    model.get_para_model(paraval=p)
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
    # print 'a'
    # print p
    # print pvelp
    return L





def logp_anneal(p):
    T       = disp_data[:, 0]
    pvelo   = disp_data[:, 1]
    stdpvelo= disp_data[:, 2]
    # p       = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])
    # paraval = np.zeros(13)
    # try:
    #     paraval[0]  = x1
    # except:
    #     paraval[0]  = x1._value
    
    # print p
    model   = vmodel.model1d()
    model.get_para_model(paraval=p)
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
    # print 'a'
    # print p
    # print pvelp
    # print misfit
    return misfit


# def logp_sampyl(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
#     paraval=np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])
#     return paraval.sum()
# 
# p   = np.loadtxt('synthetic_iso_inv/real_para.txt')
# 
# print logp_sampyl(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12])