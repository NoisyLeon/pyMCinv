import vmodel, modparam, data, vprofile
import fast_surf, theo
import numpy as np
import pymc3 as pm

disp_data   = np.loadtxt('synthetic_iso_inv/in.disp')

def logp_sampyl(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
    T       = disp_data[:, 0]
    pvelo   = disp_data[:, 1]
    stdpvelo= disp_data[:, 2]
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

def lnpostfn_emcee(p):
    if not np.all(p > space[0, :]):
        return -np.inf
    if not np.all(p < space[1, :]):
        return -np.inf
    T       = disp_data[:, 0]
    pvelo   = disp_data[:, 1]
    stdpvelo= disp_data[:, 2]
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
    return np.log(L)

def logp_emcee(p):
    if not np.all(p > space[0, :]):
        return -np.inf
    if not np.all(p < space[1, :]):
        return -np.inf
    return 0.

def logl_emcee(p):
    T       = disp_data[:, 0]
    pvelo   = disp_data[:, 1]
    stdpvelo= disp_data[:, 2]
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
    return np.log(L)

def misfit_opt(p):
    T           = disp_data[:, 0]
    pvelo       = disp_data[:, 1]
    stdpvelo    = disp_data[:, 2]
    model       = vmodel.model1d()
    model.get_para_model(paraval=p)
    model.isomod.mod2para()
    ilvry                   = 2
    nper                    = T.size
    per                     = np.zeros(200, dtype=np.float64)
    per[:nper]              = T[:]
    qsinv                   = 1./model.qs
    (ur0,ul0,cr0,cl0)       = fast_surf.fast_surf(model.nlay, ilvry, \
                                model.vpv, model.vsv, model.rho, model.h, qsinv, per, nper)
    pvelp                   = cr0[:nper]
    gvelp                   = ur0[:nper]
    
    # misfit
    misfit                  = ((pvelo - pvelp)**2/stdpvelo**2).sum()
    # L                       = np.exp(-0.5 * np.sqrt(misfit))
    # print 'run'
    return misfit




# bounds  = ()
lbounds = []
ubounds = []
for igr in range(space.shape[1]):
    lbounds.append(space[0, igr])
    ubounds.append(space[1, igr])
    # bounds  += ([space[0, igr], space[1, igr]], )

class disp_func:
    def fitness(self, x):
        T           = disp_data[:, 0]
        pvelo       = disp_data[:, 1]
        stdpvelo    = disp_data[:, 2]
        model       = vmodel.model1d()
        model.get_para_model(paraval=x)
        model.isomod.mod2para()
        ilvry                   = 2
        nper                    = T.size
        per                     = np.zeros(200, dtype=np.float64)
        per[:nper]              = T[:]
        qsinv                   = 1./model.qs
        (ur0,ul0,cr0,cl0)       = fast_surf.fast_surf(model.nlay, ilvry, \
                                    model.vpv, model.vsv, model.rho, model.h, qsinv, per, nper)
        pvelp                   = cr0[:nper]
        gvelp                   = ur0[:nper]
        # misfit
        misfit                  = ((pvelo - pvelp)**2/stdpvelo**2).sum()
        # L                       = np.exp(-0.5 * np.sqrt(misfit))
        # print 'run'
        return [misfit]
    
    def get_bounds(self):
        return (lbounds, ubounds)
    
# def disp_island:
    
    

    
    

# def logp_sampyl(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13):
#     paraval=np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])
#     return paraval.sum()
# 
# p   = np.loadtxt('synthetic_iso_inv/real_para.txt')
# 
# print logp_sampyl(p[0], p[1], p[2], p[3], p[4], p[5], p[6], p[7], p[8], p[9], p[10], p[11], p[12])