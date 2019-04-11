import mcpost

# vpr = mcpost.postvpr(waterdepth = 0.07, factor=2., thresh=0.)
vpr = mcpost.postvpr(waterdepth = 0.07, thresh=0.02)

# vpr.read_inv_data('water_hongda/mc_inv.M3.npz')
# vpr.read_data('water_hongda/mc_data.M3.npz')

vpr.read_inv_data('../test_h5inv/mc_inv.188.0_60.5.npz')
vpr.read_data('../test_h5inv/mc_data.188.0_60.5.npz')
# 
vpr.get_vmodel()
vpr.plot_disp(showfig=False)
vpr.plot_profile()
