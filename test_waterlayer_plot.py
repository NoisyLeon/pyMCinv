import mcpost

vpr = mcpost.postvpr(waterdepth = 2.9)

vpr.read_inv_data('water_hongda/mc_inv.M3.npz')
vpr.read_data('water_hongda/mc_data.M3.npz')

vpr.get_vmodel()
vpr.plot_disp(showfig=False)
vpr.plot_profile()
