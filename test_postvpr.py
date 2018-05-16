import mcpost

vpr = mcpost.postvpr(waterdepth = 2.9)
# vpr.read_inv_data('/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_results/mc_inv.TA.TOLK.npz')
# vpr.read_data('/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_results/mc_data.TA.TOLK.npz')

# vpr.read_inv_data('../../mc_results_Miller/mc_inv.TA.TOLK.npz')
# vpr.read_data('../../mc_results_Miller/mc_data.TA.TOLK.npz')

# vpr.read_inv_data('/home/lili/new_mc_results_Miller/mc_inv.AK.HDA.npz')
# vpr.read_data('/home/lili/new_mc_results_Miller/mc_data.AK.HDA.npz')

# vpr.read_inv_data('../../mc_results_Miller/mc_inv.AK.HDA.npz')
# vpr.read_data('../../mc_results_Miller/mc_data.AK.HDA.npz')
# 
# vpr.read_inv_data('/work3/leon/mc_inv_files/new_mc_results_Miller/mc_inv.AK.HDA.npz')
# vpr.read_data('/work3/leon/mc_inv_files/new_mc_results_Miller/mc_data.AK.HDA.npz')

# vpr.read_inv_data('/work3/leon/mc_inv_files/mc_results_Miller/mc_inv.TA.TOLK.npz')
# vpr.read_data('/work3/leon/mc_inv_files/mc_results_Miller/mc_data.TA.TOLK.npz')

# vpr.read_inv_data('synthetic_working/mc_inv.CU.LF.npz')
# vpr.read_data('synthetic_working/mc_data.CU.LF.npz')

# vpr.plot_hist_three_group(x1min=29, x1max=33.5, x2min=33.5, x2max=39, x3min=40.5, x3max=42.5, ind_p=-1, ind_s=-1, bins1=5, bin2=5, bins3=5)

vpr.read_inv_data('water_hongda/mc_inv.M3.npz')
vpr.read_data('water_hongda/mc_data.M3.npz')