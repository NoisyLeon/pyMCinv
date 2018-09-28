import mcpost
import numpy as np
# vpr = mcpost.postvpr(thresh=0.2)
# # vpr.read_inv_data('/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_results/mc_inv.TA.TOLK.npz')
# # vpr.read_data('/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_results/mc_data.TA.TOLK.npz')
# 
# # vpr.read_inv_data('../../mc_results_Miller/mc_inv.TA.TOLK.npz')
# # vpr.read_data('../../mc_results_Miller/mc_data.TA.TOLK.npz')
# 
# # vpr.read_inv_data('/home/lili/new_mc_results_Miller/mc_inv.AK.HDA.npz')
# # vpr.read_data('/home/lili/new_mc_results_Miller/mc_data.AK.HDA.npz')
# 
# # vpr.read_inv_data('../../mc_results_Miller/mc_inv.AK.HDA.npz')
# # vpr.read_data('../../mc_results_Miller/mc_data.AK.HDA.npz')
# # 
# # vpr.read_inv_data('/work3/leon/mc_inv_files/new_mc_results_Miller/mc_inv.AK.HDA.npz')
# # vpr.read_data('/work3/leon/mc_inv_files/new_mc_results_Miller/mc_data.AK.HDA.npz')
# 
# # vpr.read_inv_data('/work3/leon/mc_inv_files/mc_results_Miller/mc_inv.TA.TOLK.npz')
# # vpr.read_data('/work3/leon/mc_inv_files/mc_results_Miller/mc_data.TA.TOLK.npz')
# 
# # vpr.read_inv_data('synthetic_working/mc_inv.CU.LF.npz')
# # vpr.read_data('synthetic_working/mc_data.CU.LF.npz')
# 
# # vpr.plot_hist_three_group(x1min=29, x1max=33.5, x2min=33.5, x2max=39, x3min=40.5, x3max=42.5, ind_p=-1, ind_s=-1, bins1=5, bin2=5, bins3=5)
# 
# # vpr.read_inv_data('/work3/leon/mc_inv_files/mc_alaska_surf/mc_inv.205.0_65.0.npz')
# # vpr.read_data('/work3/leon/mc_inv_files/mc_alaska_surf/mc_data.205.0_65.0.npz')
# 
# 
# # vpr.read_inv_data('/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_alaska_surf_150000/mc_inv.225.0_60.0.npz')
# # vpr.read_data('/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_alaska_surf_150000/mc_data.225.0_60.0.npz')
# 
# vpr.read_inv_data('/home/leon/code/pyMCinv/workingdir/mc_inv.BOTH.npz')
# vpr.read_data('/home/leon/code/pyMCinv/workingdir/mc_data.BOTH.npz')

# vpr.read_inv_data('/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_alaska_surf/mc_inv.195.0_59.5.npz')
# vpr.read_data('/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_alaska_surf/mc_data.195.0_59.5.npz')

# real_paraval = np.loadtxt('synthetic_iso_inv/real_para.txt')
# 
# vpr.read_inv_data('synthetic_working/mc_inv.CU.LF.npz')
# vpr.read_data('synthetic_working/mc_data.CU.LF.npz')


vpr = mcpost.postvpr(thresh=0.5, factor=1., stdfactor=20.)
vpr.read_data('/home/leon/code/pyMCinv/workingdir/mc_data.BOTH.npz')
# vpr.read_inv_data('/home/leon/code/pyMCinv/workingdir/mc_inv.BOTH.npz', thresh_misfit=1.)
vpr.read_inv_data('/home/leon/code/pyMCinv/workingdir/mc_inv.BOTH.npz')
vpr.get_vmodel()

# vpr = mcpost.postvpr(thresh=0.5, factor=1.)
# vpr.read_inv_data('/home/leon/code/pyMCinv/workingdir/mc_inv.BOTH.npz')
# vpr.read_data('/home/leon/code/pyMCinv/workingdir/mc_data.BOTH.npz')
# vpr.get_vmodel()
# 
# vpr_ph = mcpost.postvpr(thresh=0.2, factor=1.)
# vpr_ph.read_inv_data('/home/leon/code/pyMCinv/workingdir/mc_inv.PH.npz')
# vpr_ph.read_data('/home/leon/code/pyMCinv/workingdir/mc_data.BOTH.npz')
# vpr_ph.get_vmodel()
# 
# vpr_gr = mcpost.postvpr(thresh=0.2, factor=1.)
# vpr_gr.read_inv_data('/home/leon/code/pyMCinv/workingdir/mc_inv.GR.npz')
# vpr_gr.read_data('/home/leon/code/pyMCinv/workingdir/mc_data.BOTH.npz')
# vpr_gr.get_vmodel()

# vpr = mcpost.postvpr(thresh=0.5, factor=1., stdfactor=3.)
# vpr.read_inv_data('/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20180919_150000_both/mc_inv.206.0_64.5.npz')
# vpr.read_data('/work1/leon/ALASKA_work/mc_inv_files/mc_alaska_surf_20180919_150000_both/mc_data.206.0_64.5.npz')
# vpr.get_vmodel()

