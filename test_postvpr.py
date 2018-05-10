import mcpost

vpr = mcpost.postvpr()
# vpr.read_inv_data('/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_results/mc_inv.TA.TOLK.npz')
# vpr.read_data('/scratch/summit/life9360/ALASKA_work/mc_inv_files/mc_results/mc_data.TA.TOLK.npz')

vpr.read_inv_data('../../mc_results_Miller/mc_inv.TA.TOLK.npz')
vpr.read_data('../../mc_results_Miller/mc_data.TA.TOLK.npz')

# vpr.read_inv_data('/work3/leon/mc_inv_files/mc_results_Miller/mc_inv.TA.TOLK.npz')
# vpr.read_data('/work3/leon/mc_inv_files/mc_results_Miller/mc_data.TA.TOLK.npz')

# vpr.read_inv_data('synthetic_working/mc_inv.CU.LF.npz')
# vpr.read_data('synthetic_working/mc_data.CU.LF.npz')