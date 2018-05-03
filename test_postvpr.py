import mcpost

vpr = mcpost.postvpr()
vpr.read_inv_data('workingdir/mc_inv.AK.WRH.npz')
vpr.read_data('workingdir/mc_data.AK.WRH.npz')