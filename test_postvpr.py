import mcpost

vpr = mcpost.postvpr()
vpr.read_inv_data('workingdir/mc_inv.AK.CAST.npz')
vpr.read_data('workingdir/mc_data.AK.CAST.npz')