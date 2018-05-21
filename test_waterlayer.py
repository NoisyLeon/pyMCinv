import vprofile
import modparam



vpr = vprofile.vprofile1d()
vpr.readdisp(infname='M3_disp.txt')
vpr.readmod(infname='water.mod')
vpr.getpara()

# vpr.mc_joint_inv_iso_mp(outdir='./water_hongda', wdisp=1., rffactor=40., pfx='M3', numbrun=150000, verbose=True)

