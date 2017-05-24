# input the pp, if the pp is between 0 and 1, do joint inversion, otherwise , no inversion, just sample the a prior distribution
# do group or phase
# both sur and rf are involved
# input also vpvs for each group
# change the Model class, group class and the way the model to layered model
# water layer could also be considered
# optimized version of Weisen's MCMC code

from __future__ import absolute_import
import param, vmodel

import math
import sys
import string
import time
import random
import os
import numpy
import scipy

####################################################
def get_misfit(para, model, p, nn): 
	tmodel = model.para2mod(para)
	tmodel.update()
	tmodel.compute_rf()
	tmodel.compute_disp()
	tmodel.compute_misfit(p,nn)
	para = tmodel.mod2para(para)
	return para.L, para.misfit, tmodel
#####################################################

if len(sys.argv)<2:
    print "input [sur_flag] [dispp]/[dispg] [rf] [model] [fpara] [fcontrol]"
    print "flag: 1/2 phase/group; 3: phase + group"
    print "[fcontrol] contains 1. outdir; 2. pp; 3.monoc; 4. nn "
    print "pp: weighting between RF/SUR;  nn: dependence factor of RF;"
    sys.exit()

surflag = int(sys.argv[1]) # tell the code to read which sur.
if (surflag == 1 or surflag == 2):
	ninput = 7         # input only one type of sur
elif (surflag == 3):
	ninput = 8         # input both group and phase
else:
	print "input sur_flag is problematic!!!!!"
	sys.exit()
#####################################################

if (len(sys.argv) != ninput):
        print "input [sur_flag] [dispp]/[dispg] [rf] [model] [fpara] [fcontrol]"
        sys.exit()

################## initialize the model #############
model = vmodel.vprofile()
para0 = param.para()
#####################################################
for l1 in open(sys.argv[-1]):
	l2 = l1.rstrip().split()
outdir  = l2[0]
outdir  = outdir.rstrip("/")
pp      = float(l2[1])
monoc   = int(l2[2])
nn      = float(l2[3])
#####################################################

################### read all things  ################
if surflag ==1 or surflag==3: pfname = sys.argv[2]
if surflag ==2: gfname = sys.argv[2]
else: gfname = None
model.readdisp(pfname=pfname, gfname=gfname, surflag=surflag)
model.readrf(sys.argv[-4])
model.readmod(sys.argv[-3])
fpara = sys.argv[-2]
# model.readpara(fpara)
para0.read(fpara)
#####################################################
###################### do forward calculation #######
model.update()
model.compute_rf()
model.compute_disp()
model.compute_misfit(pp,nn)
#####################################################

tname = "MC." + sys.argv[-1]
model.write_model(tname, outdir)
print "read ok! and compute ok!"

#####################################################
oldL        = model.data.L
oldmisfit   = model.data.misfit
#####################################################
############## mod2para and para2mod ################
tmodel  = model
para    = model.mod2para(para0)

############store input model into para #############

######################### ff is to write the result ##
#############
if not os.path.isdir(outdir): os.makedirs(outdir)
ff = open("%s/MC.%s.out" % (outdir,sys.argv[-1]), "w")
ffb = open("%s/MC.%s.bin" % (outdir,sys.argv[-1]), "wb")
######################################################

start = time.time()
print "Original misfit: ", oldL, oldmisfit

#####################################################
i = 0;  # count new paras;
ii = 0; # count acceptance model
iii = 0;
jj = 0;
#####################################################
tname = "MC." + sys.argv[-1]
k = 1 # the key that controls the smapling
# test_i = 0
while ( k > 0 ):
    i = i + 1
    # test_i += 1
    if ( i > 10000 or ii > 2000 or time.time()-start > 3600.):
        k = 0
    if (math.fmod(i,500) ==0):
        print i, time.time()-start
    if ( math.fmod(i,1501) == 1500 ):
        para1   = para.new_para(0)
        ttmodel = model.para2mod(para1)
        ttmodel.update()
        iii     = 0
        while (ttmodel.goodmodel([0,1],[]) == 0):
            iii = iii + 1
            para1 = para.new_para(0)
            ttmodel = model.para2mod(para1)
            ttmodel.update()
        ttmodel.compute_rf()
        ttmodel.compute_disp()
        ttmodel.compute_misfit(pp,nn)
        oldL = ttmodel.data.L
        oldmisfit = ttmodel.data.misfit
        para = para1
        ii = ii + 1
        print para.parameter
        print "new para!!", oldL, oldmisfit
    ############################ do inversion#####################################
    # sample the posterior distribution ##########################################
    if (pp >= 0 and pp <=1):
        para1 = para.new_para(1)
        ttmodel = model.para2mod(para1)
        ttmodel.update()
        if (monoc == 1):
            newL = 0.;
            newmisfit = 100;
            if (ttmodel.goodmodel([0,1],[]) == 0):
                continue
        (newL,newmisfit,ttmodel) = get_misfit(para1,model,pp,nn)
        if (newL < oldL):
            prob = (oldL-newL)/oldL
            cvt = random.random()
            # reject
            if (cvt<prob):
                ff.write("-1 %d %d " % (i,ii))
                for j in range (para1.npara):
                    ff.write("%g " % para1.parameter[j])
                ff.write("%g %g %g %g %g %g %g\n" % (newL, newmisfit, ttmodel.data.rf.L, ttmodel.data.rf.misfit, ttmodel.data.disp.L,
                                                    ttmodel.data.disp.misfit, time.time()-start))
                ttmodel.writeb (para1, ffb,[-1,i,ii])
                continue
        ff.write("1 %d %d " % (i,ii))
        for j in range (para1.npara):
            ff.write("%g " % para1.parameter[j])
        ff.write("%g %g %g %g %g %g %g\n" % (newL,newmisfit,ttmodel.data.rf.L,ttmodel.data.rf.misfit,ttmodel.data.disp.L,ttmodel.data.disp.misfit,time.time()-start));
        print "accept!! ", i, ii, oldL, newL, ttmodel.data.rf.L, ttmodel.data.rf.misfit, ttmodel.data.disp.L, \
            ttmodel.data.disp.misfit, time.time()-start
        tname1 = tname + ".%d" % ii
        ttmodel.write_model(tname1, outdir)
        ttmodel.writeb (para1,ffb,[1,i,ii])
        para = para1
        oldL = newL
        oldmisfit = newmisfit
        ii = ii + 1
        continue
    else:
        if (monoc == 1):
            para1 = para.new_para(1)
            ttmodel = model.para2mod(para1)
            ttmodel.update()
            if (ttmodel.goodmodel([0,1],[]) == 0):
                continue
        else:
            para1 = para.new_para(0)
        ff.write("-2 %d 0 " % i)
        for j in range (para1.npara):
            ff.write("%g " % para1.parameter[j])
        ff.write("\n")
        para = para1
        continue
ffb.close()
ff.close()
