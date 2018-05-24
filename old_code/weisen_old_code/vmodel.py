
from __future__ import absolute_import
import numpy as np
import math
import param
from numba import jit, float32, int32, float64
import fast_surf, theo
import struct
import copy
import os
		
@jit(nopython=True)
def _compute_disp_1(newnlayer, thick, vp, vs, rho, qs):
	tvs 	= []
	tvp 	= []
	tqs 	= []
	trho 	= []
	tthick 	= []
	nn = 0
	for i in xrange (newnlayer-1):
		ttt = 0.
		if thick[i] < 0.1 and i > 1:
			ttt = thick[i]
			# print "no!!!!!", ttt, i
			continue
		tvs.append(vs[i] )
		tvp.append(vp[i] )
		trho.append(rho[i] )
		tthick.append(thick[i] + ttt)
		nn = nn + 1
	
	for i in xrange(nn):
		if qs[i] > 0.:
			tqs.append(1./qs[i])
		else:
			tqs.append(1./(qs[i]+0.001))
	return tvs, tvp, tqs, trho, tthick, nn
		
@jit(nopython=True)
def _update_1(i, value1, thick1, flag, flag0, vpvs, nlay):
	vs 		= []
	thick 	= []
	vp 		= []
	vpvsLst = []
	rho 	= []
	qs 		= []
	qp 		= []
	tdep = 0.
	for j in xrange (nlay):
		tvalue = value1[j]
		tthick = thick1[j]
		thick.append(tthick) 
		tdep = tdep + tthick
		if flag == 5:
			tvpvs 	= -1.
			tvs 	= 0.0
			tvp 	= tvalue
			trho 	= 1.02
			tqs 	= 10000.
			tqp 	= 57822.
		elif i==0 and flag0 != 5 or i==1 and flag0 == 5:    # if it is for sediments, vpvs = 2.
			tvpvs 	= vpvs
			tvs 	= tvalue
			tvp 	= tvpvs * tvs
			tqp 	= 160.
			tqs 	= 80.
			trho 	= 0.541 + 0.3601*tvp
		else:
			tvpvs 	= vpvs
			tvs 	= tvalue
			tvp 	= tvpvs * tvs
			if (tdep < 18.):
				tqp = 1400.  # PREM
				tqs = 600.
			if (tvp < 7.5):
				trho = 0.541 + 0.3601*tvp  # 
			else:
				trho = 3.35 # Kaban, M. K et al. (2003), Density of the continental roots: Compositional and thermal contributions, 
		vpvsLst.append(tvpvs)
		vs.append(tvs)
		vp.append(tvp)
		rho.append(trho)
		qs.append(tqs)
		qp.append(tqp)
	return vpvsLst, vs, vp, rho, qs, qp

@jit(nopython=True)
def _compute_misfit_1(arro, arrp, arrun):
	tempv1=0.
	for i in range (arro.size):
		tempv1 	= tempv1 + (arro[i] - arrp[i])**2/arrun[i]**2
	return tempv1

@jit(nopython=True)
def _compute_misfit_2(nrfo, rfo, rfn, unrfo, to, tn):
	tempv3 = 0.
	k = 0
	for i in range (nrfo):
		for j in range(tn.size):
			if to[i] == tn[j] and to[i] <= 10 and to[i] >= 0 :
				tempv3 = tempv3 + ((rfo[i] - rfn[j])**2/unrfo[i]**2)
				k = k+1
				break
	return tempv3, k

@jit
def _write_model_1(outfname, rfn, tn, rfo, to, unrfo):
	ff 	= open(outfname,"w")
	for i in range (len(rfn)):		
		if tn[i] <=10: 
			ff.write("%g %g %g %g %g\n" % (tn[i], rfn[i], to[i], rfo[i], unrfo[i]))
	ff.close()
	
@jit
def _write_model_2(outfname, per, vel, velo, unvelo):
	ff 	= open(outfname,"w")
	for i in range (len(vel)):		
		ff.write("%g %g %g %g \n" % (per[i], vel[i], velo[i], unvelo[i]))
	ff.close()
	
@jit
def _write_model_3(outfname, nlayer, vs, vp, rho, thick):
	ff = open(outfname,"w")
	dep = 0.
	for i in range (nlayer):
		ff.write("%g %g %g %g\n" % (dep, vs[i], vp[i], rho[i]))
		dep = dep + thick[i]
		ff.write("%g %g %g %g\n" % (dep, vs[i], vp[i], rho[i]))
	ff.close()

class vprofile(object):
	def __init__ (self):
		self.groups = param.groupLst()
		self.laym0 	= param.layermod()
		self.data 	= param.data()
		self.para	= param.para()
		self.cc 	= 2 # compute both
		self.flag 	= 0. # (no layered model ok, 1 if the layered model is ok)
	
	def readdisp(self, pfname=None, gfname=None, surflag=1):
		self.data.disp.readtxt(pfname=pfname, gfname=gfname, surflag=surflag)
	
	def readrf(self, fname):
		self.data.rf.readtxt(fname=fname)
		
	def readmod(self, fname):
		self.groups.readtxt(fname=fname)
		
	def readpara(self, fname):
		self.para.read(fname)
	def update(self):
		
		tnlay = 0

		vs 		= []
		thick 	= []
		vp 		= []
		vpvs 	= []
		rho 	= []
		qs 		= []
		qp 		= []
			
		tdep = 0.
		for i in xrange(self.groups.ngroup):
			self.groups[i].update()
			tvpvs, tvs, tvp, trho, tqs, tqp = _update_1(i, self.groups[i].value1, self.groups[i].thick1,
					self.groups[i].flag, self.groups[0].flag, self.groups[i].vpvs, self.groups[i].nlay)
			vpvs	+= tvpvs
			vs		+= tvs
			vp		+= tvp
			rho		+= trho
			qs		+= tqs
			qp		+= tqp
			thick	+= self.groups[i].thick1.tolist()
			tnlay 	= tnlay + self.groups[i].nlay
			
			# # # # for j in xrange (self.groups[i].nlay):
			# # # # 	tvalue = self.groups[i].value1[j]
			# # # # 	tthick = self.groups[i].thick1[j]
			# # # # 	thick.append(tthick) 
			# # # # 	tdep = tdep + tthick
			# # # # 	if self.groups[i].flag == 5:
			# # # # 		tvpvs 	= -1.
			# # # # 		tvs 	= 0.0
			# # # # 		tvp 	= tvalue
			# # # # 		trho 	= 1.02
			# # # # 		tqs 	= 10000.
			# # # # 		tqp 	= 57822.
			# # # # 	elif i==0 and self.groups[0].flag != 5 or i==1 and self.groups[0].flag == 5:    # if it is for sediments, vpvs = 2.
			# # # # 		tvpvs 	= self.groups[i].vpvs
			# # # # 		tvs 	= tvalue
			# # # # 		tvp 	= tvpvs * tvs
			# # # # 		tqp 	= 160.
			# # # # 		tqs 	= 80.
			# # # # 		trho 	= 0.541 + 0.3601*tvp
			# # # # 	else:
			# # # # 		tvpvs 	= self.groups[i].vpvs
			# # # # 		tvs 	= tvalue
			# # # # 		tvp 	= tvpvs * tvs
			# # # # 		if (tdep < 18.):
			# # # # 			tqp = 1400.  # PREM
			# # # # 			tqs = 600.
			# # # # 		if (tvp < 7.5):
			# # # # 			trho = 0.541 + 0.3601*tvp  # 
			# # # # 		else:
			# # # # 			trho = 3.35 # Kaban, M. K et al. (2003), Density of the continental roots: Compositional and thermal contributions, 
			# # # # 	vpvs.append(tvpvs)
			# # # # 	vs.append(tvs)
			# # # # 	vp.append(tvp)
			# # # # 	rho.append(trho)
			# # # # 	qs.append(tqs)
			# # # # 	qp.append(tqp)
			# # # # tnlay = tnlay + self.groups[i].nlay
		self.laym0.nlayer 	= tnlay
		self.laym0.vs 		= np.asarray(vs)
		self.laym0.thick 	= np.asarray(thick)
		self.laym0.vpvs 	= np.asarray(vpvs)
		self.laym0.vp 		= np.asarray(vp)
		self.laym0.rho 		= np.asarray(rho)
		self.laym0.qs 		= np.asarray(qs)
		self.laym0.qp 		= np.asarray(qp)
		self.flag = 1
		return

	def compute_disp(self):
		if self.flag != 1:
			print "no layerd model updated!!!"
			self.update()
		
		newnlayer = self.laym0.nlayer + 1 # add the last layers;
		tvs, tvp, tqs, trho, tthick, nn = _compute_disp_1(newnlayer, self.laym0.thick, self.laym0.vp,
														self.laym0.vs, self.laym0.rho, self.laym0.qs)
		tvs.append(self.laym0.vs[self.laym0.nlayer-1]-0.)
		tvp.append(self.laym0.vp[self.laym0.nlayer-1]-0.)
		tqs.append(tqs[nn-2]-0.)
		trho.append(self.laym0.rho[self.laym0.nlayer-1]-0.)
		tthick.append(0.)
		nn = nn + 1
		ur0 	= np.zeros(200)
		ul0 	= np.zeros(200)
		cr0 	= np.zeros(200)
		cl0 	= np.zeros(200)
		period	= np.zeros(200)
		nper 	= 0
		period1 = np.asarray(self.data.disp.period1[:])

		npper 	= self.data.disp.npper
		ngper 	= self.data.disp.ngper
		nper	= len(period1)
		period[:nper] = period1
		(ur0,ul0,cr0,cl0) = fast_surf.fast_surf(2, tvp, tvs, trho,tthick,tqs,period,nper,nn)
		
		if npper > 0: self.data.disp.pvel = cr0[:npper]
		if ngper > 0: self.data.disp.gvel = ur0[:ngper]
		return

	def compute_rf(self):
		if (self.flag != 1):
			print "no layerd model updated!!!"
			self.update()
		if self.laym0.nlayer < 100: newnlayer = self.laym0.nlayer + 1 # add the last layers;
		else: newnlayer = 100

		tvs 	= np.zeros(100)
		tvpvs 	= np.zeros(100)
		tqs 	= np.zeros(100)
		tqp 	= np.zeros(100)
		trho 	= np.zeros(100)
		tthick 	= np.zeros(100)
		
		index 	= self.laym0.vs>0
		tvs1 	= self.laym0.vs[index]; 	tvs[:tvs1.size] = tvs1; nl = tvs1.size
		tvpvs1 	= self.laym0.vpvs[index]; 	tvpvs[:nl] 		= tvpvs1
		tqs1 	= self.laym0.qs[index]; 	tqs[:nl] 		= tqs1
		tqp1 	= self.laym0.qp[index]; 	tqp[:nl] 		= tqp1
		tthick1 = self.laym0.thick[index]; 	tthick[:nl] 	= tthick1
		tvs[nl] = (tvs[nl-1] -0.)
		tvpvs[nl] = (tvpvs[nl-1] -0.)
		tqs[nl] = (tqs[nl-1] -0.)
		tqp[nl] = (tqp[nl-1] -0.)
		tthick[nl] = 0.
		nl += 1
		# # 
		# # nl = 0
		# # for i in range (newnlayer-1):
		# # 	if (model1.laym0.vs[i] > 0.):
		# # 		tvs[nl] = model1.laym0.vs[i] - 0.
		# # 		tvpvs[nl] = model1.laym0.vpvs[i] - 0.
		# # 		tqs[nl] = model1.laym0.qs[i] - 0.
		# # 		tqp[nl] = model1.laym0.qp[i] - 0.
		# # 		tthick[nl] = model1.laym0.thick[nl] - 0.
		# # 		nl = nl + 1
		nn 	= 1000
		rx 	= np.zeros(nn)
		slow= 0.06
		din = 180.*math.asin(tvs[nl-1]*tvpvs[nl-1]*slow)/math.pi
		rt 	= self.data.rf.rt
		newnlayer 	= nl
		rx 			= theo.theo(newnlayer,tvs,tthick,tvpvs,tqp,tqs,rt,din,2.5,0.005,0,nn)
		# # # newt = []
		# # # newrf = []
		# # # for i in range(nn):
		# # # 	tempt = i*1./model1.data.rf.rt
		# # # 	newt.append(tempt)
		# # # 	newrf.append(rx[i]-0.)
		self.data.rf.tn = np.arange(nn, dtype=float)/self.data.rf.rt
		self.data.rf.rfn= rx[:nn]
		return

	def compute_misfit(self, inp=1, nn=40):
		"""
		nn	- relavive weight for receiver function
		"""
		dmisfit = 0.
		ndisp 	= 0
		dL 		= 0.
		################################################### DISP ###############################################
		tS 		= 100
		tempv1	= 0.
		if self.data.disp.fphase > 0. and self.data.disp.npper > 0:
			tempv1=_compute_misfit_1(self.data.disp.pvelo , self.data.disp.pvel, self.data.disp.unpvelo)
			if tempv1 > 0:
				tmisfit = math.sqrt(tempv1/self.data.disp.npper)
				tS 		= tempv1
				if tS > +50.: tS = math.sqrt(tS*50)
				tL = math.exp(-0.5 * tS)
			else:
				print "strange misfit calculation!!!! be careful!!! "
				tmisfit = -1.
				tL 		= -1.
		else:
			tmisfit = -1.
			tL 		= -1.
		self.data.disp.pmisfit = tmisfit
		self.data.disp.pL = tL
		ndisp += self.data.disp.npper
		###########################################################################################################
		tS 		= 100.
		tempv2	= 0.
		if (self.data.disp.fgroup > 0 and self.data.disp.ngper > 0):
			tempv2=_compute_misfit_1(self.data.disp.gvelo , self.data.disp.gvel, self.data.disp.ungvelo)
			if (tempv2 > 0):
				tmisfit = math.sqrt(tempv2/self.data.disp.ngper)
				tS 		= tempv2
				if tS > 50.: tS = math.sqrt(tS*50.)
				tL = math.exp(-0.5 * tS)
			else:
				print "stange misfit calculation!!!! be careful!!! "
				tmisfit = -1.
				tL 		= -1.
		else:
			tmisfit = -1.
			tL 		= -1.
		self.data.disp.gmisfit = tmisfit
		self.data.disp.gL = tL
		##########################################################################################################
		if self.data.disp.npper > 0 or self.data.disp.ngper > 0:
			dmisfit = math.sqrt((tempv1+tempv2)/ndisp)
			tS = tempv1 + tempv2
			if tS > 50.: tS = math.sqrt(tS*50.)
			if tS > 50.: tS = math.sqrt(tS*50.)
			dL = math.exp(-0.5 * tS)
		else:		
			dmisfit = 0.
			dL 		= 0.
		self.data.disp.L 		= dL
		self.data.disp.misfit 	= dmisfit
		##########################################################################################################
		tp 		= 0.
		tempv3, k  = _compute_misfit_2(self.data.rf.nrfo, self.data.rf.rfo, self.data.rf.rfn,
									self.data.rf.unrfo, self.data.rf.to, self.data.rf.tn)
		# # # k=0
		# # # tempv3=0.
		# # # for i in range (self.data.rf.nrfo):
		# # # 	for j in range(len(self.data.rf.tn)):
		# # # 		if (self.data.rf.to[i] == self.data.rf.tn[j] and self.data.rf.to[i] <= 10 and self.data.rf.to[i] >= 0 ):
		# # # 			tempv3 = tempv3 + ((self.data.rf.rfo[i] - self.data.rf.rfn[j])**2/self.data.rf.unrfo[i]**2)
		# # # 			k = k+1
		# # # 			break
		tmisfit2 = math.sqrt(tempv3/k)
		tS = tempv3/nn
		if (tS> 50.): tS = math.sqrt(tS*50.)
		tL2 = math.exp(-0.5 * tS)
		self.data.rf.L = tL2
		self.data.rf.misfit = tmisfit2	
#################### combination misfit and L########################################################################
		self.data.L 	= (dL**inp)*(tL2**(1-inp))
		self.data.misfit= inp*dmisfit + (1-inp)*tmisfit2 
#####################################################################################################################
		return

	def write_model(self, outfpfx, outdir):
		if not os.path.isdir(outdir): os.makedirs(outdir)
		namerf 	= "%s/%s.rf" % (outdir, outfpfx)
		if self.data.rf.tn.size>0:
			_write_model_1(namerf, self.data.rf.rfn, self.data.rf.tn, self.data.rf.rfo, self.data.rf.to, self.data.rf.unrfo)
			# # # ff 		= open(namerf,"w")
			# # # for i in range (len(self.data.rf.rfn)):		
			# # # 	if len(self.data.rf.tn) > 0 and self.data.rf.tn[i] <=10: 
			# # # 		ff.write("%g %g %g %g %g\n" % (self.data.rf.tn[i], self.data.rf.rfn[i], self.data.rf.to[i], self.data.rf.rfo[i], self.data.rf.unrfo[i]))
			# # # ff.close()
		
		if self.data.disp.npper > 0:
			namedisp = "%s/%s.p.disp" % (outdir, outfpfx)
			_write_model_2(namedisp, self.data.disp.pper, self.data.disp.pvel, self.data.disp.pvelo, self.data.disp.unpvelo)
			# # # ff = open(namedisp,"w")
			# # # for i in range (self.data.disp.npper):
			# # # 	ff.write("%g %g %g %g\n" % (self.data.disp.pper[i],self.data.disp.pvel[i],self.data.disp.pvelo[i],self.data.disp.unpvelo[i]));
			# # # ff.close()
			
		if self.data.disp.ngper > 0:
			namedisp = "%s/%s.g.disp" % (outdir, outfpfx)
			_write_model_2(namedisp, self.data.disp.gper, self.data.disp.gvel, self.data.disp.gvelo, self.data.disp.ungvelo)
			
			# # # ff = open(namedisp,"w")
			# # # for i in range (self.data.disp.ngper):
			# # # 		ff.write("%g %g %g %g\n" % (self.data.disp.gper[i],self.data.disp.gvel[i],self.data.disp.gvelo[i],self.data.disp.ungvelo[i]))
			# # # ff.close()
		
		namemod = "%s/%s.mod" % (outdir, outfpfx)
		_write_model_3(namemod, self.laym0.nlayer, self.laym0.vs, self.laym0.vp, self.laym0.rho, self.laym0.thick)
		
		# # # ff = open(namemod,"w")
		# # # dep = 0.
		# # # for i in range (self.laym0.nlayer):
		# # # 	ff.write("%g %g %g %g\n" % (dep, self.laym0.vs[i],self.laym0.vp[i],self.laym0.rho[i]))
		# # # 	dep = dep + self.laym0.thick[i]
		# # # 	ff.write("%g %g %g %g\n" % (dep, self.laym0.vs[i],self.laym0.vp[i],self.laym0.rho[i]))
		# # # ff.close()
		
		return
	
	def writeb (self, para1, ffb, ids):
		ngroup = len(self.groups)
		data = struct.pack('iii',ids[0],ids[1],ids[2])
		ffb.write(data)
		data = struct.pack('dddd', self.data.L, self.data.misfit, (self.data.rf.misfit), (self.data.disp.misfit))
		ffb.write(data)

		data = struct.pack('iii',para1.npara, ngroup, self.laym0.nlayer)
		ffb.write(data)

		for i in range (para1.npara):
			data = struct.pack('d',(para1.parameter[i]))
			ffb.write(data)

		for i in range (ngroup):
			data = struct.pack('d',(self.groups[i].thickness))
			ffb.write(data)
		for i in range (self.laym0.nlayer):
			data = struct.pack('dddd',(self.laym0.thick[i]),(self.laym0.vs[i]),(self.laym0.vp[i]),(self.laym0.rho[i]))
			ffb.write(data)
		return

	def goodmodel(self, gl0, gl1):
		ngroup = len(self.groups)
		for i in range (ngroup-1):
			if (self.groups[i+1].value1[0] < self.groups[i].value1[-1]):
				return 0
		#### monotonic change ##################################
		for j in (gl0):
			for i in range (self.groups[j].nlay-1):
				if (self.groups[j].value1[i+1] < self.groups[j].value1[i]):
					return 0
		#########################################################
		#### gradient check #####################################
		for j in (gl1):
			if (self.groups[j].value1[1] < self.groups[j].value1[0]):
				return 0
		#########################################################
		return 1
	
	def mod2para(self, para=None):
		if isinstance(para, param.para):
			para1 = copy.deepcopy(para)
		else:
			para1 = copy.deepcopy(self.para)
			para  = self.para
		para1.parameter = []
		para1.L 		= self.data.L
		para1.misfit 	= self.data.misfit
		for i in range (para.npara):
			ng = para.para0[i][4]
			if para.para0[i][0] == 0:	     # value (vs/Bs)
				nv = para.para0[i][5]
				tv = self.groups[ng].value[nv]
			elif para.para0[i][0] == 1:           # thickness of group
				tv = self.groups[ng].thickness
			elif para.para0[i][0] == -1:          # vpvs value;
				tv = self.groups[ng].vpvs
			else:
				print "WTF???? "
				sys.exit()
			para1.parameter.append(tv - 0.);
			if (para.flag != 1):
				tstep = para.para0[i][3];
				if (para.para0[i][1] == 1):
					tmin = tv - para.para0[i][2];
					tmax = tv + para.para0[i][2];
				else:
					tmin = tv - tv*para.para0[i][2]/100.
					tmax = tv + tv*para.para0[i][2]/100.
				tmin = max (0.,tmin);
				tmax = max (tmin + 0.0001, tmax);
	
				if (para.para0[i][0] == 0 and i == 0 and para.para0[i][5] == 0): # if it is the upper sedi:
					tmin = max (0.2, tmin)
					tmax = max (0.5, tmax)
				para1.space1.append([tmin,tmax,tstep])
				para1.flag = 1
		self.para = para1
		return para1
	
	def para2mod(self, para=None):
	############################## final version, it'll od the convention systematically #####
	# given a para,
	# transfer value from para.parameter to models
	##################################################################################
		if not isinstance(para, param.para):
			para  = self.para
		model1 	= copy.deepcopy(self)
		for i in range (para.npara):
			newv = para.parameter[i]
			ng = para.para0[i][4]
			if para.para0[i][0] == 0:  # value
				nv = para.para0[i][5]
				model1.groups[ng].value[nv] = newv
			elif para.para0[i][0] == 1: # thickness of group
				model1.groups[ng].thickness = newv
			elif para.para0[i][0] == -1:          # vpvs value;
				model1.groups[ng].vpvs = newv
			else:
				print "WTF???? "
				sys.exit()
		return model1
	
