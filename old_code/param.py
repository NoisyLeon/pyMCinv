
from __future__ import absolute_import
from numba import jit, float32, int32
import numpy as np
import bspline_basis
import os.path
import sys
import random
import copy
		
class layermod(object):
    def __init__(self, nlayer=0):
        self.nlayer	= nlayer
        self.vs 	= np.array([])
        self.vpvs 	= np.array([])
        self.vp 	= np.array([])
        self.rho 	= np.array([])   
        self.qs 	= np.array([])
        self.qp 	= np.array([])
        self.thick  = np.array([])

@jit
def _aux_update2(bspl_basis, npts, nBs, value, thickness):
    value0 = np.zeros(npts)
    thick0 = np.zeros(npts)
    for i in xrange (npts):
        tvalue 	= 0.
        for j in xrange (nBs):
            tvalue = tvalue + bspl_basis[j][i] * value[j]
        value0[i] 	= tvalue
        thick0[i]	= thickness/npts
    return value0, thick0

@jit
def _aux_rf_stretch_1(ntt, nd1, ndz):
    nseis=[]
    for i in range (ndz):
        k = int(ntt[i])
        if k<len(nd1):
            nseis.append(nd1[k])
        else: break
    return np.array(nseis)

@jit
def _aux_rf_stretch_2(n1, vtt, nseis, dt):
    t2 	= []
    d2	= []
    for i in range(n1):
        tempt = i*dt
        if (tempt > 10): break
        t2.append(tempt)
        tempd = 0.
        for j in range (nseis.size):
            if (vtt[j] <= tempt and vtt[j+1]>tempt):
                tempd = nseis[j] + (nseis[j+1]-nseis[j])*(tempt-vtt[j])/(vtt[j+1]-vtt[j])
                break
        d2.append(tempd)
    return np.array(t2), np.array(d2)

@jit
def _new_para_1(npara, space1):
    parameter = np.zeros(npara)
    for i in xrange (npara):
        newv = np.random.uniform(space1[i][0], space1[i][1])
        parameter[i] = newv - 0.
    return parameter


@jit
def _new_para_2(npara, parameter, space1):
    outpara = np.zeros(npara)
    for i in xrange (npara):
        parai 	= parameter[i]
        tstep 	= space1[i][2]
        flag 	= 0
        j		= 0
        while (flag < 1 and j<10000): 
            newv = random.gauss(parai, tstep)
            if (newv >= space1[i][0] and newv <= space1[i][1]):
                flag = 2
            j +=1
        outpara[i] = newv - 0.
    return outpara

class group(object):
    def __init__(self, numbp=0, flagBs=-1, flag=-1, thickness=0., nlay=20, vpvs=1.75):
        self.numbp 		= numbp
        self.flagBs 	= flagBs # flags for B-splines. once it is converted once, it will becomes 1, indicating that Bsplines are stored and will never changed
        self.flag 		= flag # indicators for layer(1)/B-splines(2/3)/gradient layer(4)/water(5);
        self.thickness 	= thickness # total thickness for layer/B-splines;
        self.ratio 		= np.array([]) # []
        self.value 		= np.array([]) # []
        # self.ratio 		= []
        # self.value 		= []
        self.nlay 		= nlay # put converted nlay in it; initial value is 20;
        self.value1 	= np.zeros(numbp) # [] put converted velocity in it;
        self.thick1 	= np.zeros(numbp) # put converted thickness in it;
        self.Bsplines 	= np.array([])
        self.vpvs 		= vpvs # In each group the vpvs is set to be 1.75 as a default
    
    def updateBs(self):
        """
        Update cubic B-spline
        """
        if (self.thickness >= 150): self.nlay = 60
        elif self.thickness < 10: self.nlay = 5
        elif self.thickness < 20: self.nlay = 10
        else: self.nlay = 30
        nBs = self.numbp
        if nBs < 4: order = 3
        else: order = 4
        if self.flag == 2:		
            self.Bsplines 	= bspline_basis.Bspline(nBs, order, 0, self.thickness, 2., self.nlay);
            self.flagBs 	= 1
        return
    
    def update1(self):
        if self.flag == 1:
            self.nlay 	= self.numbp
            self.thick1 = self.thickness*self.ratio
            self.value1 = self.value.copy()
        else: print "something wrong happen!! update1"
        
    
    def update2(self):
        if (self.flag == 2 ):
            if (self.flagBs != 1): self.updateBs()
            npts = self.nlay
            value0, thick0 = _aux_update2(self.Bsplines, npts, self.numbp, self.value, self.thickness)
            self.thick1 = thick0
            self.value1 = value0
        else: print "something wrong happen!!!! update2"
    # 
    def update3 (self): # update group for gradient layer
        if self.flag == 4:
            tn 		= 4
            if self.thickness >= 20.: tn = 20
            if self.thickness > 10. and self.thickness < 20.: tn = int(self.thickness/1.)
            if self.thickness > 2. and self.thickness <= 10.: tn = int(self.thickness/0.5)
            if self.thickness < 0.5: tn = 2
            tth 	= self.thickness/float(tn)
            dt 		= (self.value[1] - self.value[0])/(tn - 1.)
            tvel 	= self.value[0] + np.arange(tn) *dt
            tthick 	= np.ones(tn)*tth
            self.thick1 = tthick
            self.value1 = tvel
            self.nlay   = tn
        else: print "something wrong happen!!!! update3"
    
    def update4 (self): # update group for water layer!!!
        if self.flag != 5: print "something wrong happen!!!!! update4";
        else:
            tn = 1
            self.value1 = np.array([self.value[0]])
            self.thick1 = np.array([self.thickness])
            self.nlay 	= tn
        return
    # 
    def update(self):
        if self.flag == 1: self.update1()
        elif self.flag == 2: self.update2()
        elif self.flag == 4: self.update3()
        elif self.flag == 5: self.update4() # water layer!!!
        else: print "no other choise now"
        return

class groupLst(object):
    def __init__(self, groups=None):
        self.groups=[]
        if isinstance(groups, group):
            groups = [groups]
        if groups:
            self.groups.extend(groups)
        self.ngroup = 0
    
    def __add__(self, other):
        """
        Add two groupLst with self += other.
        """
        if isinstance(other, group):
            other = groupLst([other])
        if not isinstance(other, groupLst):
            raise TypeError
        groups = self.groups + other.groups
        return self.__class__(groups=groups)
    
    def __len__(self):
        """
        Return the number of groups in the groupLst object.
        """
        return len(self.groups)
    
    def __getitem__(self, index):
        """
        __getitem__ method of groupLst objects.
        :return: group objects
        """
        if isinstance(index, slice):
            return self.__class__(groups=self.groups.__getitem__(index))
        else:
            return self.groups.__getitem__(index)
    
    def append(self, ingroup):
        """
        Append a single group object to the current groupLst object.
        """
        if isinstance(ingroup, group):
            self.groups.append(ingroup)
        else:
            msg = 'Append only supports a single group object as an argument.'
            raise TypeError(msg)
        return self
    
    def readtxt(self, fname):
        """
        Read mod groups
        column 1: id
        column 2: flag  - layer(1)/B-splines(2/3)/gradient layer(4)/water(5)
        column 3: thickness
        column 4: number of parameters for the group
        column 5 - 4+tnp : value
        column 4+tnp - 4+2*tnp: ratio
        column -1: vpvs ratio
        """
        i = 0
        for l1 in open(fname,"r"):
            i = i+1
        self.ngroup = i; # number of groups in input file
        print "Number of groups: %d " % self.ngroup
                
        for l1 in open(fname,"r"):
            l1 			= l1.rstrip()
            l2 			= l1.split()
            iid 		= int(l2[0])
            flag		= int(l2[1])
            thickness	= float(l2[2])
            tnp 		= int(l2[3]) # number of parameters
            numbp 		= tnp
            self.groups.append(group(numbp=tnp, flag=flag, thickness=thickness))
            if (int(l2[1]) == 5):  # water layer			
                if (tnp != 1):
                    print " water layer! only 1 values for Vp"
                    return 0
            if (int(l2[1]) == 4):
                if (tnp != 2):
                    print "only 2 values ok for gradient!!! and 1 value for vpvs"
                    print tnp
                    return 0
            if ( (int(l2[1])==1 and len(l2) != 4+2*tnp + 1) or (int(l2[1]) == 2 and len(l2) != 4+tnp + 1) ): # tnp parameters (+ tnp ratio for layered model) + 1 vpvs parameter
                print "wrong input !!!"
                return 0
            for i in range (tnp):
                self.groups[iid].value = np.append( self.groups[iid].value, float(l2[4+i]))
                # self.groups[iid].value.append(float(l2[4+i]))
                if (int(l2[1]) ==1):  # type 1 layer
                    self.groups[iid].ratio = np.append( self.groups[iid].ratio, float(l2[4+tnp+i]));
                    # self.groups[iid].ratio.append( float(l2[4+tnp+i]));
            self.groups[iid].vpvs = (float(l2[-1]))-0.
        if (self.ngroup >1):
            print "flag0: ", self.groups[0].flag, "flag1: ",self.groups[1].flag
        return 1

class rf(object):
    """Receiver function
    """
    def __init__(self):
        self.nrfo	= 0
        self.rt 	= 0  # rate
        self.to 	= np.array([])
        self.rfo 	= np.array([])
        self.unrfo	= np.array([])
                
        self.tn 	= np.array([])
        self.rfn 	= np.array([])
        self.tnn	= np.array([])
        self.rfnn 	= np.array([])
        self.L 		= 0.
        self.misfit = 0.
    
    def stretch(self):
        # input is rf.tn, rf.rfn;
        # output is rf.tnn, rf.rfnn;
        slowi 	= 0.06
        nd1 	= self.rfn[:]
        dzi 	= 0.5
        dzmax 	= 240.
        dZ 		= np.arange(int(dzmax/dzi))*dzi
        Rv 		= 1.73
        dt 		= self.tn[1]-self.tn[0]
        ndz 	= dZ.size
        zthk    = np.ones(ndz)*dzi
        cpv		= 6.4 * np.ones(ndz)
        cpv[np.arange(ndz)*dzi > 40] =7.8
        csv		= cpv/Rv
        pvel	= cpv
        svel	= csv
        sv2		= csv**(-2)
        pv2		= (csv*Rv)**(-2)
        cc		= dzi*(np.sqrt(sv2)-np.sqrt(pv2))
        vtt 	= np.cumsum(cc)
        vtt[0]	= 0.
        p2		= (slowi**2)*np.ones(ndz)
        cc		= (np.sqrt(sv2-p2) - np.sqrt(pv2-p2))*dzi
        mtt 	= np.cumsum(cc)
        ntt 	= np.round(mtt/dt)
        ntt[0] 	= 0
        nseis 	= _aux_rf_stretch_1(ntt, nd1, ndz)
        time1 = vtt[nseis.size - 1] # total time after stretch
        n1 = int(time1/dt)
        t2, d2 	= _aux_rf_stretch_2(n1, vtt, nseis, dt)
        self.tnn 	= t2
        self.rfnn 	= d2
        return
    
    ##############################################################
    #	read file "name" and store infor into model.data.rf
    #	return 0 if wrong, return 1 if read successfully
    def readtxt(self, fname):		
        if self.nrfo > 0:
            print "already some rf data stored!!!\n"
            print "exit"
            return 0
        inArr 		= np.loadtxt(fname)
        self.to 	= inArr[:,0]
        self.rfo	= inArr[:,1]
        self.nrfo  	= self.to.size
        try: self.unrfo	= inArr[:,2]
        except: self.unrfo = np.ones(self.nrfo)*0.1
        self.rt = int(1./(self.to[1] - self.to[0]));
        return 1


class disp(object):
    "store dispersion curve"
    def __init__(self):
        self.npper 	= 0
        self.pper 	= np.array([])
        self.pvelo 	= np.array([])
        # self.pvel = []
        self.unpvelo= np.array([])
    
        self.ngper 	= 0
        self.gper 	= np.array([])
        self.gvelo 	= np.array([])
        # self.gvel = []
        self.ungvelo= np.array([])
    
        self.period1 = []
    
        self.fphase = False
        self.fgroup = False
    
        self.pL = 0.
        self.pmisfit = 0.
    
        self.gL = 0.
        self.gmisfit = 0.
    
        self.L = 0.
        self.misfit = 0.
        
    
    def readtxt(self, pfname=None, gfname=None, surflag=1):
        if self.npper > 0 or self.ngper > 0:
            print "already some disp data stored!!!\n"
            print "exit"
            return 0
    
        if self.fgroup or self.fphase :
            print "already some disp data stored!!!\n"
            print "exit"
            return 0
    
        if surflag == 1 or surflag ==3: # phase/phase + group
            if pfname == None:
                raise ValueError('Need to specify phase velocity file name!')
            print 'read phase!!'
            inArr 		= np.loadtxt(pfname)
            self.pper 	= inArr[:,0]
            self.pvelo	= inArr[:,1]
            self.npper  = self.pper.size
            try: self.unpvelo= inArr[:,2]
            except: self.unpvelo = np.zeros(self.npper)
            self.fphase = True
    
        if (surflag == 2):
            if gfname == None:
                raise ValueError('Need to specify group velocity file name!')
            print 'read group!!'
            inArr 		= np.loadtxt(gfname)
            self.gper 	= inArr[:,0]
            self.gvelo	= inArr[:,1]
            self.ngper  = self.pper.size
            try: self.ungvelo= inArr[:,2]
            except: self.ungvelo = np.zeros(self.ngper)
            self.gphase = True
            
        if surflag ==3: # phase/phase + group
            if gfname == None:
                raise ValueError('Need to specify group velocity file name!')
            print 'read group!!'
            inArr 		= np.loadtxt(gfname)
            self.gper 	= inArr[:,0]
            self.gvelo	= inArr[:,1]
            self.ungvelo= inArr[:,2]
            self.ngper  = self.pper.size
            try: self.ungvelo= inArr[:,2]
            except: self.ungvelo = np.zeros(self.ngper)
            self.gphase = True
        period1 = list(set.union(set(self.pper), set(self.gper)))
        period1 = sorted(period1)
        self.period1 = period1[:]
        # print model.data.disp.period1
        return 1
# 
class para(object):
	def __init__(self):
		self.npara = 0
		self.para0 = []
		self.parameter = []
		self.L = 0.
		self.misfit = 0.
		self.space1 = []
#		self.pflag = [];  # pflag : 0: velocity  in crust/mantle; flag == 1: moho depth; flag == 2: sediments depth
		self.flag = 0.

	def read(self, fname):
		if not os.path.exists(fname):
			print "cannot read para ",fname
			sys.exit()
		k = 0
		for l1 in open(fname,"r"):
			tt = [];
			l2 = l1.rstrip().split();
			k = k + 1;
			ts = [];
			for i in range(len(l2)):
				if (i==2 or i == 3):
					ts.append(float(l2[i]));
				else:
					ts.append(int(float(l2[i])));
			self.para0.append(ts);
		print "read para over!"
		self.npara = k

############################# creat new para.parameter #######################
#    generate new value from para.space1
#    put the new value into para.parameter
#    use sflag to control the generation
#    sflag == 0: random new value in space
#    sflag == 1: gaussian random near the input value with a step;
###############################################################################
	def new_para(self, pflag):  # creat new para.parameter
		para = copy.deepcopy(self)
		if pflag == 0:
			para.parameter=_new_para_1(para.npara, para.space1)
		else:
			parameter       = _new_para_2(para.npara, np.asarray(para.parameter), np.asarray(para.space1))
			para.parameter  = parameter
		return para

class data(object):
	"store data"
	def __init__(self):
		self.rf 	= rf()
		self.disp 	= disp()
		self.p 		= 0.
		self.L 		= 0.
		self.misfit = 0.
