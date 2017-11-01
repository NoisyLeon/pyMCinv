import os

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