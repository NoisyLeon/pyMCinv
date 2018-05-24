import param
import numpy as np

para = param.para()
para.read('in.para')
para.space1=np.random.rand(13, 3)
para.parameter=np.random.rand(13)
para.new_para(0)