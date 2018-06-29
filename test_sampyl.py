import sampler_test_tool
import sampyl as smp
# from sampyl import np
import seaborn
import numpy as np

# 
# p   = np.loadtxt('synthetic_iso_inv/real_para.txt')
# 
# start = {u'x1': p[0], u'x2': p[1], u'x3': p[2], u'x4': p[3], u'x5': p[4], u'x6': p[5], u'x7': p[6],\
#          u'x8': p[7], u'x9': p[8], u'x10': p[9],u'x11': p[10], u'x12': p[11], u'x13': p[12]}
# 
# 
# nuts = smp.NUTS(sampler_test_tool.logp_sampyl, start)
# chain = nuts.sample(2)

# seaborn.jointplot(chain.x, chain.y, stat_func=None)


import sampyl as smp
import numpy as  np
import seaborn

icov = np.linalg.inv(np.array([[1., .8], [.8, 1.]]))
def logp(x, y):
    d = np.zeros(2)
    # try:
    #     d[0]    = x
    # except:
    #     d[0]    = x._value
    # try:
    #     d[1]    = y
    # except:
    #     d[1]    = y._value
    # d       = np.array([x, y])
    d[:]    = np.array([x, y])
    return -.5 * np.dot(np.dot(d, icov), d)

start = {u'x': 1., u'y': 1.}
nuts = smp.NUTS(logp, start)
chain = nuts.sample(1000)

seaborn.jointplot(chain.x, chain.y, stat_func=None)