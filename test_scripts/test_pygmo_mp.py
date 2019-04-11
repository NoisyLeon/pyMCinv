import sampler_test_tool
import numpy as np
from scipy import optimize
import vprofile, vmodel
import pygmo as pg
# from pygmo import *


prob    = pg.problem(sampler_test_tool.disp_func())

algo    = pg.algorithm(pg.de(gen = 1000))
# 
# algo.set_verbosity(10)

# pop     = pg.population(prob, 20)

# pop     = algo.evolve(pop)

# isl = pg.island(algo = pg.de(gen = 1000), prob = sampler_test_tool.disp_func(), size=200, udi=pg.ipyparallel_island())
archi = pg.archipelago(n=12, algo=algo, prob=prob, pop_size=20)
# from pygmo import *
# algo = algorithm(de(gen = 500))
# algo.set_verbosity(100)
# prob = problem(rosenbrock(10))
# pop = population(prob, 20)
# pop = algo.evolve(pop)

# from pygmo import *
# archi = archipelago(n = 16, algo = de(), prob = rosenbrock(10), pop_size = 20, seed = 32)
# archi.evolve()
# archi 