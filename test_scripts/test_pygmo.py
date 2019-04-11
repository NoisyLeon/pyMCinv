import sampler_test_tool
import numpy as np
from scipy import optimize
import vprofile, vmodel
import pygmo as pg
# from pygmo import *


prob    = pg.problem(sampler_test_tool.disp_func())
algo    = pg.algorithm(pg.de(gen = 1000))
algo.set_verbosity(10)
pop     = pg.population(prob, 20)
pop     = algo.evolve(pop)



# from pygmo import *
# algo = algorithm(de(gen = 500))
# algo.set_verbosity(100)
# prob = problem(rosenbrock(10))
# pop = population(prob, 20)
# pop = algo.evolve(pop) 