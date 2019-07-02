import multiprocessing
import numpy as np
import time
import random

def do_calculation(data):
    rand=np.random.rand()
    # rand=random.random()
    print data, rand
    time.sleep(rand)
    return data * 2

if __name__ == '__main__':
    pool_size = multiprocessing.cpu_count() * 2
    pool = multiprocessing.Pool(processes=pool_size)

    inputs = list(range(10))
    print 'Input   :', inputs

    pool_outputs = pool.map(do_calculation, inputs)
    print 'Pool    :', pool_outputs