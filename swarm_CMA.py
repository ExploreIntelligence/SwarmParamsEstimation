import matplotlib
import numpy as np
import cma
import time
from utils import *

'''Show the CMA options related to the key words'''
opts = cma.CMAOptions('variable')
print(opts)
# time.sleep(10)
#opts.set('CMA_elitist', False)
#opts.set('popsize', 100)

'''Let's optimize a self-defined function'''
# x: [position=[x0, x1]  velocity=[x2, x3]  param_alfa = x4  param_d0 = x5 param_d1 = x6  param_K = x7]
x0 = [0.5, 0.5, 9.5, 0.5, 149, 101, 201, 0.5]   # initialization
#import random
#x0=[random.randint(0,5), random.randint(0,5), random.randint(5,10), random.randint(140,160), random.randint(190,210), random.randint(1,5)]
sigma0 = 0.5
low_bound = [-1, -1, 9, -1, 148, 98, 198, 0]
high_bound = [1, 1, 11, 1, 152, 102, 202, 2]
es = cma.CMAEvolutionStrategy(x0, sigma0, {'bounds': [low_bound, high_bound], 'CMA_elitist': False, 'popsize': 10})
generation_nums = 0
real_y_t = get_real_track()
while (not es.stop()) and generation_nums < 100:
    generation_nums = generation_nums+1
    fit, X = [], []
    # print(es.ask(number=1))       # number=popsize by default, let number=1 thus return only one sample
    # print(es.ask(1))              # number=1, return a list that contains one array
    # print(es.ask(1)[0])           # number=1, return a list, not an array
    while len(X) < es.popsize:      # popsize=10 by default
        curr_fit = None
        while curr_fit in (None, np.NaN):
            # the first element of ask(1), only one array
            x = es.ask(1)[0]
            curr_fit = cal_mse_loss(real_y_t, generate_track(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7]))
            # get the fitness function value
            # curr_fit = cma.ff.somenan(x, cma.ff.elli)
        X.append(x)
        fit.append(curr_fit)
    print(generation_nums)
    es.tell(X, fit)    # X and fit both have the size of popsize
    es.logger.add()
    es.disp()
es.result_pretty()
print(es.result_pretty())

