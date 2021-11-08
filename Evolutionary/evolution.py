# Pytania:
# 1. Czy w wyrażeniu exp{T*N(0, 1)} nie dałoby się wciągnąć by było exp{N(0, T)}?
# 2. W samoadaptacji, ciągle otrzymuję rozbieżność zaczynając od punktu (-100) czy to normalne?
# 3. Samoadaptacja ciągle utyka w lokalnym minimum na 30 kroku - zwiększanie sigmy niewiele daje - to źle (ok 600)?

# matplotlib contour

# Wstęp
# Opis eksperymentów
# Wyniki i dyskusja
# Wnioski

# zależności od:
# 1. sigmy
# 2. lambdy
# 3. metody

# Po 5 powtórzeń z różnymi ziarnami

from math import *
import numpy as np
import sys


def spheric(Alpha):
    def fun(X):
        """
        Parameters:
        alpha - coefficiant
        X - list of n values to count function from

        Result:
        sum from i=1 to i=n of alpha^((i-1)/(n-1))*x_i^2
        """
        sum = 0
        n = len(X)
        for i in range(1, n+1):
            sum += Alpha**((i-1)/(n-1))*X[i-1]**2
        return sum
    return fun


def happy_cat(Dimensions):
    def func(X):
        """
        Parameters:
        Dimensions - number of dimensions
        X - values of each dimension
        """
        try:
            m = sqrt(sum(i**2 for i in X))
            summ = ((m**2 - Dimensions)**2)**(1/8)
            summ += (1/Dimensions)*(0.5*m**2 + sum(X)) + 0.5
            return summ
        except:
            return sys.maxsize
    return func


class EvolutionStrategy(object):

    def __init__(self, func, center, sigma, parameters) -> None:
        self.param = parameters
        self.function = func
        self.generation = 0
        if center:
            self.centroid = center
        else:
            self.centroid = np.random.uniform(low=parameters['limits'][0], high=parameters['limits'][1], size=parameters['dimension'])
        self.cpopulation = []
        self.sigma = sigma
        self.shift = self.param['prec'] + 1
        self.logs = []              # stores logs of each generation as (centroid, sigma)
        self.bsf = (([], sys.maxsize))               # stores best so far organism


    def exe(self):
        while(self.generation < self.param['max_iter'] and self.shift > self.param['prec']):
            self.new_population()           # using sigma & centroid, generate lambda points
            self.selection()                # sort vis function and kill all except first mu
            self.recombination()            # new centroid & sigma, logging, shift & generation actualisation

        return


    def new_population(self):
        """
            Each point consists of a tuple(). It is defined
            by its coefficiants and used sigma.
        """
        if self.param['self-adaptation']:   # generate using self-adaptation
            self.cpopulation = [None] * self.param['lambda']
            for i in range(self.param['lambda']):
                altered_sigma = self.sigma * exp((1/sqrt(self.param['dimension'])) * np.random.normal(0, 1))
                self.cpopulation[i] = (np.random.normal(self.centroid, altered_sigma), altered_sigma)
        else:                               # generate using gauss
            self.cpopulation = [(np.random.normal(self.centroid, self.sigma), self.sigma) for _ in range(self.param['lambda'])]
        return


    def selection(self):
        self.cpopulation = [(x, self.function(x[0])) for x in self.cpopulation] # count the function values
        self.cpopulation.sort(key=lambda x:x[1])                                # sort via function value
        self.cpopulation = self.cpopulation[:self.param['mu']]                  # kill all not in range mu
        if self.cpopulation[0][1] < self.bsf[1]:                                # if the new organism is better than current best
            self.bsf = self.cpopulation[0]                                      # replace best with it
        return

    # cpopulation = [point]
    # point = (organism, function_value) = (([coeffs], used_sigma), function_value)
    def recombination(self):
        self.logs.append((self.centroid, self.sigma))
        self.generation += 1
        self.centroid = sum(point[0][0] for point in self.cpopulation) / self.param['mu']
        if self.param['self-adaptation']:
            self.sigma = sum(i[1] for i in self.cpopulation) / self.param['mu'] # count expected sigma
        else:
            self.sigma *= exp((1/sqrt(self.param['dimension'])) * np.random.normal(0, 1))
        if(self.generation > self.param['gamma']):
            self.shift = self.centroid - self.logs[-self.param['gamma']][0]
            self.shift = sqrt(sum(i**2 for i in self.shift))
        return



def INTERFACE():
    params = {
        'dimension' : 10,           # Fucntion dimensions
        'prec' : 0.000001,          # The minimum shift in gamma generations
        'mu': 10,                   # The survival population
        'lambda' : 20,              # The generated population
        'gamma' : 20,               # number of iterations to define stagnation
        'self-adaptation' : True,   # Algorithm flag
        'limits' : [-100, 100],     # Limits of parameters to choose from uniform distribution
        'debug' : False,            # Show debug info
        'max_iter' : 2000}          # Maximum iterations (generations)

    seeds = [1]# [-10, 1, 19, 98, 1002]

    population = [int(i) for i in input("Plese inticate populations to research:\n").split(" ")]
    sigma = [float(i) for i in input("Please indicate sigmas to research: \n").split(" ")]

    centroid = None # [-100] * params['dimension']

    for pop in population:
        params['lambda'] = pop
        for sig in sigma:
            for sd in seeds:
                np.random.seed(sd)
                params['self-adaptation'] = False
                LMR = EvolutionStrategy(spheric(params['dimension']), centroid, sig, params)
                LMR.exe()
                print(LMR.bsf)
                print(LMR.generation)

                params['self-adaptation'] = True
                SA = EvolutionStrategy(spheric(params['dimension']), centroid, sig, params)
                SA.exe()
                print(SA.bsf)
                print(SA.generation)


INTERFACE()