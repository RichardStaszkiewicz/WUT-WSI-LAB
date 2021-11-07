# Pytania:
# 1. Czy w wyrażeniu exp{T*N(0, 1)} nie dałoby się wciągnąć by było exp{N(0, T)}?

# matplotlib contour

# Wstęp
# Opis eksperymentów
# Wyniki i dyskusja
# Wnioski

# zależności od:
# 1. sigmy
# 2. lambdy
# 3. metody

from math import *
import numpy as np

def magnitude(X):
    """
    Counts the magnitude of given vector
    Arguments:
        X - vector as array
    Result
        mag - magnitude of X
    """
    return sqrt(sum(i**2 for i in X))


def spheric_function(Dimensions):
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
            sum += Dimensions**((i-1)/(n-1))*X[i-1]**2
        return sum
    return fun


def second_function(Dimensions):
    def func(X):
        """
        Parameters:
        Dimensions - number of dimensions
        X - values of each dimension
        """
        m = magnitude(X)
        sum = ((m**2 - Dimensions)**2)**(1/8)
        sum += (1/Dimensions)*(0.5*m**2 + sum(X)) + 0.5
        return sum
    return func


# Parameters dict:
# {
#     'dimension'
#     'func'
#     'prec'
#     'mu'
#     'lambda'
#     'start_sigma'
#     'self-adaptation'
#     'limits'
#     'debug'
#     'max_iter'
# }

# Samoadaptacja -> dla każdego organizmu osobna sigma
# Log-Gauss -> jedna wspólna sigma dla każdego organizmu

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


    def exe(self):
        while(self.generation < self.param['max_iter'] and self.shift > self.param['prec']):
            self.new_population()           # na podstawie centroidu generowane lambda punktów
            self.selection()                # posortuj po funkcji i ubij połowę najgorszych
            self.recombination()            # nowy centroid, nowa sigma, aktualizacja shiftu


    def new_population(self):
        """
            Each point consists of a tuple(). In case of LMR
            the tuple contains only coefficiants. In case of EA,
            on the second place is the used sigma.
        """
        if self.param['self-adaptation']:   # generate using self-adaptation
            self.cpopulation = [None] * self.param['lambda']
            for i in range(self.param['lambda']):
                altered_sigma = self.sigma * exp((1/sqrt(self.param['dimension'])) * np.random.normal(0, 1))
                self.cpopulation[i] = (np.random.normal(self.centroid, altered_sigma), altered_sigma)
        else:                               # generate using gauss
            self.cpopulation = [(np.random.normal(self.centroid, self.sigma)) for _ in range(self.param['lambda'])]


    def selection(self):
        self.cpopulation = [(x, self.function(x[0])) for x in self.cpopulation] # count the function values
        self.cpopulation = self.cpopulation.sort(key=lambda x:x[1])             # sort via function value
        self.cpopulation = self.cpopulation[:self.param['mu']]                  # kill all not in range mu


    def recombination(self):
        self.logs.append((self.centroid, self.sigma))
        self.generation += 1
        self.centroid = sum(self.cpopulation[0][0]) / self.param['mu']
        if self.param['self-adaptation']:
            self.sigma = sum(self.cpopulation[0][1]) / self.param['mu'] # count expecte sigma
        else:
            self.sigma *= exp((1/sqrt(self.param['dimension'])) * np.random.normal(0, 1))
        if(self.generation > self.param['gamma']):
            self.shift = self.centroid - self.logs[-(self.param['gamma'] + 1)][0]



def INTERFACE():
    params = {
        'dimension' : 10,
        'prec' : 0.000001,          # The minimum shift in gamma generations
        'mu': 10,                   # The survival population
        'lambda' : 20,              # The generated population
        'start_sigma' : 1,          # Starting sigma
        'gamma' : 20,               # number of iterations to sefine stagnation
        'self-adaptation' : True,   # Algorithm flag
        'limits' : [-100, 100],     # Limits of parameters
        'debug' : False,            # Show debug info
        'max_iter' : 2000}          # Maximum iterations (generations)

    population = [int(i) for i in input().split(" ")]
    sigma = [float(i) for i in input().split(" ")]

    for pop in population:
        for sig in sigma:
            params['lambda'] = pop
            params['start_sigma'] = sig
