# Pytania:
# 1. Funkcja sferyczna i tasiemiec to funkcje kary, a nie te do optymalizacji?
# 2. Jeszcze raz - przesuwanie rozkładem po przestrzeni

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
        self.generation = 0
        if center:
            self.centroid = center
        else:
            self.centroid = np.random.uniform(low=parameters['limits'][0], high=parameters['limits'][1], size=parameters['dimension'])
        self.cpopulation = []


    def exe(self):
        while(self.generation < self.param['max_iter'] and self.shift > self.param['precision']):
            self.new_population()           # na podstawie centroidu generowane lambda punktów
            self.selection()                # posortuj po funkcji i ubij połowę najgorszych
            self.recombination()            # nowy centroid, nowa sigma




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
