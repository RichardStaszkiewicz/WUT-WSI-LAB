# Pytania:
# 1. Funkcja sferyczna i tasiemiec to funkcje kary, a nie te do optymalizacji?
# 2. Jeszcze raz - przesuwanie rozkładem po przestrzeni

from math import *

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

class EvolutionStrategy(object):

    def __init__(self, parameters) -> None:
        self.param = parameters
        self.cpopulation = self.new_population(parameters)
        self.generation = 0
        self.centroid = 1/self.param['mu'] * sum(self.cpopulation)
    pass


def INTERFACE():
    params = {
        'dimension' : 10,
        'func' : spheric_function(10),
        'prec' : 0.000001,
        'mu': 10,
        'lambda' : 200,
        'start_sigma' : 1,
        'self-adaptation' : True,
        'limits' : [-100, 100],
        'debug' : False,
        'max_iter' : 1000}

    population = [int(i) for i in input().split(" ")]
    sigma = [float(i) for i in input().split(" ")]

    for pop in population:
        for sig in sigma:
            params['lambda'] = pop
            params['start_sigma'] = sig
