# import matplotlib.pyplot as plt
# import numpy as np
# import numdifftools as nd
# import csv
# from math import *
# import time

# def exe_time(function):
#     def wrapper(*args, **kwargs):
#         # summ = 0
#         # for _ in range(3):
#         #     start = time.time()
#         #     function(*args, **kwargs)
#         #     summ += time.time() - start
#         # return summ / 3
#         start = time.time()
#         function(*args, **kwargs)
#         return time.time() - start
#     return wrapper

# def plotTest(testing_object):
#     x = exe_time(testing_object.exe)()
#     return x

# def magnitude(X):
#     """
#     Counts the magnitude of given vector
#     Arguments:
#         X - vector as array
#     Result
#         mag - magnitude of X
#     """
#     return sqrt(sum(i**2 for i in X))

# def dafaultFunction(alpha):
#     """
#     Return function with alpha coefficiant
#     """
#     def fun(X):
#         """
#         Parameters:
#         alpha - coefficiant
#         X - list of n values to count function from

#         Result:
#         sum from i=1 to i=n of alpha^((i-1)/(n-1))*x_i^2
#         """
#         sum = 0
#         n = len(X)
#         for i in range(1, n+1):
#             sum += alpha**((i-1)/(n-1))*X[i-1]**2
#         return sum
#     return fun


# class SimpleGradient(object):
#     def __init__(self, rreal, ffunc, pprec, sstep, spoint = None, min_max=-1, debugMode=False, rg=[-100, 100], max_iter=None) -> None:
#         self.real = rreal                                                        # no. of dimensions
#         self.func = ffunc                                                        # the function to minimize
#         self.prec = pprec                                                        # precision
#         if spoint is None: self.current_point=np.random.uniform(low=rg[0], high=rg[1], size=rreal)     # Starting position X
#         elif spoint.all(): self.current_point=spoint
#         else: self.current_point=spoint
#         self.step  = sstep                                                       # Constant step of gradient
#         self.debug = debugMode
#         self.min_max = min_max
#         self.range = rg
#         if max_iter: self.maxiter = max_iter
#         else: self.maxiter = 10000

#     def exe(self):
#         """
#         Parameters:
#             None
#         Result:
#             Point - vector of coefficiants found
#         """
#         mgnt = 5                    # the shift
#         steps = 0
#         while mgnt >= self.prec and steps < self.maxiter:    # War1-> przesunięcie względne; War2 -> liczba iteracji
#             new_pos = self.make_step()
#             mgnt = magnitude(new_pos - self.current_point)
#             self.current_point = new_pos
#             steps += 1
#             if steps % 1000 == 0 and self.debug:
#                 print(f"Coordinates after {steps} steps:\n{list(self.current_point)}")
#         return self.current_point

#     def make_step(self):
#         """
#         Steps taken as a function: x_i_new = x_i_old - grad(X)[i]*step (for finding minimum)
#         Parameters:
#             None
#         Result:
#             Point - vector of coefficients found
#         """
#         grad = nd.Gradient(self.func)(self.current_point)
#         nn = self.current_point + self.min_max*self.step*grad
#         # ss = self.step
#         # while True:
#         #     nn = self.current_point + self.min_max*ss*grad
#         #     if nn.max() > self.range[1] or nn.min() < self.range[0]:
#         #         ss /= 2
#         #     else:
#         #         break
#         return nn


# class NewtonAlgorithm(object):
#     def __init__(self, rreal, ffunc, pprec, sstep, spoint = None, bbacktracking=False, min_max=-1, debugMode=False, rg=[-100, 100], max_iter=None) -> None:
#         self.real = rreal                                                        # no. of dimensions
#         self.func = ffunc                                                        # the function to minimize
#         self.prec = pprec                                                        # precision
#         if spoint is None: self.current_point=np.random.uniform(low=rg[0], high=rg[1], size=rreal)     # Starting position X
#         else: self.current_point = spoint
#         self.step  = sstep                                                       # Constant step of gradient
#         self.debug = debugMode
#         self.min_max = min_max
#         self.range = rg
#         self.backtracking = bbacktracking
#         if max_iter: self.maxiter = max_iter
#         else: self.maxiter = 10000

#     def exe(self):
#         """
#         Parameters:
#             None
#         Result:
#             Point - vector of coefficiants found
#         """
#         mgnt = 5                    # the shift
#         steps = 0
#         while mgnt >= self.prec and steps < self.maxiter:    # while there is a shift greater than precision
#             if self.backtracking: new_pos = self.make_stepBC()
#             else: new_pos = self.make_stepNBC()
#             mgnt = magnitude(new_pos - self.current_point)
#             self.current_point = new_pos
#             steps += 1
#             if steps % 1000 == 0 and self.debug:
#                 print(f"Coordinates after {steps} steps:\n{list(self.current_point)}")
#         return self.current_point

#     def make_stepNBC(self):
#         """
#         Steps taken as a function: x_i_new = x_i_old - grad(X)[i]*step (for finding minimum)
#         Parameters:
#             None
#         Result:
#             Point - vector of coefficients found
#         """
#         Hessi = np.array(nd.Hessian(self.func)(self.current_point))
#         Hessi = np.linalg.inv(Hessi)
#         grad = nd.Gradient(self.func)(self.current_point)
#         ss = self.step
#         while True:
#             new_point = self.current_point + self.min_max*ss*np.dot(Hessi,grad)
#             if new_point.max() > self.range[1] or new_point.min() < self.range[0]:
#                 ss /= 2
#             else:
#                 break
#         return new_point

#     def make_stepBC(self):
#         """
#         Steps taken as a function: x_i_new = x_i_old - grad(X)[i]*step (for finding minimum)
#         Parameters:
#             None
#         Result:
#             Point - vector of coefficients found
#         """
#         Hessi = np.array(nd.Hessian(self.func)(self.current_point))
#         Hessi = np.linalg.inv(Hessi)
#         grad = nd.Gradient(self.func)(self.current_point)
#         penalty_coeff = 1
#         ss = self.step
#         while True:
#             new_point = self.current_point + self.min_max*ss*np.dot(Hessi,grad)
#             if new_point.max() > self.range[1] or new_point.min() < self.range[0]:
#                 ss /= 2
#             elif self.func(self.current_point) < (1 - 0.9*penalty_coeff)*self.func(new_point) and self.min_max == -1:
#                 print(f'penalty {penalty_coeff}')
#                 penalty_coeff /=2
#             elif self.func(self.current_point) > (1 - 0.9*penalty_coeff)*self.func(new_point) and self.min_max == 1:
#                 print(f'penalty {penalty_coeff}')
#                 penalty_coeff /=2
#             else:
#                 break
#         return new_point

# # steps = [0.001, 0.01, 0.1, 1, 10, 100]

# times10 = [[], [], []]        # [SimpleGrad, NetonNB, NewtonB]
# times20 = [[], [], []]
# precision = 0.000001
# # times10[0] = [exe_time(SimpleGradient(10, dafaultFunction(alpha), precision, 0.01, None, -1).exe)() for alpha in (1, 10, 100)]
# # times20[0] = [exe_time(SimpleGradient(20, dafaultFunction(alpha), precision, 0.01, None, -1).exe)() for alpha in (1, 10, 100)]
# print('doneSG')
# times10[1] = [exe_time(NewtonAlgorithm(10, dafaultFunction(alpha), precision, 1, None, False, -1).exe)() for alpha in (1, 10, 100)]
# times20[1] = [exe_time(NewtonAlgorithm(20, dafaultFunction(alpha), precision, 1, None, False, -1).exe)() for alpha in (1, 10, 100)]
# print('doneNNB')
# times10[2] = [exe_time(NewtonAlgorithm(10, dafaultFunction(alpha), precision, 1, None, True, -1).exe)() for alpha in (1, 10, 100)]
# times20[2] = [exe_time(NewtonAlgorithm(20, dafaultFunction(alpha), precision, 1, None, True, -1).exe)() for alpha in (1, 10, 100)]
# print('doneNBC')
# #plt.plot([1, 10, 100], times10[0], '-r', label="Simple Gradient. Step = 0.01/0.001", markersize=3)

# with open('Random_starting_point.csv', 'w') as handle:
#     fieldnames = ['Simple Gradient', 'Newton no Backtracking', 'Newton with Backtracking']

# plt.plot([1, 10, 100], times10[1], '-b', label="Newton no Backtracking. Step = 1", markersize=3)
# plt.plot([1, 10, 100], times10[2], '-g', label="Newton with Backtracking. Step = 0.01", markersize=3)
# plt.legend()
# plt.title(label = "10 Dimensions. Random starting point", loc = 'center')
# plt.gcf().savefig('Steps.png', format='png')
# plt.clf()
# plt.plot([1, 10, 100], times20[0], '-r', label="Simple Gradient. Step = 0.01/0.001", markersize=3)
# plt.plot([1, 10, 100], times20[1], '-b', label="Newton no Backtracking. Step = 1", markersize=3)
# plt.plot([1, 10, 100], times20[2], '-g', label="Newton with Backtracking. Step = 0.01", markersize=3)
# plt.legend()
# plt.title(label = "20 Dimensions. Random starting point", loc = 'center')
# plt.gcf().savefig('Steps.png', format='png')
# plt.clf()

# # Działające parametry:
# # a = 1, 10; n = 10, 20; precision = 0.000001
# # SimpleGradient -> step = 0.005
# # NewtonAlgorithm no backtracking -> step = 1
# # NewtonAlgorithm wht backtracking -> step = 1
# # a = 100; n = 10, 20; precision = 0.000001
# # SimpleGradient -> step = 0.001 (ledwo - ponad 6k kroków)
# # NewtonAlgorithm no backtracking -> step = 1
# # NewtonAlgorithm with backtracking -> step = 1

import numpy as np
import numdifftools as nd
from math import *
import csv
import time

HREALITY=[]             # realities to be verified
ALPHA=[]                # alphas to be verified
MIMX=0                  # -1 -> find minimum & 1 -> find maximum
MAX_ITER=10000          # maximal iteration


def magnitude(X):
    """
    Counts the magnitude of given vector
    Arguments:
        X - vector as array
    Result
        mag - magnitude of X
    """
    return sqrt(sum(i**2 for i in X))


def dafaultFunction(alpha):
    """
    Return function with alpha coefficiant
    """
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
            sum += alpha**((i-1)/(n-1))*X[i-1]**2
        return sum
    return fun


class SimpleGradient(object):
    def __init__(self, rreal, ffunc, pprec, sstep, spoint = None, min_max=-1, debugMode=False, rg=[-100, 100], max_iter=None) -> None:
        self.real = rreal                                                        # no. of dimensions
        self.func = ffunc                                                        # the function to minimize
        self.prec = pprec                                                        # precision
        if spoint is None: self.current_point=np.random.uniform(low=rg[0], high=rg[1], size=rreal)     # Starting position X
        elif spoint.all(): self.current_point=spoint
        else: self.current_point=spoint
        self.step  = sstep                                                       # Constant step of gradient
        self.debug = debugMode
        self.min_max = min_max
        self.range = rg
        if max_iter: self.maxiter = max_iter
        else: self.maxiter = 10000

    def exe(self):
        """
        Parameters:
            None
        Result:
            Point - vector of coefficiants found
        """
        mgnt = 5                    # the shift
        steps = 0
        while mgnt >= self.prec and steps < self.maxiter:    # War1-> przesunięcie względne; War2 -> liczba iteracji
            new_pos = self.make_step()
            mgnt = magnitude(new_pos - self.current_point)
            self.current_point = new_pos
            steps += 1
            if steps % 1000 == 0 and self.debug:
                print(f"Coordinates after {steps} steps:\n{list(self.current_point)}")
        return (self.current_point, steps)

    def make_step(self):
        """
        Steps taken as a function: x_i_new = x_i_old - grad(X)[i]*step (for finding minimum)
        Parameters:
            None
        Result:
            Point - vector of coefficients found
        """
        grad = nd.Gradient(self.func)(self.current_point)
        # print(grad)
        # print(self.current_point)
        nn = self.current_point + self.min_max*self.step*grad
        # ss = self.step
        # while True:
        #     nn = self.current_point + self.min_max*ss*grad
        #     if nn.max() > self.range[1] or nn.min() < self.range[0]:
        #         ss /= 2
        #     else:
        #         break
        return nn


class NewtonAlgorithm(object):
    def __init__(self, rreal, ffunc, pprec, sstep, spoint = None, bbacktracking=False, min_max=-1, debugMode=False, rg=[-100, 100], max_iter=None) -> None:
        self.real = rreal                                                        # no. of dimensions
        self.func = ffunc                                                        # the function to minimize
        self.prec = pprec                                                        # precision
        if spoint is None: self.current_point=np.random.uniform(low=rg[0], high=rg[1], size=rreal)     # Starting position X
        else: self.current_point = spoint
        self.step  = sstep                                                       # Constant step of gradient
        self.debug = debugMode
        self.min_max = min_max
        self.range = rg
        self.backtracking = bbacktracking
        if max_iter: self.maxiter = max_iter
        else: self.maxiter = 10000

    def exe(self):
        """
        Parameters:
            None
        Result:
            Point - vector of coefficiants found
        """
        mgnt = 5                    # the shift
        steps = 0
        while mgnt >= self.prec and steps < self.maxiter:    # while there is a shift greater than precision
            if self.backtracking: new_pos = self.make_stepBC()
            else: new_pos = self.make_stepNBC()
            mgnt = magnitude(new_pos - self.current_point)
            self.current_point = new_pos
            steps += 1
            if steps % 1000 == 0 and self.debug:
                print(f"Coordinates after {steps} steps:\n{list(self.current_point)}")
        return (self.current_point, steps)

    def make_stepNBC(self):
        """
        Steps taken as a function: x_i_new = x_i_old - grad(X)[i]*step (for finding minimum)
        Parameters:
            None
        Result:
            Point - vector of coefficients found
        """
        Hessi = np.array(nd.Hessian(self.func)(self.current_point))
        Hessi = np.linalg.inv(Hessi)
        grad = nd.Gradient(self.func)(self.current_point)
        ss = self.step
        while True:
            new_point = self.current_point + self.min_max*ss*np.dot(Hessi,grad)
            if new_point.max() > self.range[1] or new_point.min() < self.range[0]:
                ss /= 2
            else:
                break
        return new_point

    def make_stepBC(self):
        """
        Steps taken as a function: x_i_new = x_i_old - grad(X)[i]*step (for finding minimum)
        Parameters:
            None
        Result:
            Point - vector of coefficients found
        """
        Hessi = np.array(nd.Hessian(self.func)(self.current_point))
        Hessi = np.linalg.inv(Hessi)
        grad = nd.Gradient(self.func)(self.current_point)
        penalty_coeff = 1
        ss = self.step
        while True:
            new_point = self.current_point + self.min_max*ss*np.dot(Hessi,grad)
            if new_point.max() > self.range[1] or new_point.min() < self.range[0]:
                ss /= 2
            elif self.func(self.current_point) < (1 - 0.9*penalty_coeff)*self.func(new_point) and self.min_max == -1:
                print(f'penalty {penalty_coeff}')
                penalty_coeff /=2
            elif self.func(self.current_point) > (1 - 0.9*penalty_coeff)*self.func(new_point) and self.min_max == 1:
                print(f'penalty {penalty_coeff}')
                penalty_coeff /=2
            else:
                break
        return new_point


def INTERFACE():
    print("Plese indicate the space seperated dimensions you will be researching:")
    HREALITY = input().split(" ")
    HREALITY = [int(i) for i in HREALITY]
    print("Please indicate the space seperated alphas you will be researching:")
    ALPHA = input().split(" ")
    ALPHA = [float(i) for i in ALPHA]
    print("Please indicate, if you want to find minimum (-1) or maximum (1):")
    MIMX = int(input())
    print("Would you like to recieve messages of processing? (y/n):")
    debug = 1 if str(input()) == "y" else 0
    print(f"Information: the default precision is set to 0.000001\nCOMPUTING...")

    # logger:
    # Dimension; alpha; Starting point random; SimpleGradientIT; SimpleGradient Time; NewtonConstIT; NewtonConst Time; NewtonBacktrIT; NewtonBacktr Time
    with open('logger.csv', 'w') as lg:
        fieldnames = ['Dimensions', 'alpha', 'SP-random', 'SG-It', 'SG-T', 'SG-R2', 'NC-It', 'NC-T', 'NC-R2', 'NB-It', 'NB-T', 'NB-R2']
        writer = csv.writer(lg, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(fieldnames)
        for real in HREALITY:
            for alpha in ALPHA:
                for point in (np.array(real*[100]), None):
                    log = [real, alpha]
                    if point is None: log.append(True)
                    else: log.append(False)

                    SG = SimpleGradient(real, dafaultFunction(alpha), 0.000001, 0.005, point, MIMX, debug)
                    start = time.time()
                    ans = SG.exe()
                    log.append(ans[1])
                    log.append(time.time() - start)
                    log.append(magnitude(ans[0]))

                    NCS = NewtonAlgorithm(real, dafaultFunction(alpha), 0.000001, 1, point, False, MIMX, debug)
                    start = time.time()
                    ans = NCS.exe()
                    log.append(ans[1])
                    log.append(time.time() - start)
                    log.append(magnitude(ans[0]))

                    NCS = NewtonAlgorithm(real, dafaultFunction(alpha), 0.000001, 1, point, True, MIMX, debug)
                    start = time.time()
                    ans = NCS.exe()
                    log.append(ans[1])
                    log.append(time.time() - start)
                    log.append(magnitude(ans[0]))

                    writer.writerow(log)



if __name__ == "__main__":
    INTERFACE()

