import numpy as np
import numdifftools as nd
import matplotlib.pyplot as plt
from math import *
import csv
import time

from numpy.core.numeric import Inf

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
        logs = []                   # logs of function values in steps
        while mgnt >= self.prec and steps < self.maxiter:    # War1-> przesunięcie względne; War2 -> liczba iteracji
            logs.append(self.func(self.current_point))
            new_pos = self.make_step()
            mgnt = magnitude(new_pos - self.current_point)
            self.current_point = new_pos
            steps += 1
            if steps % 1000 == 0 and self.debug:
                print(f"Coordinates after {steps} steps:\n{list(self.current_point)}")
        logs.append(self.func(self.current_point))
        return (self.current_point, steps, logs)

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
        logs = []
        while mgnt >= self.prec and steps < self.maxiter:    # while there is a shift greater than precision
            logs.append(self.func(self.current_point))
            if self.backtracking: new_pos = self.make_stepBC()
            else: new_pos = self.make_stepNBC()
            mgnt = magnitude(new_pos - self.current_point)
            self.current_point = new_pos
            steps += 1
            if steps % 1000 == 0 and self.debug:
                print(f"Coordinates after {steps} steps:\n{list(self.current_point)}")
        logs.append(self.func(self.current_point))
        return (self.current_point, steps, logs)

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

                    SG = SimpleGradient(real, dafaultFunction(alpha), 0.000001, 0.005, point, MIMX, debug, max_iter=2000)
                    start = time.time()
                    ans = SG.exe()
                    log.append(ans[1])
                    log.append(time.time() - start)
                    log.append(magnitude(ans[0]))
                    if point is None: (stl, inf) = ('-r', 'Simple Gradient random SP')
                    else: (stl, inf) = ('--r', 'Simple Gradient corner SP')
                    plt.plot(range(ans[1] + 1), ans[2], stl, label=inf, markersize=3)

                    NCS = NewtonAlgorithm(real, dafaultFunction(alpha), 0.000001, 1, point, False, MIMX, debug, max_iter=2000)
                    start = time.time()
                    ans = NCS.exe()
                    log.append(ans[1])
                    log.append(time.time() - start)
                    log.append(magnitude(ans[0]))
                    if point is None: (stl, inf) = ('-b', 'Newton Constant Step random SP')
                    else: (stl, inf) = ('--b', 'Newton Constant Step corner SP')
                    plt.plot(range(ans[1] + 1), ans[2], stl, label=inf, markersize=3)

                    NCS = NewtonAlgorithm(real, dafaultFunction(alpha), 0.000001, 1, point, True, MIMX, debug, max_iter=2000)
                    start = time.time()
                    ans = NCS.exe()
                    log.append(ans[1])
                    log.append(time.time() - start)
                    log.append(magnitude(ans[0]))
                    if point is None: (stl, inf) = ('-g', 'Newton with Backtracking random SP')
                    else: (stl, inf) = ('--g', 'Newton with Backtracking corner SP')
                    plt.plot(range(ans[1] + 1), ans[2], stl, label=inf, markersize=3)

                    writer.writerow(log)

                plt.legend()
                plt.xlabel("Iterations")
                plt.ylabel("Function value")
                plt.yscale('log')
                plt.xscale('log')
                plt.title(label=f"Dimensions={real}, Alpha={alpha}", loc="center")
                plt.gcf().savefig(f'Dimensions={real}_Alpha={alpha}.png', format='png')
                plt.clf()

        for step, col in zip([1, 0.01, 0.005, 0.001], ['r', 'y', 'g', 'b']):
            for point in (np.array(real*[100]), None):
                SG = SG = SimpleGradient(10, dafaultFunction(100), 0.000001, step, point, -1, max_iter=2000)
                ans = SG.exe()
                if point is None: (stl, inf) = (f'-{col}', f'Random SP. Step={step}')
                else: (stl, inf) = (f'--{col}', f'Corner SP. Step={step}')
                plt.plot(range(ans[1] + 1), ans[2], stl, label=inf, markersize=3)
        plt.legend()
        plt.title(label=f"Steps vs Iterations in Simple Gradient", loc="center")
        plt.xlabel("Iterations")
        plt.ylabel("Function value")
        plt.yscale('log')
        plt.xscale('log')
        plt.gcf().savefig(f"Iter-vs-Step.png", format='png')
        plt.clf()




if __name__ == "__main__":
    INTERFACE()


