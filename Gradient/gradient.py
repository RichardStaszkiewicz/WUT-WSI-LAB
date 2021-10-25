# Zaimplementuj algorytm gradientu prostego oraz algorytm Newtona. Algorytm Newtona powinien móc działać w dwóch trybach:
# ze stałym parametrem kroku
# z adaptacją parametru kroku przy użyciu metody z nawrotami.
# Następnie zbadaj zbieżność obu algorytmów, używając następującej funkcji:

# f(x) = sum a^(i-1)/(n-1)x_i^2

# Zbadaj wpływ wartości parametru kroku na zbieżność obu metod. W swoich badaniach rozważ następujące wartości parametru  oraz dwie wymiarowości . Ponadto porównaj czasy działania obu algorytmów.

# Pamiętaj, że przeprowadzone eksperymenty numeryczne powinny dać się odtworzyć.

# Znajdź maximum/minimum
# https://ewarchul.github.io/

# GIT_SSL_NO_VERIFY=true


# można zrobić obiekt z flagami - nie chcę rozwalać kodu...
# od 18.00 do 20.00 w pon w sali 26 parter


import numpy as np
import numdifftools as nd
from math import *
import csv


STEP_PAR=0.01
HREALITY=[]             # realities to be verified
ALPHA=[]                # alphas to be verified
MIMX=0                  # -1 -> find minimum & 1 -> find maximum
PRECISION=0.000001      # default precision
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
        return self.current_point

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
        return self.current_point

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
    print(f"Information: the default precision is set to {PRECISION}\nCOMPUTING...")
    for real in HREALITY:
        for alpha in ALPHA:
            print(f"\n\nCurrent parameters:\ndimensions={real}\nalpha={alpha}")
            if debug: print(f"Initiating Simple Gradient method...")
            point = np.array(real*[100])
            SG = SimpleGradient(real, dafaultFunction(alpha), PRECISION, 0.005, None, MIMX, debug)
            print(f"Coefficient found by Simple Gradient method:\n{list(SG.exe())}")
            # if debug: print(f'Initiating Newton Constant Step algorithm...')
            # NCS = NewtonAlgorithm(real, dafaultFunction(alpha), PRECISION, 1, None, False, MIMX, debug)
            # print(f"Coefficient found by Newton Constant algorithm:\n{list(NCS.exe())}")
            # if debug: print(f'Initiating Newton Backtracing algorithm...')
            # NCS = NewtonAlgorithm(real, dafaultFunction(alpha), PRECISION, 1, point, True, MIMX, debug)
            # print(f"Coefficient found by Newton Backtracking algorithm:\n{list(NCS.exe())}")


if __name__ == "__main__":
    INTERFACE()


# Działające parametry:
# a = 1, 10; n = 10, 20; precision = 0.000001
# SimpleGradient -> step = 0.01 (przy step = 1 wpada w oscylację)
# NewtonAlgorithm no backtracking -> step = 1
# NewtonAlgorithm wht backtracking -> step = 0.01
# a = 100; n = 10, 20; precision = 0.000001
# SimpleGradient -> step = 0.001 (ledwo - ponad 6k kroków)
# NewtonAlgorithm no backtracking -> step = 1
# NewtonAlgorithm with backtracking -> step = 0.01