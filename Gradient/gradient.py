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


# Pytania:
# -> W jaki sposób zastosować PREC jeżeli nie znamy optimum globalnego?


import numpy as np
import numdifftools as nd
from math import *

STEP_PAR=0
HREALITY=[]             # realities to be verified
ALPHA=[]                # alphas to be verified
MIMX=0                  # -1 -> find minimum & 1 -> find maximum
PRECISION=0.0001        # default precision


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
    def __init__(self, rreal, ffunc, pprec, sstep, min_max=-1, debugMode=False, rg=[-100, 100]) -> None:
        self.real = rreal                                                        # no. of dimensions
        self.func = ffunc                                                        # the function to minimize
        self.prec = pprec                                                        # precision
        self.current_point=np.random.uniform(low=rg[0], high=rg[1], size=rreal)     # Starting position X
        self.step  = sstep                                                       # Constant step of gradient
        self.debug = debugMode
        self.min_max = min_max
        self.range = rg

    def exe(self):
        """
        Parameters:
            None
        Result:
            Point - vector of coefficiants found
        """
        mgnt = 5                    # the shift
        steps = 0
        while mgnt >= self.prec:    # while there is a shift greater than precision
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
        while True:
            nn = self.current_point + self.min_max*self.step*grad
            if nn.max() > self.range[1] or nn.min() < self.range[0]:
                grad = grad/2
            else:
                break
        return nn


def INTERFACE():
    print("Plese indicate the space seperated realities you will be researching:")
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
            print(f"\n\nCurrent parameters:\nrealities={real}\nalpha={alpha}")
            if debug: print(f"Initiating Simple Gradient method...")
            SG = SimpleGradient(real, dafaultFunction(alpha), PRECISION, 0.0001, MIMX, debug)
            print(f"Coefficient found by Simple Gradient method:\n{list(SG.exe())}")




INTERFACE()
