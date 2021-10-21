# Zaimplementuj algorytm gradientu prostego oraz algorytm Newtona. Algorytm Newtona powinien móc działać w dwóch trybach:
# ze stałym parametrem kroku
# z adaptacją parametru kroku przy użyciu metody z nawrotami.
# Następnie zbadaj zbieżność obu algorytmów, używając następującej funkcji:

# f(x) = sum a^(i-1)/(n-1)x_i^2

# Zbadaj wpływ wartości parametru kroku na zbieżność obu metod. W swoich badaniach rozważ następujące wartości parametru  oraz dwie wymiarowości . Ponadto porównaj czasy działania obu algorytmów.

# Pamiętaj, że przeprowadzone eksperymenty numeryczne powinny dać się odtworzyć.

# Znajdź maximum/minimum
# https://ewarchul.github.io/


import numpy as np
from math import *

STEP_PAR=0
HREALITY=[]             # realities to be verified
ALPHA=[]                # alphas to be verified
MIMX=0                  # -1 -> find minimum & 1 -> find maximum
PRECISION=0.0001        # default precision

def dafaultFunction(alpha, X):
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


class SimpleGradient(object):
    def __init__(self, rreal, aalpha, ffunc, pprec) -> None:
        real = rreal
        alpha = aalpha
        func = ffunc
        prec = pprec


def INTERFACE():
    print("Plese indicate the space seperated realities you will be researching:")
    HREALITY = input().split(" ")
    HREALITY = [int(i) for i in HREALITY]
    print("Please indicate the space seperated alphas you will be researching:")
    ALPHA = input().split(" ")
    ALPHA = [float(i) for i in ALPHA]
    print("Please indicate, if you want to find minimum (-1) or maximum (1):")
    MIMX = int(input())
    print(f"Information: te default precision is set to {PRECISION}\nCOMPUTING")




INTERFACE()
