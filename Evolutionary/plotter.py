from evolution import EvolutionStrategy, spheric
import numpy as np
import matplotlib.pyplot as plt
import os

# Logs of ES return:
# --> Best fitting current
# --> Logs of changes in best fitting and sigma


params = {
    'dimension' : 9,           # Fucntion dimensions
    'prec' : 0.000001,          # The minimum shift in gamma generations
    'mu': 10,                   # The survival population
    'lambda' : 20,              # The generated population
    'gamma' : 20,               # number of iterations to define stagnation
    'self-adaptation' : True,   # Algorithm flag
    'limits' : [-100, 100],     # Limits of parameters to choose from uniform distribution
    'debug' : True,             # Show debug info
    'max_iter' : 2000}          # Maximum iterations (generations)

seeds = [1]# [-10, 1, 19, 98, 1002]

population = [20]
sigma = [1]

centroid = [-100] * params['dimension']

functions = [spheric]

for func in functions:
    if not os.path.exists(f"{os.getcwd()}/Charts/{func.__name__}"):
        os.makedirs(f"{os.getcwd()}/Charts/{func.__name__}")


for pop in population:
    params['lambda'] = pop
    params['mu'] = pop//2
    for sig in sigma:
        for func in functions:
            # center_chart = np.array([])
            # sigma_chart = np.array([])
            for method in (True, False):
                center_bsf = [0] * params['dimension']
                center_sigma = [0] * params['max_iter']
                center_centroid = np.array([0] * params['max_iter'])
                min_iter = params['max_iter']
                for sd in seeds:
                    np.random.seed(sd)
                    params['self-adaptation'] = method
                    ES = EvolutionStrategy(func(params['dimension']), centroid, sig, params)
                    ES.exe()
                    center_bsf += ES.bsf[1] # bsf = (organism, function_value)
                    ES.logs = np.array(ES.logs, dtype=object)
                    ES.logs = ES.logs.T # logs = ([[organism1, fval1, sigma1], ...])
                    if len(ES.logs[0]) < min_iter:
                        min_iter = len(ES.logs[0])
                        center_centroid = ES.logs[1] + center_centroid[:min_iter]
                        center_sigma = ES.logs[2] + center_sigma[:min_iter]
                    else:
                        center_centroid = ES.logs[1][:min_iter] + center_centroid
                        center_sigma = ES.logs[2][:min_iter] + center_sigma
                if method: info, col = "Self-Adaptation", '--g'
                else: info, col = "Long-Normal", '--r'
                plt.plot(range(len(center_sigma)), center_sigma, col, label=info, markersize=3)
            plt.legend()
            plt.xlabel("Iterations")
            plt.ylabel("Function value")
            plt.yscale('log')
            plt.xscale('log')
            plt.title(label=f"Population={pop}, Sigma={sig}, Function={func.__name__}", loc="center")
            plt.gcf().savefig(f"./Charts/{func.__name__}/Population={pop}-Sigma={sig}-Function={func.__name__}.png", format='png')
            plt.clf()


