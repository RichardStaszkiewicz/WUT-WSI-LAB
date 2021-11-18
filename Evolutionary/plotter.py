from evolution import EvolutionStrategy, spheric, happy_cat
import numpy as np
import matplotlib.pyplot as plt
import os
import csv

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
    'debug' : False,             # Show debug info
    'max_iter' : 2000}          # Maximum iterations (generations)

seeds = [10, 1, 19, 98, 1002]

population = [20, 100]
sigma = [0.1, 1, 10]

centroid = [-100] * params['dimension']

functions = [spheric, happy_cat]

for func in functions:
    if not os.path.exists(f"{os.getcwd()}/Charts/{func.__name__}"):
        os.makedirs(f"{os.getcwd()}/Charts/{func.__name__}")

with open('Evolution_raport.csv', 'w+') as handle:
    fieldnames = ['Function', 'Sigma', 'Population', 'LMR Iter', 'LMR Prec', 'SA Iter', 'SA Prec']
    writer = csv.writer(handle, delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(fieldnames)
    for pop in population:
        print(pop)
        params['lambda'] = pop
        params['mu'] = pop//2
        for sig in sigma:
            print(f"\tsigma={sig}")
            for func in functions:
                print(f"\t\tfunction={func.__name__}")
                raport = [func.__name__, sig, pop]
                for method in (False, True):
                    center_it = 0
                    center_bsf = 0
                    center_final = 0
                    center_sigma = [0] * params['max_iter']
                    center_centroid = np.array([0] * params['max_iter'])
                    min_iter = params['max_iter']
                    for sd in seeds:
                        np.random.seed(sd)
                        params['self-adaptation'] = method
                        ES = EvolutionStrategy(func(params['dimension']), centroid, sig, params)
                        ES.exe()
                        center_bsf += ES.bsf[1] # bsf = (organism, function_value)
                        center_it += ES.generation
                        ES.logs = np.array(ES.logs, dtype=object)
                        ES.logs = ES.logs.T # logs = ([[organism1, fval1, sigma1], ...])
                        if len(ES.logs[0]) < min_iter:
                            min_iter = len(ES.logs[0])
                            center_centroid = ES.logs[1] + center_centroid[:min_iter]
                            center_sigma = ES.logs[2] + center_sigma[:min_iter]
                        else:
                            center_centroid = ES.logs[1][:min_iter] + center_centroid
                            center_sigma = ES.logs[2][:min_iter] + center_sigma
                    center_bsf /= len(seeds)
                    center_it /= len(seeds)
                    center_sigma = [i/len(seeds) for i in center_sigma]
                    center_centroid = [i/len(seeds) for i in center_centroid]
                    if method: info1, info2, col = "Self-Adaptation Value", "Self-Adaptation Sigma", '-g'
                    else: info1, info2, col = "Long-Normal Value", "Long-Normal Sigma", '-r'
                    plt.plot(range(len(center_centroid)), center_centroid, col, label=info1, markersize=3)
                    plt.plot(range(len(center_sigma)), center_sigma, f'-{col}', label=info2, markersize=3)
                    raport.append(round(center_it))
                    raport.append(center_bsf)
                plt.legend()
                plt.xlabel("Iterations")
                plt.ylabel("Function value")
                plt.yscale('log')
                plt.title(label=f"Population={pop}, Sigma={sig}, Function={func.__name__}", loc="center")
                plt.gcf().savefig(f"./Charts/{func.__name__}/Population={pop}-Sigma={sig}-Function={func.__name__}.png", format='png')
                plt.clf()
                writer.writerow(raport)


