from math import sin, cos
from matplotlib import pyplot as plt
import random
import numpy as np
from sklearn.model_selection import train_test_split
from NNetwork import NNetwork
from layers import Layer

MAX_X = 40
MIN_X = -40


def f(x):
    return x**2
    return x**2*sin(x)+100*sin(x)*cos(x)


def generate_samples(n, seed=1, function=f, min=MIN_X, max=MAX_X):
    """Generates n random floating point numbers from range [MIN_X, MAX_X]
       returns list X with random numbers and y with corresponding function values"""
    random.seed(seed)
    X = []
    y = []
    for i in range(n):
        x = random.uniform(min, max)
        X.append([x])
        y.append([function(x)])
    return X, y


def draw_function_graph(X_act, y_act, X_est, y_est):
    """Draws a graph with function values:
        first plot: X_act, y_act - actual values, not estimation
        second plot: X_est, y_est - values estimated using neural nets"""
    X_act, y_act = zip(*sorted(zip(X_act, y_act)))
    X_est, y_est = zip(*sorted(zip(X_est, y_est)))
    plt.plot(X_act, y_act, label="Actual")
    plt.plot(X_est, y_est, label="Estimated")
    plt.legend()
    plt.show()

def test_1():
    n_network = NNetwork()

    n_network.add_layer(Layer(1, 1, 'tanh', constant=np.array([1], np.float64)))
    n_network.add_layer(Layer(20, 1, 'tanh', constant=np.array([1 for _ in range(20)], np.float64)))
    #n_network.add_layer(Layer(20, 20, 'tanh', constant=np.array([1 for _ in range(20)], np.float64)))
    n_network.add_layer(Layer(1, 20, weights=np.zeros((20, 1)), constant=np.array([1], np.float64)))

    n = 100
    seed = 100
    test_part = 0.2
    X, y = generate_samples(n, seed, f, -4, 4)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_part)

    n_network.train_batch(X_train, y_train, 0.001, 10000)
    predictions = []
    for x in X_train:
        predictions.append(n_network.predict(x))
    draw_function_graph(X_train, y_train, X_train, predictions)


def main():
    n = 1000
    test_part = 0.1
    X, y = generate_samples(n, f, -1, 1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_part)
    draw_function_graph(X_train, y_train, X_test, y_test)


if __name__ == "__main__":
    test_1()