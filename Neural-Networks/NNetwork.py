"""!
@file NNetwork.py
@brief Implements only one class representing Neural Network.
@section requirements
Numpy version >= 1.19.4\n
Layers version >= 0.1
"""

from layers import Layer
import numpy as np


class NNetwork(object):
    """!
    @brief
    Represents a neural network.
    """

    def __init__(self, global_seed=None) -> None:
        """!
        @brief
        Constructs a Neural network with no layers.
        @return NNetwork object
        """
        self.layers = []
        self.global_seed = global_seed if global_seed is not None else 1

    @property
    def layers(self):
        return self.__layers

    @layers.setter
    def layers(self, lay):

        for l in lay:
            if not isinstance(l, Layer): raise NameError

        self.__layers = lay

    def add_layer(self, layer: Layer):
        """!
        @brief Adds given layer to current Neural Network.
        @param layer [Layer] Layer to add to Neural Network.
        @return None
        """
        if not isinstance(layer, Layer): raise NameError

        self.layers.append(layer)

    def predict(self, X):
        """!
        @brief Feeds an Neural Network with input values.
        @param X [array-like object] Input values.
        @return [array-like object] Output values.
        """
        for layer in self.layers:
            X = layer.activate(X)
        return X

    def backpropagation(self, In, Target, step):
        """!
        @brief Performs the backpropagation on every each layer. Updates the weights.
        @param In [array-like object] Vector of inputs.
        @param Target [array-like object] Vector of targeted outputs
        @param step [float] The learning rate of the algorithm. The higher, the more impact (up to 1)
        @return None
        """
        predicted = self.predict(In)

        for n_lay in reversed(range(len(self.layers))):
            current_layer = self.layers[n_lay]

            if n_lay == len(self.layers) - 1:       #the output layer
                current_layer.error = Target - predicted#0.5 * np.mean(np.square(Target-predicted)) #Target - predicted
                current_layer.sigma = current_layer.error * current_layer.activation_derivative(predicted)
            else:
                incomming_layer = self.layers[n_lay + 1]
                current_layer.error = np.dot(incomming_layer.weights, incomming_layer.sigma)
                current_layer.sigma = current_layer.error * current_layer.activation_derivative(current_layer.last_activation)

        ## upgarade weights???
        ## jakiś wzorek weights += sigma * transposed input to layer * step
        ## skąd????

        for n_lay in range(len(self.layers)):
            current_layer = self.layers[n_lay]

            given = np.atleast_2d(In if n_lay == 0 else self.layers[n_lay - 1].last_activation)

            current_layer.weights += current_layer.sigma * given.T * step
            current_layer.constant += current_layer.sigma * step

        return

    def train_batch(self, X, Y, step, max_cycles, debug_info=True):
        """!
        @brief Batch training of the model
        @param X [array-like objects] Inputs
        @param Y [array-like objects] Expected Outputs
        @param step [float] Learning rate coefficiant
        @param max_iterations [int] Maximal amount of iterations
        @return [array-like object] The MSE report
        """

        report = []
        if debug_info: print(f"Orginal MSE: {np.mean(np.square(Y - self.predict(X)))}")
        for no in range(max_cycles):
            for x in range(len(X)):
                self.backpropagation(X[x], Y[x], step)
            if no % 5 == 0:
                mse = np.mean(np.square(Y - self.predict(X)))
                report.append(mse)
                if debug_info: print(f"Cycle: {no}, MSE: {mse}")


def f(x):
    return x**2*np.sin(x)+100*np.sin(x)*np.cos(x)

if __name__ == "__main__":
    n = NNetwork()

    # """Średnio działa"""
    n.add_layer(Layer(1, 1, 'tanh', weights=np.array([[-0.3]]), constant=np.array([1], np.float64)))
    n.add_layer(Layer(20, 1, 'tanh'))
    n.add_layer(Layer(20, 20, 'tanh'))
    n.add_layer(Layer(1, 20, weights=np.zeros((20, 1))))
    sample = 80 * np.random.rand(1000) - 40
    answer = [[f(i)] for i in sample]
    sample = [[i] for i in sample]
    n.train_batch(sample, answer, 0.2, 1000)
    n.train_batch(sample, answer, 0.2, 1000)


    """Dziwnie działa"""
    # n.add_layer(Layer(1, 1, 'tanh', weights=np.array([[-0.3]]), constant=np.array([1], np.float64)))
    # n.add_layer(Layer(2, 1, 'tanh', weights=np.array([[0.2, -0.5]]), constant=np.array([1, 1], np.float64)))
    # n.add_layer(Layer(1, 2, weights=np.zeros((2, 1)), constant=np.array([1], np.float64)))

    # samples = []

    # n.backpropagation([3], [f(3)], 0.5)

    # for c in range(100000):
    #     sample = np.array([80 * np.random.rand() - 40])
    #     samples.append(sample)
    #     answer = np.array(f(sample))
    #     n.train_batch(sample, answer, 0.5, 1)

    #     if c % 1000 == 0:
    #         print(f"Cycle: {c}, Total MSE: {np.mean([np.square(f(s)-n.predict(s)) for s in samples])}, MSE in current data: {np.mean(np.square(answer-n.predict(sample)))}")


    """Działa"""
    # n.add_layer(Layer(2, 2, 'sigmoid', np.array([[0.15, 0.25], [0.2, 0.3]]), np.array([0.35, 0.35], np.float64))) #[neuron1], [neuron2]
    # n.add_layer(Layer(2, 2, 'sigmoid', np.array([[0.4, 0.5], [0.45, 0.55]]), np.array([0.6, 0.6], np.float64))) #[neuron1], [neuron2]
    # n.add_layer(Layer(2, 2, weights=np.zeros((2, 2)), constant=np.array([1, 1], np.float64)))
    # print(n.predict(np.array([0.05, 0.1])))

    # n.train_batch([[0.05, 0.1]], [[0.01, 0.99]], 0.5, 10000, debug_info=False)
    # print(n.predict(np.array([0.05, 0.1])))
