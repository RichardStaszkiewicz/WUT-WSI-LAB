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
                current_layer.error = Target - predicted
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

            change = current_layer.sigma * given.T * step
            current_layer.weights += current_layer.sigma * given.T * step

        return


if __name__ == "__main__":
    n = NNetwork()
    n.add_layer(Layer(5, 1, 'sigmoid', seed=10))
    n.add_layer(Layer(5, 5, 'sigmoid', seed=2))
    n.add_layer(Layer(1, 5, 'sigmoid', seed=5, weights=np.zeros((5, 1))))
    print(n.predict([-9228]))
    for i in range(1000): n.backpropagation([-9228], [21], 0.8)
    print(n.predict([-9228]))
