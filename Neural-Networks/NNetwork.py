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

    def __init__(self) -> None:
        """!
        @brief
        Constructs a Neural network with no layers.
        @return NNetwork object
        """
        self.layers = []

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


if __name__ == "__main__":
    n = NNetwork()
    n.add_layer(Layer(5, 1, 'tanh'))
    n.add_layer(Layer(1, 5, 'sigmoid'))
    print(n.predict([1]))