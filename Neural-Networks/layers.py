"""!
@file layers.py
@brief Implements only one class representing Layer.
@version 0.1
@section requirements
Numpy version >= 1.19.4
"""

import numpy as np

class Layer(object):
    """!
    @brief
    Represent a layer of neural network.
    """

    def __init__(self, neurons_amount :int, input_amount :int, activation=None, weights=None, constant=None, afunctions=None, dafunctions=None, seed=None) -> None:
        """!
        @param neurons_amount [int] Amount of neurons in the layer.
        @param input_amount [int] Amount of layer inputs.
        @param activation [str] Activation fuction to be used.
        @param weights [array-like object] Matrix of weights in layer (M[neurons][input]).
        @param constant [array-like object] List of constants for each neuron (L[neurons]).
        @param afunctions [dict] Map of strings to fuctions of X containing possible activation functions. Default are: None, sigmoid and tanh.
        @param dafunctions [dict] Map of strings to functions implementing derivatives of possible layers activation functions. Default are: None, tanh and sigmoid.
        @param seed [int] Random seed of the layer for result repetivness.
        @return [Layer] Initialized class.
        """
        np.random.seed(1 if seed is None else seed)

        self.activation = f"{activation}"

        if weights is not None: self.weights = weights
        else: self.weights = np.random.uniform(-1/input_amount, 1/input_amount, (input_amount, neurons_amount))#np.random.rand(input_amount, neurons_amount)

        if constant is not None: self.constant = constant
        else: self.constant = np.random.rand(neurons_amount)

        if afunctions is not None: self.afunctions = afunctions
        else: self.afunctions = {
            "None": lambda x: x,
            "tanh": lambda x: np.tanh(x),
            "sigmoid": lambda x: 1 / (1 + np.exp(-x))
        }

        if dafunctions is not None: self.dafunctions = dafunctions
        else: self.dafunctions = {
            "None": lambda x: np.array([1 for i in x], np.float64),
            "tanh": lambda x: 1 - x**2,
            "sigmoid": lambda x: x * (1 - x)
        }

        if self.activation not in self.afunctions.keys(): self.activation = "None"


    def activate(self, In):
        """!
        @brief Activates the neuron. It comes down to activating the dot product of
        the layer. Actualises parameter last_activation.
        @param In [array-like object] Input of the layer.
        @return [array-like object] Output of the layer. Mind the outputs amount is equal to the one of neurons in layer.
        """

        summator = np.dot(In, self.weights)
        summator += self.constant
        self.last_activation = self.__activatef(summator)
        return self.last_activation


    def __activatef(self, values):
        """!
        @brief Activates neuron activation function.

        @param values [array-like object] Values of neurons to activate.
        @return [array-like object] Values of the neurons after activation.
        """
        return self.afunctions[self.activation](values)


    def activation_derivative(self, values):
        """!
        @brief Calculates an activation function derivative.

        @param values [array-like object] Values to count the derivative from.
        @return [array-like object] Value of derivative for input.
        """
        return self.dafunctions[self.activation](values)
