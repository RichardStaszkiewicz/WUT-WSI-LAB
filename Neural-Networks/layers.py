"""!
@package layers
Package implementing the layer of neural network.
@file layers.py
@brief Implements only one class representing Layer.
@section requirements
Numpy version >= 1.19.4
"""

import numpy as np

np.random.seed(1)

class Layer(object):
    """!
    @brief
    Represent a layer of neural network.
    Layer as a basic component is fully functional optimizing structure.
    """

    def __init__(self, neurons_amount :int, input_amount :int, activation=None, weights=None, constant=None, activation_functions=None) -> None:
        """
        @param neurons_amount [int] Amount of neurons in the layer.
        @param input_amount [int] Amount of layer inputs.
        @param activation [str] Activation fuction to be used.
        @param weights [array-like object] Matrix of weights in layer (M[neurons][input]).
        @param constant [array-like object] List of constants for each neuron (L[neurons]).
        @param activation_functions [dict] Map of strings to fuctions of X containing possible activation functions.
        @return [Layer] Initialized class.
        """
        super().__init__()

    def __activate(self, value):
        """
        Activates the neuron. It comes down to performing an layer's activation
        fuction on the given value.

        @param value [float] Value of the neuron to activate.
        @return [float] Value of the neuron after activation.
        """

        return value



