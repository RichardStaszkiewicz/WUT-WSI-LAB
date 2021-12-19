"""!
@file NNetwork.py
@brief Implements only one class representing Neural Network.
@section requirements
Numpy version >= 1.19.4\n
Layers version >= 0.1
"""

from layers import Layer


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
        if not isinstance(lay, list): raise NameError

        for l in lay:
            if not isinstance(l, Layer): raise NameError

        self.__layers = lay

