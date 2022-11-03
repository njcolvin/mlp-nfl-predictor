from enum import Enum
import numpy as np
import math

class Activation(Enum):
    Sigmoid = "sigmoid"
    Linear = "linear"
    ReLU = "relu"
    Tanh = "tanh"

class Neuron:

    def __init__(self, is_input:bool=False, is_output:bool=False, activation:Activation=Activation.Sigmoid) -> None:
        if is_input and is_output:
            raise Exception("Cannot set input and output flags")
        
        self.is_input = is_input
        self.is_output = is_output
        self.activation_function = activation
        self.weights_in = np.array()
        self.weights_out = np.array()

    def activation(self):
        x = np.dot(self.weights_in, self.weights_out)

        if self.activation_function == Activation.Sigmoid:
            return 1 / (1 + math.exp(-x))
        elif self.activation_function == Activation.Linear:
            return x
        elif self.activation_function == Activation.ReLU:
            return max([0, x])
        else: # Tanh
            return math.tanh(x)

    