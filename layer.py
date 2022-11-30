from activation import Activation
from abc import ABC, abstractmethod
import numpy as np
from initialization import GaussianInitialization

class Layer(ABC):

    def __init__(self, weights=np.zeros(0), bias=1) -> None:
        self.weights = weights
        self.bias = bias

    @abstractmethod
    def forward_propagate(self, input:np.ndarray):
        assert input.size == self.weights.size

        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias

        return self.output

    @abstractmethod
    def back_propagate(self, error:float, learning_rate:float):
        input_error = np.dot(error, self.weights.T)
        weights_error = np.dot(self.input.T, error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * error
        return input_error


class InputLayer(Layer):
    
    def __init__(self, input_size:int) -> None:
        super().__init__(input_size, 0)

    def forward_propagate(self, input):
        super().forward_propagate(input)

    def back_propagate(self, error, learning_rate):
        super().back_propagate(error, learning_rate)
        

class HiddenLayer(Layer):

    def __init__(self, input_size:int, output_size:int) -> None:
        gi = GaussianInitialization(input_size)
        self.input = gi.get()
        gi.size = output_size
        self.output = gi.get()

    def forward_propagate(self, input):
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def back_propagate(self, error, learning_rate):
        input_error = np.dot(error, self.weights.T)
        weights_error = np.dot(self.input.T, error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * error
        return input_error
    