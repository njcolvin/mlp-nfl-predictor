from layer import Layer
import math
import numpy as np

class Activation(Layer):

    def __init__(self, function, derivative) -> None:
        self.function = function
        self.derivative = derivative

    def forward_propagate(self, input:np.ndarray) -> np.ndarray:
        self.input = input
        self.output = self.function(input)
        return self.output

    def back_propagate(self, de_dy:float, learning_rate:float) -> float:
        return de_dy * self.derivative(self.input)

class SigmoidActivation(Activation):

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def __init__(self) -> None:
        super().__init__(self.sigmoid, self.sigmoid_derivative)

class ReLUActivation(Activation):

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        y = np.maximum(0, x)
        y[y >= 0] = 1
        return y

    def __init__(self) -> None:
        super().__init__(self.relu, self.relu_derivative)

class TanhActivation(Activation):

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        t = self.tanh(x)
        return 1 - t ** 2

    def __init__(self) -> None:
        super().__init__(self.tanh, self.tanh_derivative)