from layer import Layer
import numpy as np

class Activation(Layer):

    def __init__(self, function, derivative) -> None:
        self.function = function
        self.derivative = derivative

    def forward_propagate(self, input):
        self.input = input
        self.output = self.function(input)
        return self.output

    def back_propagate(self, de_dy, learning_rate):
        return de_dy * self.derivative(self.input)

class IdentityActivation(Activation):
    def __init__(self) -> None:
        super().__init__(lambda x: x, lambda x: 1)

class SigmoidActivation(Activation):
    def __init__(self) -> None:
        super().__init__(sigmoid, sigmoid_derivative)

class ReLUActivation(Activation):
    def __init__(self) -> None:
        super().__init__(relu, relu_derivative)

class TanhActivation(Activation):
    def __init__(self) -> None:
        super().__init__(tanh, tanh_derivative)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_derivative(x):
    y = x
    y[y > 0] = 1
    y[y <= 0] = 0
    return y

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    t = tanh(x)
    return 1 - t ** 2