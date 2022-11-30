from layer import Layer
import math

class Activation(Layer):

    def __init__(self, activation, activation_gradient) -> None:
        self.function = activation
        self.gradient = activation_gradient

    def forward_propagate(self, input):
        self.input = input
        self.output = self.function(input)
        return self.output

    def back_propagate(self, error, learning_rate):
        return error * self.gradient(self.input)

class SigmoidActivation(Activation):

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def sigmoid_gradient(self, x):
        s = self.sigmoid(x)
        return s * (1 - s)

    def __init__(self) -> None:
        super().__init__(self.sigmoid, self.sigmoid_gradient)

class LinearActivation(Activation):

    def linear(self, x):
        return self.k * x

    def linear_gradient(self):
        return self.k

    def __init__(self, k) -> None:
        self.k = k
        super().__init__(self.linear, self.linear_gradient)

class ReLUActivation(Activation):

    def relu(self, x):
        return math.max([0, x])

    def relu_gradient(self, x):
        if x > 0:
            return 1
        else:
            return 0

    def __init__(self, k) -> None:
        self.k = k
        super().__init__(self.relu, self.relu_gradient)

class TanhActivation(Activation):

    def tanh(self, x):
        return math.tanh(x)

    def tanh_gradient(self, x):
        t = self.tanh(x)
        return 1 - t ** 2

    def __init__(self, k) -> None:
        self.k = k
        super().__init__(self.tanh, self.tanh_gradient)