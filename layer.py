import numpy as np
from initialization import Initialization

class Layer():

    def __init__(self, initialization:Initialization) -> None:
        self.weights = initialization.get_weights()
        self.bias = initialization.get_bias()

    def forward_propagate(self, input:np.ndarray) -> np.ndarray:
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.output

    def back_propagate(self, de_dy:float, learning_rate:float) -> float:
        de_dx = np.dot(de_dy, self.weights.T)
        de_dw = np.dot(self.input.T, de_dy)
        self.weights -= learning_rate * de_dw
        self.bias -= learning_rate * de_dy
        return de_dx