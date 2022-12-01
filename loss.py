import numpy as np

class Loss():

    def __init__(self, function, derivative) -> None:
        self.function = function
        self.derivative = derivative

    def forward_propagate(self, y_pred:np.ndarray, y_true:np.ndarray) -> float:
        return self.function(y_pred, y_true)

    def back_propagate(self, y_pred:np.ndarray, y_true:np.ndarray) -> float:
        return self.derivative(y_pred, y_true)

class MSE(Loss):

    def mse(self, y_pred, y_true):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_derivative(self, y_pred, y_true):
        return 2 * (y_pred - y_true) / y_pred.size

    def __init__(self) -> None:
        super().__init__(self.mse, self.mse_derivative)

