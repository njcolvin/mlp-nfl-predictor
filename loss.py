import numpy as np

class Loss():

    def __init__(self, function, derivative) -> None:
        self.function = function
        self.derivative = derivative

    def forward_propagate(self, y_pred:np.ndarray, y_true:np.ndarray):
        return self.function(y_pred, y_true)

    def back_propagate(self, y_pred:np.ndarray, y_true:np.ndarray):
        return self.derivative(y_pred, y_true)

class MSE(Loss):

    def __init__(self) -> None:
        super().__init__(mse, mse_derivative)

class MAE(Loss):

    def __init__(self) -> None:
        super().__init__(mae, mae_derivative)

def mse(y_pred, y_true):
    return np.mean(np.power(y_true - y_pred, 2))

def mse_derivative(y_pred, y_true):
    return 2 * (y_pred - y_true) / y_pred.size

def mae(y_pred, y_true):
    return np.mean(np.abs(y_true - y_pred))

def mae_derivative(y_pred, y_true):
    rtn = [[0, 0]]
    if y_true[0][0] > y_pred[0][0]:
        rtn[0][0] = -1
    elif y_true[0][0] < y_pred[0][0]:
        rtn[0][0] = 1
    if y_true[0][1] > y_pred[0][1]:
        rtn[0][1] = -1
    elif y_true[0][1] < y_pred[0][1]:
        rtn[0][1] = 1
    return np.array(rtn)