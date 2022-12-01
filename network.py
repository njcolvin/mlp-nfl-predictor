from loss import Loss, MSE
import numpy as np
from layer import Layer
from initialization import GaussianInitialization, UniformInitialization
from activation import SigmoidActivation, TanhActivation, ReLUActivation

class Network:
    def __init__(self, loss:Loss):
        self.layers = []
        self.loss = loss

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # predict output for given input
    def predict(self, samples):
        # sample dimension first
        n = len(samples)
        result = []

        # run network over all samples
        for i in range(n):
            # forward propagation
            output = samples[i]
            for layer in self.layers:
                output = layer.forward_propagate(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, learning_rate=0.1, max_iter=200):
        # sample dimension first
        n = len(x_train)

        # training loop
        for i in range(max_iter):
            err = 0
            for j in range(n):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagate(output)

                # compute loss (for display purpose only)
                err += self.loss.forward_propagate(output, y_train[j])

                # backward propagation
                de_dy = self.loss.back_propagate(output, y_train[j])
                for layer in reversed(self.layers):
                    de_dy = layer.back_propagate(de_dy, learning_rate)

            # calculate average error on all samples
            err /= n
            print('iter %d/%d   error=%f' % (i+1, max_iter, err))


def main():
    x_train = np.array([[[0,0]], [[0,1]], [[1,0]], [[1,1]]])
    y_train = np.array([[[0]], [[1]], [[1]], [[0]]])

    # network
    net = Network(MSE())
    net.add(Layer(UniformInitialization(2, 3)))
    net.add(ReLUActivation())
    net.add(Layer(UniformInitialization(3, 1)))
    net.add(ReLUActivation())

    # train
    net.fit(x_train, y_train, max_iter=2000, learning_rate=0.15)

    # test
    out = net.predict(x_train)
    print(out)

main()