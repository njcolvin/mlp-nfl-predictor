from loss import Loss, MSE, MAE
from layer import Layer
from initialization import Initialization, GaussianInitialization, UniformInitialization
from activation import Activation, SigmoidActivation, TanhActivation, ReLUActivation, IdentityActivation
import plotly.express as px
import numpy as np
from numpy import linalg as LA
import pandas as pd

class Network:

    def __init__(self, hidden_layer_configs:tuple=((100, ReLUActivation()),), max_iter=200, initialization:Initialization=GaussianInitialization(), loss:Loss=MAE(), learning_rate:float=0.001):
        
        assert max_iter > 0
        assert learning_rate > 0
        self.layers = []
        self.hidden_layer_configs = hidden_layer_configs
        self.max_iter = max_iter
        self.initialization = initialization
        self.loss = loss
        self.learning_rate = learning_rate

    def predict(self, samples:list):
        n = len(samples)
        result = []

        for i in range(n):
            output = samples[i]
            for layer in self.layers:
                output = layer.forward_propagate(output)
            result.append(output)

        return result

    def fit(self, x_train, y_train):
        n = len(x_train)
        assert n > 0
        assert len(y_train) == n

        self.__build_network(x_train, y_train)

        for i in range(self.max_iter):
            err = 0

            for j in range(n):
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagate(output)

                err += self.loss.forward_propagate(output, y_train[j])

                de_dy = self.loss.back_propagate(output, y_train[j])
                for layer in reversed(self.layers):
                    de_dy = layer.back_propagate(de_dy, self.learning_rate)

            err /= n
            print('iter %d/%d   error=%f' % (i + 1, self.max_iter, err))

    def __build_network(self, x, y):
        num_hidden_layers = len(self.hidden_layer_configs)
        assert num_hidden_layers > 0

        # build network

        # input layer
        self.initialization.set_layer_size(len(x[0][0]), self.hidden_layer_configs[0][0])
        self.layers.append(Layer(self.initialization))
        # activation
        self.layers.append(self.hidden_layer_configs[0][1]())
        # hidden layer(s)
        for i in range(1, num_hidden_layers):
            self.initialization.set_layer_size(self.hidden_layer_configs[i - 1][0], self.hidden_layer_configs[i][0])
            self.layers.append(Layer(self.initialization))
            self.layers.append(self.hidden_layer_configs[i][1]())
        # output layer
        self.initialization.set_layer_size(self.hidden_layer_configs[num_hidden_layers - 1][0], len(y[0][0]))
        self.layers.append(Layer(self.initialization))

        # ReLU cannot be used to activate output layer
        if self.hidden_layer_configs[num_hidden_layers - 1][1] == ReLUActivation:
            self.layers.append(IdentityActivation())
        else:
            self.layers.append(self.hidden_layer_configs[num_hidden_layers - 1][1]())

    def visualize(self, x, y, y_sigma, y_mu, num_predictions=25, add_index_scores=False):
        n = len(x)
        
        predictions = self.predict(x)

        d = {'index': [], 'feature_name': [], 'value': []}
        names = ['t0_adv', 't0_name', 't0_ml', 't1_adv', 't1_name', 't1_ml', 's_o', 'ou_o', 'ou_c']
        for i in range(n): # for each sample
            for j in range(len(x[i][0])): # for each weight
                d['index'].append(i)
                d['feature_name'].append(names[j])
                d['value'].append(LA.norm(np.dot(x[i][0][j], self.layers[0].weights[j])))
        
        df = pd.DataFrame(data=d)

        # get best predictions
        diffs = []
        for i in range(n):
            diff = LA.norm(y[i][0] - predictions[i][0])
            diffs.append((i, diff))

        diffs.sort(key=lambda x: x[1])

        # get values for num_predictions best predictions
        # put values as text in values_plus_realindex
        # also put game index and real, predicted scores in last entry of values_plus_realindex if add_index_scores is true
        d2 = {'index': [], 'feature_name': [], 'value': []}
        values_plus_realindex = []
        for i in range(num_predictions):
            index = diffs[i][0] * 9
            rows = df.loc[df['index'] == diffs[i][0]]
            for j in range(len(rows) - 1):
                values_plus_realindex.append(round(rows['value'][j + index], 3))
    
            if add_index_scores:
                # gross notation warning
                predictions[diffs[i][0]][0][0] = int(predictions[diffs[i][0]][0][0] * y_sigma[0] + y_mu[0])
                predictions[diffs[i][0]][0][1] = int(predictions[diffs[i][0]][0][1] * y_sigma[1] + y_mu[1])
                values_plus_realindex.append(str(round(rows['value'][len(rows) - 1 + index], 3)) + '\ngame ' + str(diffs[i][0]) + '\npred=' +
                                         str(predictions[diffs[i][0]][0]) + '\nreal=' + str((y[diffs[i][0]] * y_sigma + y_mu)[0]))
            else:
                values_plus_realindex.append(round(rows['value'][len(rows) - 1 + index], 3))
            
            # append best values
            for j in range(index, index + len(rows)):
                d2['index'].append(i)
                d2['feature_name'].append(rows['feature_name'][j])
                d2['value'].append(rows['value'][j])

        df2 = pd.DataFrame(data=d2)
        fig = px.bar(df2, x='index', y='value', color='feature_name', text=values_plus_realindex)

        fig.show()
