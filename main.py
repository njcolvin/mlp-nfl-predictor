from sklearn.neural_network import MLPRegressor
import numpy as np
from network import Network
from activation import ReLUActivation, SigmoidActivation, TanhActivation
from initialization import UniformInitialization, GaussianInitialization
from sklearn.linear_model import LinearRegression
from preprocess import standardize, build_dataset
from postprocess import caluclate_profit_sklearn, caluclate_profit_network
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def main():

    # train test split
    Xtrn = np.genfromtxt('./data/train-x', missing_values=0, skip_header=0, delimiter=' ', dtype=float)
    Xtst = np.genfromtxt('./data/test-x', missing_values=0, skip_header=0, delimiter=' ', dtype=float)
    Ytrn = np.genfromtxt('./data/train-y', missing_values=0, skip_header=0, delimiter=' ', dtype=int)
    Ytst = np.genfromtxt('./data/test-y', missing_values=0, skip_header=0, delimiter=' ', dtype=int)

    # get averages, standard devs
    xtst_mu, xtst_sigma = np.mean(Xtst, axis=0), np.std(Xtst, axis=0)
    ytst_mu, ytst_sigma = np.mean(Ytst, axis=0), np.std(Ytst, axis=0)

    # standardize data
    Xtrn, Xtst, Ytrn, Ytst = standardize(Xtrn), standardize(Xtst), standardize(Ytrn), standardize(Ytst)

    # linear regression
    print("linear regression")

    linear = LinearRegression().fit(Xtrn, Ytrn)
    lin_pred = linear.predict(Xtst)

    p = caluclate_profit_sklearn(Xtst * xtst_sigma + xtst_mu, lin_pred * ytst_sigma + ytst_mu, Ytst * ytst_sigma + ytst_mu)

    s = linear.score(Xtst, Ytst)

    print("profit: " + str(np.sum(p)))
    print("score: " + str(s))
    
    # sample for tonight's game

    print("")
    
    # sklearn MLPRegressor
    print("MLPRegressor")

    avg_profit = []
    avg_score = []
    for _ in range(10):
        nn = MLPRegressor(hidden_layer_sizes=(100,32,), max_iter=100, learning_rate_init=0.003).fit(Xtrn, Ytrn)
        nn_pred = nn.predict(Xtst)
    
        p = caluclate_profit_sklearn(Xtst * xtst_sigma + xtst_mu, nn_pred * ytst_sigma + ytst_mu, Ytst * ytst_sigma + ytst_mu)

        s = nn.score(Xtst, Ytst)
        print('s: ' + str(s))
        avg_profit.append(p)
        avg_score.append(s)

    print("avg profit: " + str(np.mean(avg_profit, axis=0)))
    print("avg score: " + str(np.mean(avg_score, axis=0)))
    
    print("")
    
    # own implementation
    print("own implementation")

    # train test split

    Xtrn = np.genfromtxt('./data/train-x', missing_values=0, skip_header=0, delimiter=' ', dtype=float)
    Xtst = np.genfromtxt('./data/test-x', missing_values=0, skip_header=0, delimiter=' ', dtype=float)
    Ytrn = np.genfromtxt('./data/train-y', missing_values=0, skip_header=0, delimiter=' ', dtype=int)
    Ytst = np.genfromtxt('./data/test-y', missing_values=0, skip_header=0, delimiter=' ', dtype=int)

    Ytrn = np.array([[[row[0], row[1]]] for row in Ytrn])
    Xtrn = np.array([[[row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],row[8], row[9]]] for row in Xtrn])
    Ytst = np.array([[[row[0], row[1]]] for row in Ytst])
    Xtst = np.array([[[row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7],row[8], row[9]]] for row in Xtst])

    # get averages, standard devs
    xtst_mu, xtst_sigma = np.mean(Xtst, axis=0), np.std(Xtst, axis=0)
    ytst_mu, ytst_sigma = np.mean(Ytst, axis=0), np.std(Ytst, axis=0)

    # standardize data
    Xtrn, Xtst, Ytrn, Ytst = standardize(Xtrn), standardize(Xtst), standardize(Ytrn), standardize(Ytst)

    better_nn = Network(hidden_layer_configs=((100, ReLUActivation),), max_iter=1000, learning_rate=0.003, initialization=GaussianInitialization())
    better_nn.fit(Xtrn, Ytrn)
    y_pred = better_nn.predict(Xtst)

    p = caluclate_profit_network(Xtst * xtst_sigma + xtst_mu, y_pred * ytst_sigma + ytst_mu, Ytst * ytst_sigma + ytst_mu, include_winloss=False)

    Xtst_2d = np.genfromtxt('./data/test-x', missing_values=0, skip_header=0, delimiter=' ', dtype=float)
    ytst_2d = np.genfromtxt('./data/test-y', missing_values=0, skip_header=0, delimiter=' ', dtype=float)

    xtst_mu, xtst_sigma = np.mean(Xtst_2d, axis=0), np.std(Xtst_2d, axis=0)
    ytst_mu, ytst_sigma = np.mean(ytst_2d, axis=0), np.std(ytst_2d, axis=0)
    
    Xtst_2d = standardize(Xtst_2d)
    y_pred_2d = np.array([[row[0][0], row[0][1]] for row in y_pred])
    y_pred_2d = y_pred_2d * ytst_sigma + ytst_mu
    y_pred_2d = np.array([[int(row[0]), int(row[1])] for row in y_pred_2d])

    print("r2: " + str(r2_score(ytst_2d, y_pred_2d)))
    
    for i in range(100):
        plt.scatter(y_pred_2d[i,0], y_pred_2d[i,1], color ='b')
        plt.scatter(ytst_2d[i,0], ytst_2d[i,1], color ='r')
        plt.plot([y_pred_2d[i,0], ytst_2d[i,0]], [y_pred_2d[i,1], ytst_2d[i,1]])
    plt.show()
    
main()