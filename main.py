from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from network import Network
from activation import ReLUActivation, SigmoidActivation, TanhActivation
from initialization import UniformInitialization, GaussianInitialization
from sklearn.linear_model import LinearRegression
from preprocess import standardize, build_dataset
from postprocess import caluclate_profit_sklearn, caluclate_profit_network

def main():
    """ 
    M = np.genfromtxt('./data/data-no-wl', missing_values=0, skip_header=1, delimiter=' ', dtype=object)
    
    y = [[row[len(M[0]) - 2], row[len(M[0]) - 1]] for row in M]
    x = [row[1:len(M[0]) - 2] for row in M]
    y = np.array(y)
    x = np.array(x)

    

    # encode team names
    encoder = LabelEncoder()
    x[:, 1] = encoder.fit_transform(x[:, 1])
    x[:, 4] = encoder.transform(x[:, 4])

    # set dtype to float
    x = x.astype('float64')
    y = y.astype('float64')
    """

    # train test split
    #Xtrn, Xtst, Ytrn, Ytst = train_test_split(x, y)
    Xtrn = np.genfromtxt('./data/train-x', missing_values=0, skip_header=0, delimiter=' ', dtype=float)
    Xtst = np.genfromtxt('./data/test-x', missing_values=0, skip_header=0, delimiter=' ', dtype=float)
    Ytrn = np.genfromtxt('./data/train-y', missing_values=0, skip_header=0, delimiter=' ', dtype=int)
    Ytst = np.genfromtxt('./data/test-y', missing_values=0, skip_header=0, delimiter=' ', dtype=int)

    """     np.savetxt('data/train-x', Xtrn, fmt='%d %d %d %d %d %d %1.1f %1.1f %1.1f %1.1f')
    np.savetxt('data/train-y', Ytrn, fmt='%d %d')
    
    np.savetxt('data/test-x', Xtst, fmt='%d %d %d %d %d %d %1.1f %1.1f %1.1f %1.1f')
    np.savetxt('data/test-y', Ytst, fmt='%d %d') """

    # get averages, standard devs
    xtst_mu, xtst_sigma = np.mean(Xtst, axis=0), np.std(Xtst, axis=0)
    ytst_mu, ytst_sigma = np.mean(Ytst, axis=0), np.std(Ytst, axis=0)

    # standardize data
    Xtrn, Xtst, Ytrn, Ytst = standardize(Xtrn), standardize(Xtst), standardize(Ytrn), standardize(Ytst)

    # linear regression
    print("linear regression")

    linear = LinearRegression().fit(Xtrn, Ytrn)
    lin_pred = linear.predict(Xtst) * ytst_sigma + ytst_mu

    p = caluclate_profit_sklearn(Xtst * xtst_sigma + xtst_mu, lin_pred, Ytst * ytst_sigma + ytst_mu)

    s = linear.score(Xtst, Ytst)

    print("profit: " + str(np.sum(p)))
    print("score: " + str(s))
    
    # sample for tonight's game
    #sample = [-1, encoder.transform([b'Miami'])[0], 174, 1, encoder.transform([b'LAChargers'])[0], -140, -5.5, 3, 51.5, 54.5]
    #sample = (sample - xtst_mu) / xtst_sigma
    #y_sample = linear.predict([sample])
    #print("MIA @ LAC: " + str(y_sample * ytst_sigma + ytst_mu))

    print("")
    
    # sklearn MLPRegressor
    print("MLPRegressor")

    avg_profit = []
    avg_score = []
    for _ in range(10):
        nn = MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, learning_rate_init=0.003).fit(Xtrn, Ytrn)
        y_pred = nn.predict(Xtst) * ytst_sigma + ytst_mu
        
        p = caluclate_profit_sklearn(Xtst * xtst_sigma + xtst_mu, y_pred, Ytst * ytst_sigma + ytst_mu)

        #sample = [-1, encoder.transform([b'Miami'])[0], 174, 1, encoder.transform([b'LAChargers'])[0], -140, -5.5, 3, 51.5, 54.5]
        #sample = (sample - xtst_mu) / xtst_sigma
        #y_sample = nn.predict([sample])
        #print("MIA @ LAC: " + str(y_sample * ytst_sigma + ytst_mu))

        s = nn.score(Xtst, Ytst)
        avg_profit.append(p)
        avg_score.append(s)

    print("avg profit: " + str(np.mean(avg_profit, axis=0)))
    print("avg score: " + str(np.mean(avg_score, axis=0)))
    
    print("")
    
    # own implementation
    print("own implementation")

    """ 
    y = np.array([[[row[len(M[0]) - 2], row[len(M[0]) - 1]]] for row in M])
    x = np.array([[row[1:len(M[0]) - 2]] for row in M])

    # encode team names
    encoder = LabelEncoder()
    x[:, 0, 1] = encoder.fit_transform(x[:, 0, 1])
    x[:, 0, 4] = encoder.transform(x[:, 0, 4])

    # set dtype to float
    x = x.astype('float64')
    y = y.astype('float64')
    """

    # train test split
    # Xtrn, Xtst, Ytrn, Ytst = train_test_split(x, y)

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
    
    #sample = [-1, encoder.transform([b'Miami'])[0], 174, 1, encoder.transform([b'LAChargers'])[0], -140, 0, 8.5, 51.5, 54.5]
    #sample = (sample - xtst_mu) / xtst_sigma
    #y_sample = better_nn.predict([sample])
    #print("MIA @ LAC: " + str(y_sample * ytst_sigma + ytst_mu))

main()