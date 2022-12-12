from sklearn.neural_network import MLPRegressor
import numpy as np
from network import Network
from activation import ReLUActivation, SigmoidActivation, TanhActivation
from initialization import UniformInitialization, GaussianInitialization
from sklearn.linear_model import LinearRegression
from preprocess import standardize, build_dataset
from postprocess import caluclate_profit
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def main():

    # train test split
    """ Xtrn = np.genfromtxt('./data/train-x', missing_values=0, skip_header=0, delimiter=' ', dtype=float)
    Xtst = np.genfromtxt('./data/test-x', missing_values=0, skip_header=0, delimiter=' ', dtype=float)
    Ytrn = np.genfromtxt('./data/train-y', missing_values=0, skip_header=0, delimiter=' ', dtype=int)
    Ytst = np.genfromtxt('./data/test-y', missing_values=0, skip_header=0, delimiter=' ', dtype=int)

    Xtst_orig = Xtst

    Xtrn = np.delete(Xtrn, 9, axis=1)
    Xtst = np.delete(Xtst, 9, axis=1)

    # get averages, standard devs
    ytst_mu, ytst_sigma = np.mean(Ytst, axis=0), np.std(Ytst, axis=0)

    # standardize data
    Xtrn, Xtst, Ytrn, Ytst = standardize(Xtrn), standardize(Xtst), standardize(Ytrn), standardize(Ytst)

    print("MLPRegressor")

    avg_ml_record, avg_spread_record, avg_ou_record, avg_ml_profit, avg_spread_profit, avg_ou_profit, avg_r2, avg_mae, avg_mse = [], [], [], [], [], [], [], [], []
    for _ in range(10):
        nn = MLPRegressor(hidden_layer_sizes=(100,), max_iter=100, learning_rate_init=0.003).fit(Xtrn, Ytrn)
        nn_pred = nn.predict(Xtst)
        ml_record, spread_record, ou_record, ml_profit, spread_profit, ou_profit = caluclate_profit(Xtst_orig, nn_pred * ytst_sigma + ytst_mu, Ytst * ytst_sigma + ytst_mu)
        avg_ml_record.append(ml_record)
        avg_spread_record.append(spread_record)
        avg_ou_record.append(ou_record)
        avg_ml_profit.append(ml_profit)
        avg_spread_profit.append(spread_profit)
        avg_ou_profit.append(ou_profit)
        s = nn.score(Xtst, Ytst)
        mae = mean_absolute_error(Ytst, nn_pred)
        mse = mean_squared_error(Ytst, nn_pred)
        avg_r2.append(s)
        avg_mae.append(mae)
        avg_mse.append(mse)
        print("score: " + str(s))
        print("mae: " +str(mae))
        print("mse: " +str(mse))

    print("avg ml record: " + str(np.mean(avg_ml_record, axis=0)))
    print("stddev ml record: " + str(np.std(avg_ml_record, axis=0)))

    print("avg spread record: " + str(np.mean(avg_spread_record, axis=0)))
    print("stddev spread record: " + str(np.std(avg_spread_record, axis=0)))

    print("avg ou record: " + str(np.mean(avg_ou_record, axis=0)))
    print("stddev ou record: " + str(np.std(avg_ou_record, axis=0)))

    print("avg ml profit: " + str(np.mean(avg_ml_profit, axis=0)))
    print("stddev ml profit: " + str(np.std(avg_ml_profit, axis=0)))

    print("avg spread profit: " + str(np.mean(avg_spread_profit, axis=0)))
    print("stddev spread profit: " + str(np.std(avg_spread_profit, axis=0)))

    print("avg ou profit: " + str(np.mean(avg_ou_profit, axis=0)))
    print("stddev ou profit: " + str(np.std(avg_ou_profit, axis=0)))

    print("avg r2: " + str(np.mean(avg_r2, axis=0)))
    print("stddev r2: " + str(np.std(avg_r2, axis=0)))

    print("avg mae: " + str(np.mean(avg_mae, axis=0)))
    print("stddev mae: " + str(np.std(avg_mae, axis=0)))

    print("avg mse: " + str(np.mean(avg_mse, axis=0)))
    print("stddev mse: " + str(np.std(avg_mse, axis=0))) """

    
    # own implementation
    print("own implementation")
    # train test split

    Xtrn = np.genfromtxt('./data/train-x', missing_values=0, skip_header=0, delimiter=' ', dtype=float)
    Xtst = np.genfromtxt('./data/test-x', missing_values=0, skip_header=0, delimiter=' ', dtype=float)
    Ytrn = np.genfromtxt('./data/train-y', missing_values=0, skip_header=0, delimiter=' ', dtype=int)
    Ytst = np.genfromtxt('./data/test-y', missing_values=0, skip_header=0, delimiter=' ', dtype=int)

    Xtst_orig = Xtst

    Xtrn = np.delete(Xtrn, 7, axis=1)
    Xtst = np.delete(Xtst, 7, axis=1)

    #Xtrn = np.delete(Xtrn, 9, axis=1)
    #Xtst = np.delete(Xtst, 9, axis=1)

    # add dimension for network dot product
    Ytrn_3d = np.array([[[row[0], row[1]]] for row in Ytrn])
    Ytst_3d = np.array([[[row[0], row[1]]] for row in Ytst])
    Xtrn_3d = np.array([[[row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]]] for row in Xtrn])
    Xtst_3d = np.array([[[row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]]] for row in Xtst])

    # get averages, standard devs
    ytst_mu, ytst_sigma = np.mean(Ytst, axis=0), np.std(Ytst, axis=0)

    # standardize data
    Xtrn_3d, Xtst_3d, Ytrn_3d, Ytst, Ytst_3d = standardize(Xtrn_3d), standardize(Xtst_3d), standardize(Ytrn_3d), standardize(Ytst), standardize(Ytst_3d)
    
    better_nn = Network(hidden_layer_configs=((100, ReLUActivation),), max_iter=100, learning_rate=0.003, initialization=GaussianInitialization())
    better_nn.fit(Xtrn_3d, Ytrn_3d)
    y_pred = better_nn.predict(Xtst_3d)
    better_nn.visualize(Xtst_3d, Ytst_3d, ytst_sigma, ytst_mu, add_index_scores=False)
    y_pred_2d = np.array([[row[0][0], row[0][1]] for row in y_pred])

    # fit predict compute error
    """ avg_ml_record, avg_spread_record, avg_ou_record, avg_ml_profit, avg_spread_profit, avg_ou_profit, avg_r2, avg_mae, avg_mse = [], [], [], [], [], [], [], [], []
    for _ in range(10):
        better_nn = Network(hidden_layer_configs=((100, ReLUActivation),), max_iter=100, learning_rate=0.003, initialization=GaussianInitialization())
        better_nn.fit(Xtrn_3d, Ytrn_3d)
        y_pred = better_nn.predict(Xtst_3d)
        better_nn.visualize(Xtst_3d, Ytst_3d, ytst_sigma, ytst_mu, add_index_scores=True)

        y_pred_2d = np.array([[row[0][0], row[0][1]] for row in y_pred])
        ml_record, spread_record, ou_record, ml_profit, spread_profit, ou_profit = caluclate_profit(Xtst_orig, y_pred_2d * ytst_sigma + ytst_mu, Ytst * ytst_sigma + ytst_mu)
        avg_ml_record.append(ml_record)
        avg_spread_record.append(spread_record)
        avg_ou_record.append(ou_record)
        avg_ml_profit.append(ml_profit)
        avg_spread_profit.append(spread_profit)
        avg_ou_profit.append(ou_profit)
        s = r2_score(Ytst, y_pred_2d)
        mae = mean_absolute_error(Ytst, y_pred_2d)
        mse = mean_squared_error(Ytst, y_pred_2d)
        avg_r2.append(s)
        avg_mae.append(mae)
        avg_mse.append(mse)
        print("score: " + str(s))
        print("mae: " +str(mae))
        print("mse: " +str(mse))

    print("avg ml record: " + str(np.mean(avg_ml_record, axis=0)))
    print("stddev ml record: " + str(np.std(avg_ml_record, axis=0)))

    print("avg spread record: " + str(np.mean(avg_spread_record, axis=0)))
    print("stddev spread record: " + str(np.std(avg_spread_record, axis=0)))

    print("avg ou record: " + str(np.mean(avg_ou_record, axis=0)))
    print("stddev ou record: " + str(np.std(avg_ou_record, axis=0)))

    print("avg ml profit: " + str(np.mean(avg_ml_profit, axis=0)))
    print("stddev ml profit: " + str(np.std(avg_ml_profit, axis=0)))

    print("avg spread profit: " + str(np.mean(avg_spread_profit, axis=0)))
    print("stddev spread profit: " + str(np.std(avg_spread_profit, axis=0)))

    print("avg ou profit: " + str(np.mean(avg_ou_profit, axis=0)))
    print("stddev ou profit: " + str(np.std(avg_ou_profit, axis=0)))

    print("avg r2: " + str(np.mean(avg_r2, axis=0)))
    print("stddev r2: " + str(np.std(avg_r2, axis=0)))

    print("avg mae: " + str(np.mean(avg_mae, axis=0)))
    print("stddev mae: " + str(np.std(avg_mae, axis=0)))

    print("avg mse: " + str(np.mean(avg_mse, axis=0)))
    print("stddev mse: " + str(np.std(avg_mse, axis=0))) """


main()