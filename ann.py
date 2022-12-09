import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from network import Network
from activation import ReLUActivation, SigmoidActivation, TanhActivation
from initialization import UniformInitialization, GaussianInitialization
from sklearn.linear_model import LinearRegression
from preprocess import standardize

def reverse_line_movement_profit(x, y, include_ml=True, include_spread=True, include_ou=True):
    ml_profit, spread_profit, ou_profit = 0, 0, 0
    ml_record, spread_record, ou_record = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    for i in range(0, len(x)):
        if include_ml:
            if x[i][6] == x[i][7]:
                ml_record[3] += 1
            else:
                if x[i][6] == 0 and x[i][7] == 0:
                    ml_record[3] += 1
                elif x[i][6] < x[i][7]: # bet on underdog only if they were favored before
                    if x[i][6] < 0: # underdog bet
                        if x[i][2] < 0: # visitor is underdog
                            if y[i][0] > y[i][1]: # ML visitor win
                                ml_profit += 1 * -x[i][2] / 100
                                ml_record[0] += 1
                            elif y[i][0] < y[i][1]: # ML loss
                                ml_profit -= 1
                                ml_record[1] += 1
                            else: # push
                                ml_record[2] += 1
                        else: # home is underdog
                            if y[i][1] > y[i][0]: # ML home win
                                ml_profit += 1 * -x[i][5] / 100
                                ml_record[0] += 1
                            elif y[i][1] < y[i][0]: # ML loss
                                ml_profit -= 1
                                ml_record[1] += 1
                            else: # push
                                ml_record[2] += 1
                    else: # favorite bet
                        if x[i][2] < 0: # home is favorite
                            if y[i][1] > y[i][0]: # ML visitor win
                                ml_profit += 1 * x[i][5] / 100
                                ml_record[0] += 1
                            elif y[i][1] < y[i][0]: # ML loss
                                ml_profit -= 1
                                ml_record[1] += 1
                            else: # push
                                ml_record[2] += 1
                        else: # visitor is favorite
                            if y[i][0] > y[i][1]: # ML home win
                                ml_profit += 1 * x[i][2] / 100
                                ml_record[0] += 1
                            elif y[i][0] < y[i][1]: # ML loss
                                ml_profit -= 1
                                ml_record[1] += 1
                            else: # push
                                ml_record[2] += 1
                else: # bet on underdog
                    print("todo: rest of method")
            
        
    print("ml profit: " + str(ml_profit))
    print("ml record: " + str(ml_record))
    print("spread profit: " + str(spread_profit))
    print("spread record: " + str(spread_record))
    print("ou profit: " + str(ou_profit))
    print("ou record: " + str(ou_record))

    return (ml_profit, spread_profit, ou_profit)

def caluclate_profit(x, y_pred, y_real, include_ml=True, include_spread=True, include_ou=True):
    ml_profit, spread_profit, ou_profit = 0, 0, 0
    ml_record, spread_record, ou_record = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    #underdogs, spread_dogs = 0, 0

    for i in range(0, len(x)):
        #print(x[i])
        #print("pred: (" + str(int(y_pred[i][0])) + ", " + str(int(y_pred[i][1])) + ")")
        #print("real: (" + str(y_real[i][0]) + ", " + str(y_real[i][1]) + ")")
        # check ML
        y_pred[i][0] = int(y_pred[i][0])
        y_pred[i][1] = int(y_pred[i][1])
        if include_ml:
            if y_pred[i][0] > y_pred[i][1] and y_real[i][0] > y_real[i][1]: # ML visitor win
                if x[i][2] < 0: # underdog win!
                    #underdogs += 1
                    ml_profit += 1 * -x[i][2] / 100
                else: # favorite win
                    ml_profit += 1 * 100 / x[i][2]
                ml_record[0] += 1
            elif y_pred[i][0] < y_pred[i][1] and y_real[i][0] < y_real[i][1]: # ML home win
                if x[i][5] < 0: # underdog win!
                    ml_profit += 1 * -x[i][5] / 100
                    #underdogs += 1
                else: # favorite win
                    ml_profit += 1 * 100 / x[i][5]
                ml_record[0] += 1
            elif y_pred[i][0] == y_pred[i][1]:
                ml_record[3] += 1
            elif y_real[i][0] != y_real[i][1]: # ML loss
                #f y_pred[i][0] > y_pred[i][1] and x[i][2] < 0:
                    #underdogs += 1
                    #print('underdog away L')
                #elif y_pred[i][0] < y_pred[i][1] and x[i][5] < 0:
                    #underdogs += 1
                    #print("underdog home L")
                ml_profit -= 1
                ml_record[1] += 1
            else: # push
                ml_record[2] += 1

        if include_spread:
            if x[i][2] < 0: # visitor is underdog
                pred_spread = y_pred[i][1] - y_pred[i][0]
                real_spread = y_real[i][1] - y_real[i][0]

                if pred_spread < x[i][7]: # we bet on visitor
                    #spread_dogs += 1   
                    if real_spread < x[i][7]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread > x[i][7]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                elif pred_spread > x[i][7]: # we bet on home
                    if real_spread > x[i][7]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread < x[i][7]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                else:
                    spread_record[3] += 1
            elif x[i][2] > 0: # visitor is favorite
                pred_spread = y_pred[i][0] - y_pred[i][1]
                real_spread = y_real[i][0] - y_real[i][1]

                if pred_spread > x[i][7]: # we bet on visitor    
                    if real_spread > x[i][7]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread < x[i][7]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                elif pred_spread < x[i][7]: # we bet on home
                    #spread_dogs += 1
                    if real_spread < x[i][7]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread > x[i][7]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                else:
                    spread_record[3] += 1
        
        if include_ou:
            if np.sum(y_pred[i]) > x[i][9] and np.sum(y_real[i]) > x[i][9]:
                ou_profit += 100 / 110
                ou_record[0] += 1
            elif np.sum(y_pred[i]) < x[i][9] and np.sum(y_real[i]) < x[i][9]:
                ou_profit += 100 / 110
                ou_record[0] += 1
            elif np.sum(y_pred[i]) > x[i][9] and np.sum(y_real[i]) < x[i][9]:
                ou_profit -= 1
                ou_record[1] += 1
            elif np.sum(y_pred[i]) < x[i][9] and np.sum(y_real[i]) > x[i][9]:
                ou_profit -= 1
                ou_record[1] += 1
            elif np.sum(y_pred[i]) == x[i][9]:
                ou_record[3] += 1
            elif np.sum(y_real[i]) == x[i][9]:
                ou_record[2] += 1
            
    print("ml profit: " + str(ml_profit))
    print("ml record: " + str(ml_record))
    print("spread profit: " + str(spread_profit))
    print("spread record: " + str(spread_record))
    print("ou profit: " + str(ou_profit))
    print("ou record: " + str(ou_record))

    return (ml_profit, spread_profit, ou_profit)

def caluclate_profit_better(x, y_pred, y_real, include_ml=True, include_spread=True, include_ou=True):
    ml_profit, spread_profit, ou_profit = 0, 0, 0
    ml_record, spread_record, ou_record = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    #underdogs, spread_dogs = 0, 0

    for i in range(0, len(x)):
        #print(x[i])
        #print("pred: (" + str(int(y_pred[i][0])) + ", " + str(int(y_pred[i][1])) + ")")
        #print("real: (" + str(y_real[i][0]) + ", " + str(y_real[i][1]) + ")")
        # check ML
        y_pred[i][0][0] = int(y_pred[i][0][0])
        y_pred[i][0][1] = int(y_pred[i][0][1])
        if include_ml:
            if y_pred[i][0][0] > y_pred[i][0][1] and y_real[i][0][0] > y_real[i][0][1]: # ML visitor win
                if x[i][0][2] < 0: # underdog win!
                    #underdogs += 1
                    ml_profit += 1 * -x[i][0][2] / 100
                else: # favorite win
                    ml_profit += 1 * 100 / x[i][0][2]
                ml_record[0] += 1
            elif y_pred[i][0][0] < y_pred[i][0][1] and y_real[i][0][0] < y_real[i][0][1]: # ML home win
                if x[i][0][5] < 0: # underdog win!
                    ml_profit += 1 * -x[i][0][5] / 100
                    #underdogs += 1
                else: # favorite win
                    ml_profit += 1 * 100 / x[i][0][5]
                ml_record[0] += 1
            elif y_pred[i][0][0] == y_pred[i][0][1]:
                ml_record[3] += 1
            elif y_real[i][0][0] != y_real[i][0][1]: # ML loss
                #f y_pred[i][0][0] > y_pred[i][0][1] and x[i][0][2] < 0:
                    #underdogs += 1
                    #print('underdog away L')
                #elif y_pred[i][0][0] < y_pred[i][0][1] and x[i][0][5] < 0:
                    #underdogs += 1
                    #print("underdog home L")
                ml_profit -= 1
                ml_record[1] += 1
            else: # push
                ml_record[2] += 1

        if include_spread:
            if x[i][0][2] < 0: # visitor is underdog
                pred_spread = y_pred[i][0][1] - y_pred[i][0][0]
                real_spread = y_real[i][0][1] - y_real[i][0][0]

                if pred_spread < x[i][0][7]: # we bet on visitor
                    #spread_dogs += 1   
                    if real_spread < x[i][0][7]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread > x[i][0][7]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                elif pred_spread > x[i][0][7]: # we bet on home
                    if real_spread > x[i][0][7]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread < x[i][0][7]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                else:
                    spread_record[3] += 1
            elif x[i][0][2] > 0: # visitor is favorite
                pred_spread = y_pred[i][0][0] - y_pred[i][0][1]
                real_spread = y_real[i][0][0] - y_real[i][0][1]

                if pred_spread > x[i][0][7]: # we bet on visitor    
                    if real_spread > x[i][0][7]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread < x[i][0][7]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                elif pred_spread < x[i][0][7]: # we bet on home
                    #spread_dogs += 1
                    if real_spread < x[i][0][7]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread > x[i][0][7]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                else:
                    spread_record[3] += 1
        
        if include_ou:
            if np.sum(y_pred[i]) > x[i][0][9] and np.sum(y_real[i]) > x[i][0][9]:
                ou_profit += 100 / 110
                ou_record[0] += 1
            elif np.sum(y_pred[i]) < x[i][0][9] and np.sum(y_real[i]) < x[i][0][9]:
                ou_profit += 100 / 110
                ou_record[0] += 1
            elif np.sum(y_pred[i]) > x[i][0][9] and np.sum(y_real[i]) < x[i][0][9]:
                ou_profit -= 1
                ou_record[1] += 1
            elif np.sum(y_pred[i]) < x[i][0][9] and np.sum(y_real[i]) > x[i][0][9]:
                ou_profit -= 1
                ou_record[1] += 1
            elif np.sum(y_pred[i]) == x[i][0][9]:
                ou_record[3] += 1
            elif np.sum(y_real[i]) == x[i][0][9]:
                ou_record[2] += 1
            
    print("ml profit: " + str(ml_profit))
    print("ml record: " + str(ml_record))
    print("spread profit: " + str(spread_profit))
    print("spread record: " + str(spread_record))
    print("ou profit: " + str(ou_profit))
    print("ou record: " + str(ou_record))

    return (ml_profit, spread_profit, ou_profit)

def build_dataset():
    M = np.genfromtxt('./data/nfl odds 2007-2022.csv', missing_values=0, skip_header=1, delimiter=',', dtype=object)
    Mnew = []
    for i in range(0, len(M) - 1, 2):
        row = []
        team0 = M[i]
        team1 = M[i + 1]

        for col in team0[:M.shape[1] - 1]:
            row.append(col)
        for col in team1[1:M.shape[1] - 1]:
            row.append(col)

        # format: team0_home_advantage (-1 = visitor, 1 = home, 0 = neutral), team0_name, team0_ML (flip signs), team1_home_advantage, team1_name, team1_ML, spread_open, spread_close, tot_open, tot_close
        if row[1] == b'V':
            row[1] = -1
            row[6] = 1
        else:
            row[1] = 0
            row[6] = 0
        
        row[3] = -1 * int(row[3])
        row[8] = -1 * int(row[8])

        tmp0, tmp1, tmp2 = float(row[4]), float(row[5]), row[6]
        row[4], row[5], row[6] = tmp2, row[7], row[8]
        row[7], row[8] = tmp0, tmp1

        row[9], row[10] = float(row[9]), float(row[10])

        if row[7] > row[9]:
            row[7], row[9] = row[9], row[7]
        if row[8] > row[10]:
            row[8], row[10] = row[10], row[8]
        
        row.append(int(team0[M.shape[1] - 1]))
        row.append(int(team1[M.shape[1] - 1]))
        Mnew.append(row)
    Mnew = np.array(Mnew, dtype='object')

def main():
    M = np.genfromtxt('./data/data', missing_values=0, skip_header=1, delimiter=' ', dtype=object)

    y = np.array([[row[len(M[0]) - 2], row[len(M[0]) - 1]] for row in M])
    x = np.array([row[1:len(M[0]) - 2] for row in M])

    # encode team names
    encoder = LabelEncoder()
    x[:, 1] = encoder.fit_transform(x[:, 1])
    x[:, 4] = encoder.transform(x[:, 4])

    # set dtype to float
    x = x.astype('float64')
    y = y.astype('float64')

    # train test split
    Xtrn, Xtst, Ytrn, Ytst = train_test_split(x, y)

    # get averages, standard devs
    xtst_mu, xtst_sigma = np.mean(Xtst, axis=0), np.std(Xtst, axis=0)
    ytst_mu, ytst_sigma = np.mean(Ytst, axis=0), np.std(Ytst, axis=0)

    # standardize data
    Xtrn, Xtst, Ytrn, Ytst = standardize(Xtrn), standardize(Xtst), standardize(Ytrn), standardize(Ytst)

    # linear regression
    print("linear regression")

    linear = LinearRegression().fit(Xtrn, Ytrn)
    lin_pred = linear.predict(Xtst) * ytst_sigma + ytst_mu

    caluclate_profit(Xtst * xtst_sigma + xtst_mu, lin_pred, Ytst * ytst_sigma + ytst_mu)
    
    # sample for tonight's game
    sample = [-1, encoder.transform([b"b'LasVegas'"])[0], 275, 1, encoder.transform([b"b'LARams'"])[0], -225, 4, 6.5, 43.0, 42.0]
    sample = (sample - xtst_mu) / xtst_sigma
    y_sample = linear.predict([sample])
    print(y_sample * ytst_sigma + ytst_mu)
    
    print("")
    
    # sklearn MLPRegressor
    print("MLPRegressor")

    nn = MLPRegressor(max_iter=400, learning_rate_init=0.005).fit(Xtrn, Ytrn)
    y_pred = nn.predict(Xtst)

    caluclate_profit(Xtst * xtst_sigma + xtst_mu, y_pred * ytst_sigma + ytst_mu, Ytst * ytst_sigma + ytst_mu)

    y_sample = nn.predict([sample])
    print(y_sample * ytst_sigma + ytst_mu)

    print("")
    
    # own implementation
    print("own implementation")

    y = np.array([[[row[len(M[0]) - 2], row[len(M[0]) - 1]]] for row in M])
    x = np.array([[row[1:len(M[0]) - 2]] for row in M])

    # encode team names
    encoder = LabelEncoder()
    x[:, 0, 1] = encoder.fit_transform(x[:, 0, 1])
    x[:, 0, 4] = encoder.transform(x[:, 0, 4])

    # set dtype to float
    x = x.astype('float64')
    y = y.astype('float64')

    # train test split
    Xtrn, Xtst, Ytrn, Ytst = train_test_split(x, y)

    # get averages, standard devs
    xtst_mu, xtst_sigma = np.mean(Xtst, axis=0), np.std(Xtst, axis=0)
    ytst_mu, ytst_sigma = np.mean(Ytst, axis=0), np.std(Ytst, axis=0)

    # standardize data
    Xtrn, Xtst, Ytrn, Ytst = standardize(Xtrn), standardize(Xtst), standardize(Ytrn), standardize(Ytst)

    sample = [sample]

    better_nn = Network(hidden_layer_configs=((100, ReLUActivation),), max_iter=400, learning_rate=0.0002, initialization=GaussianInitialization())
    better_nn.fit(Xtrn, Ytrn)
    y_pred = better_nn.predict(Xtst)

    caluclate_profit_better(Xtst * xtst_sigma + xtst_mu, y_pred * ytst_sigma + ytst_mu, Ytst * ytst_sigma + ytst_mu)

    y_sample = better_nn.predict([sample])
    print(y_sample * ytst_sigma + ytst_mu)
    

main()