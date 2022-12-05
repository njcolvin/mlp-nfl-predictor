import sklearn
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import numpy as np
from network import Network
from activation import ReLUActivation
from initialization import UniformInitialization
from sklearn.linear_model import LinearRegression

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
                    ml_profit += 1 * -x[i][2] / 100
                else: # favorite win
                    ml_profit += 1 * 100 / x[i][2]
                ml_record[0] += 1
            elif y_pred[i][0] < y_pred[i][1] and y_real[i][0] < y_real[i][1]: # ML home win
                if x[i][5] < 0: # underdog win!
                    ml_profit += 1 * -x[i][5] / 100
                else: # favorite win
                    ml_profit += 1 * 100 / x[i][5]
                ml_record[0] += 1
            elif y_pred[i][0] == y_pred[i][1]:
                ml_record[3] += 1
            elif y_real[i][0] != y_real[i][1]: # ML loss
                ml_profit -= 1
                ml_record[1] += 1
            else: # push
                ml_record[2] += 1

        if include_spread:
            if x[i][2] < 0: # visitor is underdog
                pred_spread = y_pred[i][1] - y_pred[i][0]
                real_spread = y_real[i][1] - y_real[i][0]

                if pred_spread < x[i][7]: # we bet on visitor    
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

def main():
    M = np.genfromtxt('./nfl odds 2007-2022.csv', missing_values=0, skip_header=1, delimiter=',', dtype=object)
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
                
        row.append([int(team0[M.shape[1] - 1]), int(team1[M.shape[1] - 1])])
        Mnew.append(row)


    y = np.array([row[len(Mnew[0]) - 1] for row in Mnew])
    x = np.array([row[1:len(Mnew[0]) - 1] for row in Mnew])

    encoder = LabelEncoder()

    x[:, 1] = encoder.fit_transform(x[:, 1])
    x[:, 4] = encoder.transform(x[:, 4])

    x = x.astype('float64')

    Xtrn, Xtst, Ytrn, Ytst = train_test_split(x, y)

    linear = LinearRegression().fit(Xtrn, Ytrn)
    lin_pred = linear.predict(Xtst)    

    lin_profit = caluclate_profit(Xtst, lin_pred, Ytst)

    print("")

    nn = MLPRegressor(max_iter=400, learning_rate_init=0.005).fit(Xtrn, Ytrn)
    y_pred = nn.predict(Xtst)

    nn_profit = caluclate_profit(Xtst, y_pred, Ytst)
    
    print("")

    lin_score = linear.score(Xtst, Ytst)
    nn_score = nn.score(Xtst, Ytst)
    print("linear score: " + str(lin_score))
    print("nn score: " + str(nn_score))
    if lin_score > nn_score:
        print("linear wins")
    elif lin_score < nn:
        print("nn wins")
    else:
        print("tie...")

    if lin_profit[0] > nn_profit[0]:
        print("linear higher ml")
    else:
        print("nn higher ml")
    if lin_profit[1] > nn_profit[1]:
        print("linear higher spread")
    else:
        print("nn higher spread")
    if lin_profit[2] > nn_profit[2]:
        print("linear higher ou")
    else:
        print("nn higher ou")

    """ better_nn = Network(hidden_layer_configs=((100, ReLUActivation),), max_iter=400, learning_rate=0.01,
                    initialization=UniformInitialization())
    better_nn.fit(Xtrn, Ytrn)
    y_pred = better_nn.predict(Xtst)
    profit = 0
    record = [0, 0]
    for i in range(0, len(Xtst)):
        print(Xtst[i])
        print("pred: (" + str(int(y_pred[i][0])) + ", " + str(int(y_pred[i][1])) + ")")
        print("real: (" + str(Ytst[i][0]) + ", " + str(Ytst[i][1]) + ")")
        # check ML
        if int(y_pred[i][0]) > int(y_pred[i][1]) and Ytst[i][0] > Ytst[i][1]: # ML visitor win
            if Xtst[i][2] < 0: # underdog win!
                profit += 1 * -Xtst[i][2] / 100
                print("underdog away win!")
            else: # favorite win
                profit += 1 * 100 / Xtst[i][2]
                print("favorite away win")
            record[0] += 1
        elif int(y_pred[i][0]) < int(y_pred[i][1]) and Ytst[i][0] < Ytst[i][1]: # ML home win
            if Xtst[i][5] < 0: # underdog win!
                profit += 1 * -Xtst[i][5] / 100
                print("underdog home win!")
            else: # favorite win
                profit += 1 * 100 / Xtst[i][5]
                print("favorite home win")
        else: # loss
            print("L")
            profit -= 1
            record[1] += 1
        print("total profit: " + str(profit))
        print ("record: " + str(record))
    print(nn.score(Xtst, Ytst)) """

main()