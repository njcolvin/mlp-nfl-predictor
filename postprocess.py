import numpy as np

def caluclate_profit_sklearn(x, y_pred, y_real, include_winloss=False, include_ml=True, include_spread=True, include_ou=True):
    ml_profit, spread_profit, ou_profit = 0, 0, 0
    ml_record, spread_record, ou_record = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    vml_i, hml_i, s_i, ou_i = 2, 5, 7, 9

    if include_winloss:
        vml_i, hml_i, s_i, ou_i = 5, 11, 13, 15

    for i in range(0, len(x)):
        # check ML
        y_pred[i][0] = int(y_pred[i][0])
        y_pred[i][1] = int(y_pred[i][1])
        if include_ml:
            if y_pred[i][0] > y_pred[i][1] and y_real[i][0] > y_real[i][1]: # ML visitor win
                if x[i][vml_i] < 0: # underdog win!
                    ml_profit += 1 * -x[i][vml_i] / 100
                else: # favorite win
                    ml_profit += 1 * 100 / x[i][vml_i]
                ml_record[0] += 1

            elif y_pred[i][0] < y_pred[i][1] and y_real[i][0] < y_real[i][1]: # ML home win
                if x[i][hml_i] < 0: # underdog win!
                    ml_profit += 1 * -x[i][hml_i] / 100
                else: # favorite win
                    ml_profit += 1 * 100 / x[i][hml_i]
                ml_record[0] += 1
            elif y_pred[i][0] == y_pred[i][1]:
                ml_record[3] += 1
            elif y_real[i][0] != y_real[i][1]: # ML loss
                ml_profit -= 1
                ml_record[1] += 1
            else: # push
                ml_record[2] += 1

        if include_spread:
            if x[i][vml_i] < 0: # visitor is underdog
                pred_spread = y_pred[i][1] - y_pred[i][0]
                real_spread = y_real[i][1] - y_real[i][0]

                if pred_spread < x[i][s_i]: # we bet on visitor
                    if real_spread < x[i][s_i]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread > x[i][s_i]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else: # push
                        spread_record[2] += 1
                elif pred_spread > x[i][s_i]: # we bet on home
                    if real_spread > x[i][s_i]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread < x[i][s_i]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                else: # no bet
                    spread_record[3] += 1
            elif x[i][vml_i] > 0: # visitor is favorite
                pred_spread = y_pred[i][0] - y_pred[i][1]
                real_spread = y_real[i][0] - y_real[i][1]

                if pred_spread > x[i][s_i]: # we bet on visitor    
                    if real_spread > x[i][s_i]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread < x[i][s_i]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else: # push
                        spread_record[2] += 1
                elif pred_spread < x[i][s_i]: # we bet on home
                    if real_spread < x[i][s_i]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread > x[i][s_i]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else: # push
                        spread_record[2] += 1
                else: # no bet
                    spread_record[3] += 1
        
        if include_ou:
            if np.sum(y_pred[i]) > x[i][ou_i] and np.sum(y_real[i]) > x[i][ou_i]: # over win
                ou_profit += 100 / 110
                ou_record[0] += 1
            elif np.sum(y_pred[i]) < x[i][ou_i] and np.sum(y_real[i]) < x[i][ou_i]: # under win
                ou_profit += 100 / 110
                ou_record[0] += 1
            elif np.sum(y_pred[i]) > x[i][ou_i] and np.sum(y_real[i]) < x[i][ou_i]: # over pred, under real
                ou_profit -= 1
                ou_record[1] += 1
            elif np.sum(y_pred[i]) < x[i][ou_i] and np.sum(y_real[i]) > x[i][ou_i]: # under pred, over real
                ou_profit -= 1
                ou_record[1] += 1
            elif np.sum(y_pred[i]) == x[i][ou_i]: # no bet
                ou_record[3] += 1
            elif np.sum(y_real[i]) == x[i][ou_i]: # push
                ou_record[2] += 1
            
    print("ml profit: " + str(ml_profit))
    print("ml record: " + str(ml_record))
    print("spread profit: " + str(spread_profit))
    print("spread record: " + str(spread_record))
    print("ou profit: " + str(ou_profit))
    print("ou record: " + str(ou_record))

    return (ml_profit, spread_profit, ou_profit)

def caluclate_profit_network(x, y_pred, y_real, include_winloss=True, include_ml=True, include_spread=True, include_ou=True):
    ml_profit, spread_profit, ou_profit = 0, 0, 0
    ml_record, spread_record, ou_record = [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]
    vml_i, hml_i, s_i, ou_i = 2, 5, 7, 9

    if include_winloss:
        vml_i, hml_i, s_i, ou_i = 5, 11, 13, 15

    for i in range(0, len(x)):
        # check ML
        y_pred[i][0][0] = int(y_pred[i][0][0])
        y_pred[i][0][1] = int(y_pred[i][0][1])
        if include_ml:
            if y_pred[i][0][0] > y_pred[i][0][1] and y_real[i][0][0] > y_real[i][0][1]: # ML visitor win
                if x[i][0][vml_i] < 0: # underdog win!
                    ml_profit += -x[i][0][vml_i] / 100
                elif x[i][0][vml_i] > 0: # favorite win
                    ml_profit += 100 / x[i][0][vml_i]
                else:
                    ml_profit += 100 / 110

                ml_record[0] += 1
            elif y_pred[i][0][0] < y_pred[i][0][1] and y_real[i][0][0] < y_real[i][0][1]: # ML home win
                if x[i][0][hml_i] < 0: # underdog win!
                    ml_profit += (-x[i][0][hml_i] / 100)
                elif x[i][0][hml_i] > 0: # favorite win
                    ml_profit += 100 / x[i][0][hml_i]
                else: 
                    ml_profit += 100 / 110

                ml_record[0] += 1
            elif y_pred[i][0][0] == y_pred[i][0][1]:
                ml_record[3] += 1
            elif y_real[i][0][0] != y_real[i][0][1]: # ML loss
                ml_profit -= 1
                ml_record[1] += 1
            else: # push
                ml_record[2] += 1

        if include_spread:
            if x[i][0][vml_i] < 0: # visitor is underdog
                pred_spread = y_pred[i][0][1] - y_pred[i][0][0]
                real_spread = y_real[i][0][1] - y_real[i][0][0]

                if pred_spread < x[i][0][s_i]: # we bet on visitor
                    #spread_dogs += 1   
                    if real_spread < x[i][0][s_i]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread > x[i][0][s_i]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                elif pred_spread > x[i][0][s_i]: # we bet on home
                    if real_spread > x[i][0][s_i]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread < x[i][0][s_i]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                else:
                    spread_record[3] += 1
            elif x[i][0][vml_i] > 0: # visitor is favorite
                pred_spread = y_pred[i][0][0] - y_pred[i][0][1]
                real_spread = y_real[i][0][0] - y_real[i][0][1]

                if pred_spread > x[i][0][s_i]: # we bet on visitor    
                    if real_spread > x[i][0][s_i]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread < x[i][0][s_i]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                elif pred_spread < x[i][0][s_i]: # we bet on home
                    #spread_dogs += 1
                    if real_spread < x[i][0][s_i]: # win
                        spread_profit += 100 / 110
                        spread_record[0] += 1
                    elif real_spread > x[i][0][s_i]: # loss
                        spread_profit -= 1
                        spread_record[1] += 1
                    else:
                        spread_record[2] += 1
                else:
                    spread_record[3] += 1
        
        if include_ou:
            if np.sum(y_pred[i]) > x[i][0][ou_i] and np.sum(y_real[i]) > x[i][0][ou_i]:
                ou_profit += 100 / 110
                ou_record[0] += 1
            elif np.sum(y_pred[i]) < x[i][0][ou_i] and np.sum(y_real[i]) < x[i][0][ou_i]:
                ou_profit += 100 / 110
                ou_record[0] += 1
            elif np.sum(y_pred[i]) > x[i][0][ou_i] and np.sum(y_real[i]) < x[i][0][ou_i]:
                ou_profit -= 1
                ou_record[1] += 1
            elif np.sum(y_pred[i]) < x[i][0][ou_i] and np.sum(y_real[i]) > x[i][0][ou_i]:
                ou_profit -= 1
                ou_record[1] += 1
            elif np.sum(y_pred[i]) == x[i][0][ou_i]:
                ou_record[3] += 1
            elif np.sum(y_real[i]) == x[i][0][ou_i]:
                ou_record[2] += 1
            
    print("ml profit: " + str(ml_profit))
    print("ml record: " + str(ml_record))
    print("spread profit: " + str(spread_profit))
    print("spread record: " + str(spread_record))
    print("ou profit: " + str(ou_profit))
    print("ou record: " + str(ou_record))

    return (ml_profit, spread_profit, ou_profit)
