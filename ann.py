from sklearn.neural_network import MLPClassifier
from sklearn import preprocessing
import numpy as np

def main():
    M = np.genfromtxt('./nfl odds 2007-08.csv', missing_values=0, skip_header=1, delimiter=',', dtype=object)
    split = (int) (0.7 * len(M))
    if split % 2 == 1:
        split += 1
    Mtrn = M[:split, :]
    Mtst = M[split:, :]
    Ytrn = Mtrn[:, 10]
    Xtrn = Mtrn[:, :5]
    Ytst = Mtst[:, 10]
    Xtst = Mtst[:, :5]
    Xtrn_new = []
    Ytrn_new = []
    Xtst_new = []
    Ytst_new = []
    teams = np.unique(Xtrn[:, 1])

    for i in range(0, len(Xtrn) - 1, 2):
        row = []
        for col in Xtrn[i]:
            row.append(col)
        for col in Xtrn[i + 1]:
            row.append(col)

        Ytrn_new.append(np.array([int(Ytrn[i]), int(Ytrn[i + 1])]))
        Ytrn_new.append(int(Ytrn[i + 1]))

        # format: team, opponent, is_home, spread_open, spread_close, tot_open, tot_close, ml, ml_opponent
        new_row_v = [np.where(teams == row[1])[0][0], np.where(teams == row[6])[0][0], 0]
        new_row_h = [np.where(teams == row[6])[0][0], np.where(teams == row[1])[0][0], 1]
        
        if float(row[4]) < 0: # visitor is favorite -> spread is in first half of row before their ml
            new_row_v.append(float(row[2]) * -1) # spread_open
            new_row_v.append(float(row[3]) * -1) # spread_close
            new_row_v.append(float(row[7])) # tot_open
            new_row_v.append(float(row[8])) # tot_close
            new_row_v.append(float(row[4])) # ml
            new_row_v.append(float(row[9])) # ml_opponent
            new_row_h.append(float(row[2]))
            new_row_h.append(float(row[3]))
            new_row_h.append(float(row[7]))
            new_row_h.append(float(row[8]))
            new_row_h.append(float(row[9]))
            new_row_h.append(float(row[4]))
        else: # home is favorite -> spread is in second half of row before their ml
            new_row_v.append(float(row[7])) # spread_open
            new_row_v.append(float(row[8])) # spread_close
            new_row_v.append(float(row[2])) # tot_open
            new_row_v.append(float(row[3])) # tot_close
            new_row_v.append(float(row[4])) # ml
            new_row_v.append(float(row[9])) # ml_opponent
            new_row_h.append(float(row[7]) * -1)
            new_row_h.append(float(row[8]) * -1)
            new_row_h.append(float(row[2]))
            new_row_h.append(float(row[3]))
            new_row_h.append(float(row[9]))
            new_row_h.append(float(row[4]))

        Xtrn_new.append(new_row_v)
        Xtrn_new.append(new_row_h)

    for i in range(0, len(Xtst) - 1, 2):
        row = []
        for col in Xtst[i]:
            row.append(col)
        for col in Xtst[i + 1]:
            row.append(col)

        Ytst_new.append(int(Ytst[i]))
        Ytst_new.append(int(Ytst[i + 1]))

        # format: team, opponent, is_home, spread_open, spread_close, tot_open, tot_close, ml, ml_opponent
        new_row_v = [np.where(teams == row[1])[0][0], np.where(teams == row[6])[0][0], 0]
        new_row_h = [np.where(teams == row[6])[0][0], np.where(teams == row[1])[0][0], 1]
        
        if float(row[4]) < 0: # visitor is favorite -> spread is in first half of row before their ml
            new_row_v.append(float(row[2]) * -1) # spread_open
            new_row_v.append(float(row[3]) * -1) # spread_close
            new_row_v.append(float(row[7])) # tot_open
            new_row_v.append(float(row[8])) # tot_close
            new_row_v.append(float(row[4])) # ml
            new_row_v.append(float(row[9])) # ml_opponent
            new_row_h.append(float(row[2]))
            new_row_h.append(float(row[3]))
            new_row_h.append(float(row[7]))
            new_row_h.append(float(row[8]))
            new_row_h.append(float(row[9]))
            new_row_h.append(float(row[4]))
        else: # home is favorite -> spread is in second half of row before their ml
            new_row_v.append(float(row[7])) # spread_open
            new_row_v.append(float(row[8])) # spread_close
            new_row_v.append(float(row[2])) # tot_open
            new_row_v.append(float(row[3])) # tot_close
            new_row_v.append(float(row[4])) # ml
            new_row_v.append(float(row[9])) # ml_opponent
            new_row_h.append(float(row[7]) * -1)
            new_row_h.append(float(row[8]) * -1)
            new_row_h.append(float(row[2]))
            new_row_h.append(float(row[3]))
            new_row_h.append(float(row[9]))
            new_row_h.append(float(row[4]))

        Xtst_new.append(new_row_v)
        Xtst_new.append(new_row_h)

    nn = MLPClassifier(max_iter=400).fit(Xtrn_new, Ytrn_new)
    y_pred = nn.predict(Xtrn_new)
    for i in range(0, len(Xtst_new) - 1, 2):
        print(Xtst_new[i])
        print(Xtst_new[i + 1])
        print("v pred: " + str(y_pred[i]))
        print("v real: " + str(Ytst_new[i]))
        print("h pred: " + str(y_pred[i + 1]))
        print("h real: " + str(Ytst_new[i + 1]))
        
    print(nn.score(Xtst_new, Ytst_new))

main()