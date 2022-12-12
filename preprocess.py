import numpy as np

def standardize(X):
    """
    Args:
        'X': numpy ndarray 
    Returns:
        'X_norm': normalized X also in numpy ndarray format
    """
    X_norm = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
    return X_norm

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

    # split into x and y like this:
    # y = [[row[len(Mnew[0]) - 2], row[len(Mnew[0]) - 1]] for row in Mnew]
    # x = [row[1:len(Mnew[0]) - 2] for row in Mnew]
    np.savetxt("data", Mnew, fmt='%s')

def add_win_loss(x, y):
    i = 0
    while i < len(x):
        
        records = dict()
        while i < len(x):
            team0 = x[i][2].decode('utf-8')
            team1 = x[i][5].decode('utf-8')

            team0_win, tie = False, False
            if not team0 in records:
                records[team0] = [0, 0, 0]
            if not team1 in records:
                records[team1] = [0, 0, 0]
            if int(y[i][0]) > int(y[i][1]):
                team0_win = True
            elif int(y[i][1]) == int(y[i][0]):
                tie = True

            team0_idx = 3
            team1_idx = 9
            x[i] = list(x[i])
            for j in range(3):
                x[i].insert(team0_idx, records[team0][j])
                team0_idx += 1
            for j in range(3):
                x[i].insert(team1_idx, records[team1][j])
                team1_idx += 1

            if team0_win:
                records[team0][0] += 1
                records[team1][1] += 1
            elif not tie:
                records[team0][1] += 1
                records[team1][0] += 1
            else:
                records[team0][2] += 1
                records[team1][2] += 1

            i += 1
            if i < len(x) and x[i][0].decode('utf-8').startswith('2'):
                break

        if i >= len(x):
            break
        
        # super bowl
        team0 = x[i][2].decode('utf-8')
        team1 = x[i][5].decode('utf-8')
        
        team0_win, tie = False, False
        if not team0 in records:
            records[team0] = [0, 0, 0]
        if not team1 in records:
            records[team1] = [0, 0, 0]
        if int(y[i][0]) > int(y[i][1]):
            team0_win = True
        elif int(y[i][1]) == int(y[i][0]):
            tie = True

        team0_idx = 3
        team1_idx = 9
        x[i] = list(x[i])
        for j in range(3):
            x[i].insert(team0_idx, records[team0][j])
            team0_idx += 1
        for j in range(3):
            x[i].insert(team1_idx, records[team1][j])
            team1_idx += 1

        if team0_win:
            records[team0][0] += 1
            records[team1][1] += 1
        elif not tie:
            records[team0][1] += 1
            records[team1][0] += 1
        else:
            records[team0][2] += 1
            records[team1][2] += 1

        i += 1
    
    x = np.array(x)
    np.savetxt("data/data-wl", x, fmt='%s')
