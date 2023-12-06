import numpy as np
def a_g(matrix, time):
    cover = np.zeros(len(matrix[0]))
    d = dict(zip(range(len(matrix)), np.zeros(len(matrix))))
    rest = len(matrix)
    val = np.zeros(len(matrix))
    s = []
    while rest > 0:
        for i in d:
            temp_num = 0
            for j in range(len(matrix[0])):
                if cover[j] == 0 and matrix[i][j] == 1:
                    temp_num += 1
            d[i] = temp_num / time[i]
        pick = max(d, key=d.get)
        val[pick] = d[pick]
        s.append(pick)
        for j in range(len(matrix[0])):
            if matrix[pick][j] == 1 and cover[j] == 0:
                cover[j] = 1
        del d[pick]
        rest -= 1
    val = val / sum(val)
    return s, val

def get_apfdc(seq, matrix, time):
    tf = np.zeros(len(matrix[0]))
    cover = np.zeros(len(matrix[0]))
    rank = 0
    for pick in seq:
        for j in range(len(matrix[0])):
            if matrix[pick][j] == 1 and cover[j] == 0:
                cover[j] = 1
                tf[j] = rank
        rank += 1
    t1 = 0
    for i in range(len(matrix[0])):
        t2 = 0
        for j in range(int(tf[i]), len(matrix)):
            t2 += time[seq[j]]
        t1 += t2 - 0.5 * time[int(tf[i])]
    apfdc = t1 / (sum(time) * len(matrix[0]))
    return apfdc

def a_githouttime(matrix):
    cover = np.zeros(len(matrix[0]))
    d = dict(zip(range(len(matrix)), np.zeros(len(matrix))))
    rest = len(matrix)
    val = np.zeros(len(matrix))
    s = []
    while rest > 0:
        for i in d:
            temp_num = 0
            for j in range(len(matrix[0])):
                if cover[j] == 0 and matrix[i][j] == 1:
                    temp_num += 1
            d[i] = temp_num
        pick = max(d, key=d.get)
        val[pick] = d[pick]
        s.append(pick)
        for j in range(len(matrix[0])):
            if matrix[pick][j] == 1 and cover[j] == 0:
                cover[j] = 1
        del d[pick]
        rest -= 1
    val = val / sum(val)
    return s, val
