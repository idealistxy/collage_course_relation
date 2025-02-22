'''
Description: 
Author: 张轩誉
Date: 2024-05-17 10:12:10
LastEditors: 张轩誉
LastEditTime: 2024-05-17 13:45:10
'''
import numpy as np
from numpy import linalg as la


def loadExData2():
    return [
        [2, 0, 0, 4, 4, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 4, 0],
        [3, 3, 4, 0, 3, 0, 0, 2, 2, 0, 0],
        [5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
        [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 5],
        [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4],
        [0, 0, 0, 0, 0, 0, 5, 0, 0, 5, 0],
        [0, 0, 0, 3, 0, 0, 0, 0, 4, 5, 0],
        [1, 1, 2, 1, 1, 2, 1, 0, 4, 5, 0]]


def ecludSim(inA, inB):
    return 1.0 / (1.0 + la.norm(inA - inB))


def pearsSim(inA, inB):
    if len(inA) < 3:
        return 1.0
    return 0.5 + 0.5 * np.corrcoef(inA, inB, rowvar=0)[0][1]


def cosSim(inA, inB):
    num = float(inA.T * inB)
    denom = la.norm(inA) * la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0:
            continue
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]

        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])

        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    if len(unratedItems) == 0:
        return 'you rated everything'
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        print(f'estimate:{estimatedScore}')
        itemScores.append((item, estimatedScore))  
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[:N]


def suitable_λ():
    myMat = np.mat(loadExData2())
    U, sigma, VT = np.linalg.svd(myMat)
    sigma = sigma**2
    cnt = sum(sigma)
    print(cnt)
    value = cnt*0.9
    print(f'90%value:{value}')
    for i in range(sigma.size):
        cnt2 = sum(sigma[:i])
        print(f'前{i}个奇异值能量和:{cnt2}')
        if cnt2 >= value:
            return


def svdEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    U, Sigma, VT = la.svd(dataMat)
    Sig4 = np.mat(np.eye(5) * Sigma[:5])
    xformedItems = dataMat.T * U[:, :5] * Sig4.I
    for j in range(n):
        userRating = dataMat[user, j]
        if userRating == 0 or j == item:
            continue
        similarity = simMeas(xformedItems[item, :].T, \
                             xformedItems[j, :].T)
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    else:
        return ratSimTotal / simTotal


myMat = np.mat(loadExData2())
suitable_λ()
print(recommend(myMat, 3, 3, estMethod=svdEst, simMeas=pearsSim))
