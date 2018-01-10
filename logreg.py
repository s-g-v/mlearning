import os
import pandas
import numpy as np
from sklearn.metrics import roc_auc_score
from math import exp


def distance(w_new, w):
    return np.sqrt(np.square(w_new[0] - w[0]) + np.square(w_new[1] - w[1]))


def predictions(w, x):
    return [1/(1 + exp(-w[0] * x[i][0] - w[1] * x[i][1])) for i in range(len(x))]


def logisctic_regression(y, x, c):
    w1, w2 = 0, 0
    expected_distance = 1e-5
    k = 0.1
    max_iter = 10000
    for i in range(max_iter):
        w1new = w1 + k * np.mean(y * x[:, 0] * (1 - 1./(1 + np.exp(-y * (w1 * x[:, 0] + w2 * x[:, 1]))))) - k * c * w1
        w2new = w2 + k * np.mean(y * x[:, 1] * (1 - 1./(1 + np.exp(-y * (w1 * x[:, 0] + w2 * x[:, 1]))))) - k * c * w2
        if distance((w1new, w2new), (w1, w2)) < expected_distance:
            break
        w1, w2 = w1new, w2new
    return predictions((w1, w2), x)


def week_3_task_3():
    data = pandas.read_csv(os.path.join('data', 'data-logistic.csv'), header=None)
    target = data.values[:, :1].T[0]
    attr = data.values[:, 1:]
    p0 = logisctic_regression(target, attr, 0)
    p1 = logisctic_regression(target, attr, 10)
    roc_auc_0 = roc_auc_score(target, p0)
    roc_auc_1 = roc_auc_score(target, p1)
    print '%2.3f' % roc_auc_0, '%2.3f' % roc_auc_1


if __name__ == '__main__':
    week_3_task_3()
