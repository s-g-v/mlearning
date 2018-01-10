import os
import pandas
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,\
    roc_auc_score, precision_recall_curve
classification = pandas.read_csv(os.path.join('data', 'classification.csv'))
scores = pandas.read_csv(os.path.join('data', 'scores.csv'))


def tp_fp_fn_tn():
    tp, fp, fn, tn = 0, 0, 0, 0
    for i, t, p in classification.itertuples():
        if t == 1 and p == 1:
            tp += 1
        elif t == 0 and p == 1:
            fp += 1
        elif t == 1 and p == 0:
            fn += 1
        else:
            tn += 1
    print tp, fp, fn, tn


def metrics():
    acc = accuracy_score(classification.true, classification.pred)
    prec = precision_score(classification.true, classification.pred)
    recall = recall_score(classification.true, classification.pred)
    f = f1_score(classification.true, classification.pred)
    print acc, prec, recall, f


def areas():
    logreg = roc_auc_score(scores.true, scores.score_logreg)
    svm = roc_auc_score(scores.true, scores.score_svm)
    knn = roc_auc_score(scores.true, scores.score_knn)
    tree = roc_auc_score(scores.true, scores.score_tree)
    print logreg, svm, knn, tree


def prc(predict):
    prec, rec, _ = precision_recall_curve(scores.true, predict)
    prec = [prec[i] for i, v in enumerate(rec) if v >= 0.7]
    return np.max(prec)


if __name__ == '__main__':
    tp_fp_fn_tn()
    metrics()
    areas()
    print prc(scores.score_logreg), prc(scores.score_svm), prc(scores.score_knn), prc(scores.score_tree)
