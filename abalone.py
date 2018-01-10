import os
import numpy
import pandas
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier, RandomForestClassifier
from math import exp


def week_5_task_1():
    data = pandas.read_csv(os.path.join('data', 'abalone.csv'))
    data['Sex'] = data['Sex'].map(lambda x: 1 if x == 'M' else (-1 if x == 'F' else 0))
    y, X = data['Rings'], data.drop('Rings', axis=1)
    k_fold = KFold(n_splits=5, shuffle=True, random_state=1)
    result = {}
    for trees in range(1,51):
        alg = RandomForestRegressor(n_estimators=trees, random_state=1)
        quality = cross_val_score(estimator=alg, X=X, y=y, cv=k_fold, scoring='r2')
        result[trees] = numpy.mean(quality)
    return [i for i, v in result.iteritems() if v > 0.52]


def sigmoid(predictions):
    return [1/(1 + exp(-i)) for i in predictions]


def _log_loss(X_train, X_test, y_train, y_test, rate):
    cls = GradientBoostingClassifier(n_estimators=250, verbose=False, random_state=241, learning_rate=rate)
    cls.fit(X_train, y_train)
    train_loss = [log_loss(y_train, sigmoid(i)) for i in cls.staged_decision_function(X_train)]
    test_loss = [log_loss(y_test, sigmoid(i)) for i in cls.staged_decision_function(X_test)]
    result = {'train': (train_loss.index(min(train_loss)), min(train_loss)),
              'test': (test_loss.index(min(test_loss)), min(test_loss))}
    return result


def week_5_task_2():
    data = pandas.read_csv(os.path.join('data', 'gbm-data.csv'))
    y, X = data['Activity'], data.drop('Activity', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=241)
    for rate in [1, 0.5, 0.3, 0.2, 0.1]:
        print 'Rate:', rate
        print _log_loss(X_train, X_test, y_train, y_test, rate)
    cls = RandomForestClassifier(n_estimators=36, random_state=241)
    cls.fit(X_train, y_train)
    print log_loss(y_test, cls.predict_proba(X_test))


if __name__ == '__main__':
    print week_5_task_1()[0]
    week_5_task_2()
