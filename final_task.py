#!/usr/bin/env python
# coding=utf-8
import pandas
import numpy as np
from collections import OrderedDict
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

data = pandas.read_csv('features.csv', index_col='match_id')
y, X = data['radiant_win'], data.drop(['duration', 'radiant_win', 'tower_status_radiant',
                                       'tower_status_dire', 'barracks_status_radiant',
                                       'barracks_status_dire'], axis=1)
X_test = pandas.read_csv('features_test.csv', index_col='match_id').fillna(0)


def fill_missed_values(x):
    print 'Attributes with missed values:'
    print '\n'.join(map(str, [(i, v) for (i, v) in x.count().iteritems() if v != x.shape[0]]))
    return x.fillna(0)


def timer(name, func):
    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = func(*args, **kwargs)
        print 'Elapsed time for {}: {}'.format(name, datetime.now() - start_time)
        return result
    return wrapper


def gradient_boosting_with_timer(attrs):
    result = OrderedDict()
    k_fold = KFold(n_splits=5, shuffle=True, random_state=241)
    for trees in [10, 20, 30, 40, 50, 60]:
        alg = GradientBoostingClassifier(n_estimators=trees, random_state=241)
        timed_cv_score = timer('{} trees'.format(trees), cross_val_score)
        quality = timed_cv_score(estimator=alg, X=attrs, y=y, cv=k_fold, scoring='roc_auc')
        print 'AUC_ROC scores for {} trees {}, mean {}'.format(trees, quality, np.mean(quality))
        result[trees] = np.mean(quality)
    return max(result.items(), key=lambda x: x[1])


def logistic_regression_with_timer(attrs):
    c_grid = np.power(10.0, np.arange(-5, 6))
    result = OrderedDict()
    k_fold = KFold(n_splits=5, shuffle=True, random_state=241)
    for c in c_grid:
        alg = LogisticRegression(C=c, random_state=241, penalty='l2')
        timed_cv_score = timer('C={}'.format(c), cross_val_score)
        quality = timed_cv_score(estimator=alg, X=attrs, y=y, cv=k_fold, scoring='roc_auc')
        print 'AUC_ROC scores for C={}: {}'.format(c, quality)
        result[c] = np.mean(quality)
    return max(result.items(), key=lambda x: x[1])


def logistic_regression(attrs):
    c_grid = {'C': np.power(10.0, np.arange(-5, 6))}
    k_fold = KFold(n_splits=5, shuffle=True, random_state=241)
    alg = LogisticRegression(random_state=241, penalty='l2')
    gs = GridSearchCV(alg, c_grid, scoring='roc_auc', cv=k_fold, n_jobs=2)
    timer('found C', gs.fit)(attrs, y)
    c = gs.cv_results_['param_C']
    mean = gs.cv_results_['mean_test_score']
    stats = dict(zip(c, mean))
    print stats
    return gs


def words_bag(attrs, n):
    bag = np.zeros((attrs.shape[0], n))
    for i, match_id in enumerate(attrs.index):
        for p in xrange(5):
            bag[i, attrs.ix[match_id, 'r%d_hero' % (p + 1)] - 1] = 1
            bag[i, attrs.ix[match_id, 'd%d_hero' % (p + 1)] - 1] = -1
    return bag


if __name__ == '__main__':
    X = fill_missed_values(X)
    print '------------------------------'
    gb_result = gradient_boosting_with_timer(X)
    print 'Cross validation GB result: {}'.format(gb_result)
    print '------------------------------'
    scaler = StandardScaler()
    scaled_X = scaler.fit_transform(X)
    scaled_X_test = scaler.transform(X_test)
    print 'Cross validation LR result: {}'.format(logistic_regression_with_timer(scaled_X))
    print '------------------------------'
    r_heroes = ['r%d_hero' % i for i in range(1, 6)]
    d_heroes = ['d%d_hero' % i for i in range(1, 6)]
    X_drop_hero = X.drop(['lobby_type'] + r_heroes + d_heroes, axis=1).fillna(0)
    result = logistic_regression(scaler.fit_transform(X_drop_hero))
    print 'GridSearch LogReg result without categorial: {}, {}'.format(result.best_params_['C'], result.best_score_)
    print '------------------------------'
    hero_max_id = max([data[i].max() for i in r_heroes + d_heroes])
    print "Number of heroes: ", hero_max_id
    print '------------------------------'
    X_with_word_bag = scaler.fit_transform(np.hstack((X_drop_hero, words_bag(X, hero_max_id))))
    result = logistic_regression(X_with_word_bag)
    print 'GridSearch LogReg result with words bag: {}, {}'.format(result.best_params_['C'], result.best_score_)
    print '------------------------------'
    X_test_drop_hero = X_test.drop(['lobby_type'] + r_heroes + d_heroes, axis=1).fillna(0)
    X_test_with_word_bag = scaler.transform(np.hstack((X_test_drop_hero, words_bag(X_test, hero_max_id))))
    pred = result.predict_proba(X_test_with_word_bag)[:, 1]
    print pandas.Series(pred, index=X_test.index)
    print "Predictions: min={} and max={}".format(min(pred), max(pred))
