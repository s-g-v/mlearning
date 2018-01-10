import numpy as np

from pandas import Series
from sklearn import datasets
from sklearn.svm import SVC

from utils import prepare_data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, GridSearchCV  # version 18.0.1


def week_3_task_1():
    target, attr = prepare_data('svm-data.csv')
    # print target, attr
    clf = SVC(C=100000, kernel='linear', random_state=241)
    clf.fit(attr, target)
    print [i + 1 for i in clf.support_]


def count_c(tf_ifd, y):
    c_grid = {'C': np.power(10.0, np.arange(-5, 6))}
    cross_validation = KFold(n_splits=5, shuffle=True, random_state=241)
    clf = SVC(kernel='linear', random_state=241)
    gs = GridSearchCV(clf, c_grid, scoring='accuracy', cv=cross_validation, n_jobs=4)
    gs.fit(tf_ifd, y)
    c = gs.cv_results_['param_C']
    mean = gs.cv_results_['mean_test_score']
    stats = dict(zip(c, mean))
    print stats
    return max(stats, key=stats.get)


def week_3_task_2():
    newsgroups = datasets.fetch_20newsgroups(data_home='data', subset='all',
                                             categories=['alt.atheism', 'sci.space'])
    vectorizer = TfidfVectorizer()
    tf_ifd = vectorizer.fit_transform(newsgroups.data)

    # c = count_c(tf_ifd, newsgroups.target)  # Too long calculation. Answer is C=1.0
    # print c
    clf = SVC(C=1.0, kernel='linear', random_state=241)
    clf.fit(tf_ifd, newsgroups.target)
    feature_mapping = vectorizer.get_feature_names()

    word_indexes = np.argsort(np.abs(np.asarray(clf.coef_.todense())).reshape(-1))[-10:]
    print word_indexes
    print sorted([feature_mapping[i] for i in word_indexes])

    ind = Series(clf.coef_.toarray().reshape(-1)).abs().nlargest(10).index
    print ind
    print sorted([feature_mapping[i] for i in ind])


if __name__ == '__main__':
    week_3_task_1()
