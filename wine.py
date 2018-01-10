import numpy
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
from utils import prepare_data
cl, attr = prepare_data('wine.data')


def best_k_neighbors(attr_matrix, target_cl):
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    result = {}
    for k in range(1, 51):
        classifier = KNeighborsClassifier(n_neighbors=k)
        quality = cross_val_score(estimator=classifier, X=attr_matrix, y=target_cl, cv=k_fold, scoring='accuracy')
        result[k] = numpy.mean(quality)
    k = max(result, key=result.get)
    return k, round(result[k], 2)


def week_2_task_1():
    print 'Without scaling (answer 1 and 2): ', best_k_neighbors(attr, cl)
    print 'With scaling (answer 3 and 4): ', best_k_neighbors(scale(attr), cl)


if __name__ == '__main__':
    week_2_task_1()
