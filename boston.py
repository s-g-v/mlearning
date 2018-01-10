import numpy
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import scale
from sklearn.datasets import load_boston
df = load_boston()


def week_2_task_2():
    target = df.target
    attr = scale(df.data)
    k_fold = KFold(n_splits=5, shuffle=True, random_state=42)
    result = {}
    for p in numpy.linspace(1, 10, num=200):
        classifier = KNeighborsRegressor(n_neighbors=5, weights='distance', p=p, metric='minkowski')
        quality = cross_val_score(estimator=classifier, X=attr, y=target, cv=k_fold, scoring='neg_mean_squared_error')
        result[p] = numpy.mean(quality)
    p = max(result, key=result.get)
    return round(p, 2), round(result[p], 2)


if __name__ == '__main__':
    answer = week_2_task_2()  # Answer is 'p' value
    print 'The best quality {} with p={}'.format(answer[1], answer[0])
