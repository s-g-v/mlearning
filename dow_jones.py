import os
import pandas
from sklearn.decomposition import PCA
import numpy as np

prices = pandas.read_csv(os.path.join('data', 'close_prices.csv'))
dji = pandas.read_csv(os.path.join('data', 'djia_index.csv'))


def week_4_task_2():
    del prices['date']
    del dji['date']
    pca = PCA(n_components=10)
    pca.fit(prices)
    i, s = 0, 0
    for i, v in enumerate(pca.explained_variance_ratio_):
        s += v
        if s > 0.9:
            break
    print i + 1
    print '----------------'
    x = pca.transform(prices).transpose()
    print '%2.2f' % np.corrcoef(dji['^DJI'], x[0])[0, 1]
    print '----------------'
    print prices.columns[np.argmax(pca.components_[0])]  # https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average


if __name__ == '__main__':
    week_4_task_2()
