import os
import numpy as np
from skimage import img_as_float
from skimage.io import imread
from sklearn.cluster import KMeans


def _median_colors(n_of_clusters, point_in_cluster, data):
    medians = []
    for n in range(n_of_clusters):
        point_ids = [i for i, p in enumerate(point_in_cluster) if p == n]
        colors = np.array([data[i] for i in point_ids])
        medians.append(np.median(colors, axis=0))
    return np.array(medians)


def _psrn(n_of_clusters, data):
    alg = KMeans(n_clusters=n_of_clusters, init='k-means++', random_state=241)
    point_in_cluster = alg.fit_predict(data)

    new_data_mean = np.array([alg.cluster_centers_[p] for p in point_in_cluster])
    prsn_mean = 10 * np.log10(1.0 / np.mean((data - new_data_mean) ** 2))

    medians = _median_colors(n_of_clusters, point_in_cluster, data)
    new_data_median = np.array([medians[p] for p in point_in_cluster])
    prsn_median = 10 * np.log10(1.0 / np.mean((data - new_data_median) ** 2))

    return prsn_mean, prsn_median


def week_6_task_1():
    image = imread(os.path.join('data', 'parrots.jpg'))
    x, y, _ = image.shape
    data = img_as_float(image).reshape(x * y, 3)
    for i in range(5, 21):
        psrn_mean, psrn_median = _psrn(i, data)
        print i, psrn_mean, psrn_median
        if psrn_mean > 20 or psrn_median > 20:
            break


if __name__ == '__main__':
    week_6_task_1()
