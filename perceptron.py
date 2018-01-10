from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from utils import prepare_data


def count_accuracy(train_y, train_X, test_y, test_X):
    clf = Perceptron(random_state=241)
    clf.fit(train_X, train_y)
    actual_test_y = clf.predict(test_X)
    print clf.score(test_X, test_y)
    return accuracy_score(test_y, actual_test_y)


def week_2_task_3():
    train_y, train_X = prepare_data('perceptron-train.csv')
    test_y, test_X = prepare_data('perceptron-test.csv')
    before_scale = count_accuracy(train_y, train_X, test_y, test_X)
    print 'Before scaling: ', '%2.3f' % before_scale

    scaler = StandardScaler()
    scaled_train_X = scaler.fit_transform(train_X)
    scaled_test_X = scaler.transform(test_X)
    after_scale = count_accuracy(train_y, scaled_train_X, test_y, scaled_test_X)
    print 'After scaling: ', '%2.3f' % after_scale

    print 'Difference: ', '%2.3f' % (after_scale - before_scale)


if __name__ == '__main__':
    week_2_task_3()
