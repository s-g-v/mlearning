import os
import pandas
from sklearn.tree import DecisionTreeClassifier
data = pandas.read_csv(os.path.join('data', 'titanic.csv'), index_col='PassengerId')
f_total = float(len(data))


def week_1_task_1():
    def count_sex(x):
        return data.Sex.where(data.Sex == x).value_counts().get(x)
    print count_sex('male'), count_sex('female')
    print '%2.2f' % (data.Survived.where(data.Survived == 1).value_counts().get(1) / f_total * 100)
    print '%2.2f' % (data.Pclass.where(data.Pclass == 1).value_counts().get(1) / f_total * 100)
    print '%2.2f' % data.Age.mean(), data.Age.median()
    print '%2.2f' % data.SibSp.corr(data.Parch, method='pearson')
    female_names = data.Name.where(data.Sex == 'female').dropna().apply(lambda x: x.split('.')[1]  # Remove Mrs. or Miss.
                                                                        .strip().split()[0]  # get first word
                                                                        .replace('(', '').replace(')', ''))  # Remove ()
    print female_names.value_counts().keys()[1]


def week_1_task_2():
    df = data.filter(items=['Pclass', 'Fare', 'Age', 'Sex', 'Survived']).dropna()
    sex_map = {'male': True, 'female': False}
    X = df.filter(items=['Pclass', 'Fare', 'Age', 'Sex']).replace({'Sex': sex_map})
    y = df.Survived
    clf = DecisionTreeClassifier(random_state=241)
    clf.fit(X, y)
    print clf.feature_importances_


if __name__ == '__main__':
    week_1_task_1()
    week_1_task_2()

