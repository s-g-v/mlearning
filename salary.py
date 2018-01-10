import os
import pandas
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

from sklearn.linear_model import Ridge

text_processor = TfidfVectorizer(min_df=5)
nominal_attr_processor = DictVectorizer()
ridge = Ridge(alpha=1, random_state=241)


def _prepare_data(csv_name):
    csv_data = pandas.read_csv(os.path.join('data', csv_name))
    csv_data['FullDescription'] = csv_data['FullDescription'].apply(lambda s: s.lower())\
        .replace('[^a-zA-Z0-9]', ' ', regex=True)
    if 'train' in csv_name:
        x_tf_ifd = text_processor.fit_transform(csv_data['FullDescription'])
    else:
        x_tf_ifd = text_processor.transform(csv_data['FullDescription'])
    csv_data['LocationNormalized'].fillna('nan', inplace=True)
    csv_data['ContractTime'].fillna('nan', inplace=True)
    if 'train' in csv_name:
        x_categ = nominal_attr_processor.fit_transform(csv_data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    else:
        x_categ = nominal_attr_processor.transform(csv_data[['LocationNormalized', 'ContractTime']].to_dict('records'))
    return hstack([x_tf_ifd, x_categ]), csv_data['SalaryNormalized']


def week_4_task_1():
    x_train, y = _prepare_data('salary-train.csv')
    x_test, _ = _prepare_data('salary-test-mini.csv')
    ridge.fit(x_train, y)
    return ridge.predict(x_test)


if __name__ == '__main__':
    print week_4_task_1()
