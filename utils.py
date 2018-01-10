import os
import pandas


def prepare_data(csv_name_in_data):
    """
    Splits csv on target and attributes
    :param csv_name_in_data: file name in data folder of workspace. Data should be without header
    :return: target column (first) and attributes table (other columns)
    """
    data = pandas.read_csv(os.path.join('data', csv_name_in_data), header=None)
    return data[data.columns[0]], data.drop(data.columns[0], axis=1)
