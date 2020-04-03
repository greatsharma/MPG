import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


def arg2bool(x):
    if x is '1':
        return True
    elif x is '0':
        return False
    else:
        raise ValueError(f'bool argument should be either 1 or 0 but got {x}')


def check_X_type(X):
    if not isinstance(X, pd.DataFrame):
        raise TypeError("Provided variable X is not of type pandas.DataFrame")


def check_column_names(X, columns):
    non_existent_columns = set(columns).difference(X.columns)
    if len(non_existent_columns) > 0:
        raise KeyError(f"{list(non_existent_columns)} column(s) not in DataFrame")


def tukey_outliers(x):
    q1 = np.percentile(x,25)
    q3 = np.percentile(x,75)
    
    iqr = q3-q1
    
    min_range = q1 - iqr*1.5
    max_range = q3 + iqr*1.5
    
    outliers = x[(x<min_range) | (x>max_range)]
    return outliers


class DFAllDataTransform(BaseEstimator, TransformerMixin):

    def __init__(self):
        '''This class is specific to the data in hand. Place all data/problem specific things in
        this class. You can simply return X from the tranform method if there is nothing to be
        implemented for this class.'''
        pass

    def fit(self, X, y=None):
        check_X_type(X)
        self.fit_columns = X.columns
        return self

    def transform(self, X, copy=True):
        import os
        import joblib

        check_is_fitted(self, ['fit_columns'])
        check_X_type(X)

        if len(self.fit_columns)!=len(X.columns) or any(self.fit_columns != X.columns):
            raise ValueError('column missmatch during fit and transform')

        X_ = (
            X.copy(deep=True)
            if copy else
            X
        )

        # remove extra spaces if any
        for col in ['origin', 'name']:
            X_[col] = X_[col].apply(lambda x: ' '.join(x.split()))

        # extract car company from `name`
        X_['car_company'] = X_['name'].apply(lambda x: x.split()[0])
        X_.drop('name', axis=1, inplace=True)

        data_meta = joblib.load(os.environ.get('META_PATH') + 'data_meta.pkl')

        if 'car_company' not in data_meta['cats']:
            data_meta['cats'].extend(['car_company'])
        if 'name' in data_meta['cats']:
            data_meta['cats'].remove('name')
        if 'cylinders' in data_meta['conts']:
            data_meta['conts'].remove('cylinders')
            data_meta['cats'].extend(['cylinders'])
        if 'model_year' in data_meta['conts']:
            data_meta['conts'].remove('model_year')
            data_meta['cats'].extend(['model_year'])

        joblib.dump(data_meta, os.environ.get('META_PATH') + 'data_meta.pkl')

        name_mapper = {
            'chevroelt': 'chevrolet',
            'toyouta': 'toyota',
            'vokswagen': 'volkswagen',
            'mercedes-benz': 'mercedes',
            'maxda': 'mazda'
        }
        X_['car_company'] = X_.car_company.apply(lambda x: name_mapper[x] if x in name_mapper.keys() else x)

        return X_


class DFPassthrough(BaseEstimator, TransformerMixin):

    def __init__(self):
        # This class do nothing, just for templating empty pipelines.
        pass

    def fit(self, X, y=None):
        check_X_type(X)
        self.fit_columns = X.columns
        return self

    def transform(self, X, copy=True):
        check_is_fitted(self, ['fit_columns'])
        check_X_type(X)

        if len(self.fit_columns)!=len(X.columns) or any(self.fit_columns != X.columns):
            raise ValueError('column missmatch during fit and transform')

        return X