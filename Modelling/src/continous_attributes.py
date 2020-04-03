import pandas as pd
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .utils import check_X_type, check_column_names


class DFFirstContinous(BaseEstimator, TransformerMixin):

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
        check_is_fitted(self, ['fit_columns'])
        check_X_type(X)

        if len(self.fit_columns)!=len(X.columns) or any(self.fit_columns != X.columns):
            raise ValueError('column missmatch during fit and transform')

        return X


class DFContinousScalers(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, scaling_type='std_scl'):
        self.columns = columns
        self.scaling_type = scaling_type
    
    def _std_scaler(self, X):
        self.std_scaler = dict()
        for col in self.columns:
            scl = preprocessing.StandardScaler()
            scl.fit(X[col].values.reshape(-1,1))
            self.std_scaler[col] = scl

    def _minmax_scaler(self, X):
        self.minmax_scaler = dict()
        for col in self.columns:
            scl = preprocessing.MinMaxScaler()
            scl.fit(X[col].values.reshape(-1,1))
            self.minmax_scaler[col] = scl

    def _robust_scaler(self, X):
        self.robust_scaler = dict()
        for col in self.columns:
            scl = preprocessing.RobustScaler()
            scl.fit(X[col].values.reshape(-1,1))
            self.robust_scaler[col] = scl

    def fit(self, X, y=None):
        check_X_type(X)

        if self.columns is None:
            self.columns = X.columns

        check_column_names(X, self.columns)
        self.fit_columns = X.columns

        if self.scaling_type == "std_scl":
            self._std_scaler(X)
        elif self.scaling_type == "min_max":
            self._minmax_scaler(X)
        elif self.scaling_type == "robust":
            self._robust_scaler(X)
        else:
            raise Exception("Scaling type not supported")

        return self
    
    def transform(self, X, copy=True):
        check_is_fitted(self, ['std_scaler', 'minmax_scaler', 'robust_scaler'],
                        all_or_any=any)
        check_X_type(X)

        if len(self.fit_columns)!=len(X.columns) or any(self.fit_columns != X.columns):
            raise ValueError('column missmatch during fit and transform')

        X_ = (
            X.copy(deep=True)
            if copy else
            X
        )

        if self.scaling_type == "std_scl":
            for col, scl in self.std_scaler.items():
                X_.loc[:, col] = scl.transform(X_[col].values.reshape(-1,1))
        
        elif self.scaling_type == 'min_max':
            for col, scl in self.minmax_scaler.items():
                X_.loc[:, col] = scl.transform(X_[col].values.reshape(-1,1))
            
        elif self.scaling_type == 'robust':
            for col, scl in self.robust_scaler.items():
                X_.loc[:, col] = scl.transform(X_[col].values.reshape(-1,1))

        return X_


if __name__ == '__main__':
    # remove . infront of utils import if you are running this file directly.
    df = pd.DataFrame({'a': [1,2,3,4], 'b':[5,6,7,8]})
    std_scl = DFContinousScalers(columns=['a'], scaling_type='std_scl')
    print(std_scl.fit_transform(df))
    print(std_scl.transform(pd.DataFrame({'a': [2,1], 'b':[8,7]})))