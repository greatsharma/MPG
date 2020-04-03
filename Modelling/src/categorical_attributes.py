import pandas as pd
import category_encoders as ce
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from .utils import check_X_type, check_column_names


SUPPORTED_ENCODINGS = {
    'target_independent': ['label', 'binary', 'one_hot', 'hash'],
    'target_dependent': ['sum', 'helmert', 'loo', 'catboost']
}

class DFCategoricalEncoders(BaseEstimator, TransformerMixin):

    def __init__(self, columns=None, encoding_type='label', **kwargs):
        self.columns = columns
        self.encoding_type = encoding_type
        self.kwargs = kwargs

    def _validate_params(self):
        if 'prefixes' in self.kwargs:
            if not isinstance(self.kwargs['prefixes'], list) or len(self.kwargs['prefixes']) != len(self.columns):
                raise ValueError('`prefixes` should be a list of length `columns`')
            else:
                self.prefixes = self.kwargs['prefixes']
        else:
            self.prefixes = [None] * len(self.columns)

        if 'sparse' in self.kwargs:
            if not isinstance(self.kwargs['sparse'], bool):
                raise ValueError('`sparse` should be of type bool')
            else:
                self.sparse = sparse
        else:
            self.sparse = False

        if 'hash_lengths' in self.kwargs:
            if not isinstance(self.kwargs['hash_lengths'], list) or len(self.kwargs['hash_lengths']) != len(self.columns):
                raise ValueError('`hash_lengths` should be a list of length `columns`')
            else:
                self.hash_lengths = self.kwargs['hash_lengths']
        else:
            self.hash_lengths = [5] * len(self.columns)

    def _label_encoding(self, X):
        self.label_encoders = dict()
        for col in self.columns:
            X[col] = X[col].astype('str')
            lbl = preprocessing.LabelEncoder().fit(X[col].values)
            self.label_encoders[col] = lbl

    def _label_binarization(self, X):
        self.binary_encoder = ce.BinaryEncoder(cols=self.columns)
        self.binary_encoder.fit(X)

    def _one_hot(self, X):
        if self.sparse:
            self.onehot_encoder = preprocessing.OneHotEncoder()
            self.onehot_encoder.fit(X[self.columns].values)
        else:
            self.onehot_encoder = dict()
            self.ohe_col_names = []
            for pref, col in zip(self.prefixes, self.columns):
                self.ohe_col_names.extend(list(pd.get_dummies(X[col], prefix=pref).columns))
                X[col] = X[col].astype('str')
                ohe = preprocessing.LabelBinarizer().fit(X[col].values)
                self.onehot_encoder[col] = ohe

    def _hash_encoding(self, X):
        self.hash_encoder = dict()
        for hash_len, col in zip(self.hash_lengths, self.columns):
            hash_enc = ce.HashingEncoder(cols=[col], n_components=hash_len).fit(X)
            self.hash_encoder[col] = hash_enc

    def _sum_encoding(self, X, y):
        self.sum_encoder = ce.SumEncoder(cols=self.columns)
        self.sum_encoder.fit(X, y)

    def _helmert_encoding(self, X, y):
        self.helmert_encoder = ce.HelmertEncoder(cols=self.columns)
        self.helmert_encoder.fit(X, y)

    def _loo_encoding(self, X, y):
        self.loo_encoder = ce.LeaveOneOutEncoder(cols=self.columns)
        self.loo_encoder.fit(X, y)

    def _catboost_encoding(self, X, y):
        self.catboost_encoder = ce.CatBoostEncoder(cols=self.columns)
        self.catboost_encoder.fit(X, y)

    def fit(self, X, y=None):
        check_X_type(X)

        if self.columns is None:
            self.columns = X.columns

        check_column_names(X, self.columns)
        self.fit_columns = X.columns

        self._validate_params()

        if self.encoding_type in SUPPORTED_ENCODINGS['target_dependent'] and y is None:
            raise ValueError(f"y is needed for encoding type {self.encoding_type}")

        X_ = X.copy(deep=True)

        if self.encoding_type == "label":
            self._label_encoding(X_)
        elif self.encoding_type == "binary":
            self._label_binarization(X_)
        elif self.encoding_type == "one_hot":
            self._one_hot(X_)
        elif self.encoding_type == "hash":
            self._hash_encoding(X_)
        elif self.encoding_type == "sum":
            self._sum_encoding(X_, y)
        elif self.encoding_type == "helmert":
            self._helmert_encoding(X_, y)
        elif self.encoding_type == "loo":
            self._loo_encoding(X_, y)
        elif self.encoding_type == "catboost":
            self._catboost_encoding(X_, y)
        else:
            raise Exception("Encoding type not supported")

        return self
     
    def transform(self, X, copy=True):
        check_is_fitted(self, ['label_encoders', 'binary_encoder', 'hash_encoder',
                               'ohe_col_names', 'onehot_encoder'], all_or_any=any)
        check_X_type(X)

        if len(self.fit_columns)!=len(X.columns) or any(self.fit_columns != X.columns):
            raise ValueError('column missmatch during fit and transform')

        X_ = (
            X.copy(deep=True)
            if copy else
            X
        )

        if self.encoding_type == "label":
            for col, lbl in self.label_encoders.items():
                X_[col] = X_[col].astype('str')
                X_.loc[:, col] = lbl.transform(X_[col].values)
        
        elif self.encoding_type == 'binary':
            X_ = self.binary_encoder.transform(X_)
            
        elif self.encoding_type == "one_hot":
            if self.sparse:
                X_ = self.onehot_encoder.transform(X_[self.columns].values)
            else:
                val = pd.DataFrame()
                for col, ohe in self.onehot_encoder.items():
                    X_[col] = X_[col].astype('str')
                    val = pd.concat((val, pd.DataFrame(ohe.transform(X_[col].values))), axis=1)
                    X_.drop(col, axis=1, inplace=True)

                val.columns = self.ohe_col_names
                X_ = pd.concat((val, X_), axis=1)

        elif self.encoding_type == 'hash':
            for col, enc in self.hash_encoder.items():
                X_ = enc.transform(X_)

        elif self.encoding_type == 'sum':
            X_ = self.sum_encoder.transform(X_)

        elif self.encoding_type == 'helmert':
            X_ = self.helmert_encoder.transform(X_)

        elif self.encoding_type == 'loo':
            X_ = self.loo_encoder.transform(X_)

        elif self.encoding_type == 'catboost':
            X_ = self.catboost_encoder.transform(X_)

        return X_


class DFCategoriesToRare(BaseEstimator, TransformerMixin):

    def __init__(self, columns: list, top_size: list):
        if len(columns) != len(top_size):
            raise  ValueError('`top_size` should be list of length equal to `columns`')

        self.columns = columns
        self.top_size = top_size

    def fit(self, X, y=None):
        check_X_type(X)
        check_column_names(X, self.columns)
        self.fit_columns = X.columns

        self.tops = dict()
        for col,size in zip(self.columns, self.top_size):
            val_count = X[col].value_counts()
            if isinstance(size, int):
                idx = size
            else:
                idx = int(len(val_count)*size)
            self.tops[col] = val_count[:idx].index

        return self

    def transform(self, X, copy=True):
        check_is_fitted(self, ['tops'])
        check_X_type(X)

        if len(self.fit_columns)!=len(X.columns) or any(self.fit_columns != X.columns):
            raise ValueError('column missmatch during fit and transform')

        X_ = (
            X.copy(deep=True)
            if copy else
            X
        )

        for col, top in zip(self.columns, self.tops.values()):
            X_[col] = X_[col].apply(lambda x: f'rare_{col}' if x not in top else x)

        return X_


class DFColumnToBins(BaseEstimator, TransformerMixin):

    def __init__(self, columns: list, bins: list, precisions=None):
        if len(columns) != len(bins):
            raise ValueError('`bins` should be a list of length `columns`')

        if not precisions is None:
            if not isinstance(precisions, list) or len(precisions) != len(columns):
                raise ValueError('`precisions` should be a list of length `columns`')

        self.columns = columns
        self.bins = bins
        self.precisions = precisions

        if precisions is None:
            self.precisions = [1] * len(columns)

    def fit(self, X, y=None):
        check_X_type(X)
        check_column_names(X, self.columns)
        self.fit_columns = X.columns

        self.col_bins = {}
        for col, b, prec in zip(self.columns, self.bins, self.precisions):
            if isinstance(b, list):
                self.col_bins[col] = b
            elif isinstance(b, int):
                _, self.col_bins[col] = pd.cut(X[col], bins=b, precision=prec, retbins=True)
            else:
                raise ValueError(f'invalid bin for {col}')

        return self

    def transform(self, X, copy=True):
        check_is_fitted(self, ['col_bins'])
        check_X_type(X)

        if len(self.fit_columns)!=len(X.columns) or any(self.fit_columns != X.columns):
            raise ValueError('column missmatch during fit and transform')

        X_ = (
            X.copy(deep=True)
            if copy else
            X
        )

        for col in self.columns:
            X_[col] = pd.cut(X_[col], bins=self.col_bins[col])
            X_[col] = X_[col].astype('str')

        return X_


if __name__ == '__main__':
    df = pd.DataFrame({'name': ['a','c','b','a','c','a'],
                       'rank':[5,6,5,7,8,6],
                       'id': [13,23,45,'rare',90,67]})
    ohe = DFCategoricalEncoders(columns=['name', 'id'], encoding_type='ohe', prefixes=['name', 'id'])
    df_ = ohe.fit_transform(df)
    binary = DFCategoricalEncoders(columns=['rank'], encoding_type='binary')
    df_ = binary.fit_transform(df_)
    hash_enc = DFCategoricalEncoders(columns=['id'], encoding_type='hash', hash_lengths=[2])
    df_ = hash_enc.fit_transform(df_)
    print(df_)

    rare = DFCategoriesToRare(columns=['name', 'rank'], top_size=[2, 0.5])
    print(binary.fit_transform(df))

    bins = DFColumnToBins(columns=['rank'], bins=[2])
    print(bins.fit_transform(df))