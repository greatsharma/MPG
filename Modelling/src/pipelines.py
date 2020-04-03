import os
import joblib
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn import feature_selection
from sklearn import decomposition
from sklego.preprocessing import ColumnSelector, ColumnDropper

from . import utils
from .continous_attributes import DFContinousScalers
from .categorical_attributes import DFCategoriesToRare, DFColumnToBins, DFCategoricalEncoders


print('initiating pipelines...')

INPUT_PATH = os.environ.get("INPUT_PATH")
META_PATH = os.environ.get("META_PATH")

'''
Pipeline for entire data
'''
# This pipeline is for categorical attributes only because it maybe
# the case that some categories are not present in train.csv but are in test.csv
all_data_pipe = Pipeline([
    ('headfirst', utils.DFAllDataTransform()),
    ('cat2rare', DFCategoriesToRare(columns=['car_company', 'cylinders'], top_size=[14, 3])),
    ('col2bin', DFColumnToBins(columns=['model_year'], bins=[4])),
    ('cat2dummy', DFCategoricalEncoders(columns=['origin', 'cylinders', 'model_year', 'car_company'], encoding_type='label',)),
])

df = pd.read_csv(INPUT_PATH + 'mpg_clean.csv')
df.drop('target', axis=1, inplace=True)
df = all_data_pipe.fit_transform(df)
df.to_csv(INPUT_PATH + "alldata_transform.csv", index=False)


continous_cols = list(df.loc[:, 'displacement':'acceleration'].columns)

cont_cols_selector = ColumnSelector(continous_cols)
cont_cols_dropper = ColumnDropper(continous_cols)

cont_cols_scaler = Pipeline([
    ('std_scl', DFContinousScalers(columns=['displacement', 'weight'], scaling_type='std_scl')),
    ('robust_scl', DFContinousScalers(columns=['horsepower', 'acceleration'], scaling_type='robust'))
])

cat_cols_scaler = DFContinousScalers()

'''
Pipelines
'''
pipe_1 = Pipeline([
    ('grab_cols', cont_cols_selector),
    ('scale_cols', cont_cols_scaler),
])

pipe_2 = Pipeline([
    ('grab_cols', cont_cols_dropper),
    ('cat2onehot', DFCategoricalEncoders(columns=['origin', 'cylinders', 'model_year'], encoding_type='one_hot',
                                         prefixes=['origin', 'cylinders', 'yr'])),
    ('cat2hash', DFCategoricalEncoders(columns=['car_company'], encoding_type='hash', hash_lengths=[6])),
])

pipe_3 = Pipeline([
    ('grab_cols', cont_cols_dropper),
    ('cat2onehot', DFCategoricalEncoders(columns=['origin', ], encoding_type='one_hot', prefixes=['origin',])),
    ('cat2hash', DFCategoricalEncoders(columns=['car_company'], encoding_type='hash', hash_lengths=[6])),
])

pipe_4 = Pipeline([
    ('pipe_3', pipe_3),
    ('scale_cols', cat_cols_scaler),
])

pipe_5 = Pipeline([
    ('all_cols_scaled', FeatureUnion([
        ('pipe_1', pipe_1),
        ('pipe_4', pipe_4)
    ]))
])

pipe_6 = Pipeline([
    ('pipe_5', pipe_5),
    ('select_kbest', feature_selection.SelectKBest(k=2)),
])

pipe_7 = Pipeline([
    ('pipe_5', pipe_5),
    ('select_kbest', feature_selection.SelectKBest(k=4)),
])

pipe_8 = Pipeline([
    ('pipe_5', pipe_5),
    ('select_kbest', feature_selection.SelectKBest(k=8)),
])

pipe_9 = Pipeline([
    ('pipe_5', pipe_5),
    ('pca', decomposition.PCA(n_components=2)),
])

pipe_10 = Pipeline([
    ('pipe_5', pipe_5),
    ('pca', decomposition.PCA(n_components=4)),
])

pipe_11 = Pipeline([
    ('pipe_5', pipe_5),
    ('pca', decomposition.PCA(n_components=8)),
])


joblib.dump(all_data_pipe, META_PATH + 'all_data_pipe.pkl')
joblib.dump(pipe_1, META_PATH + 'pipe1.pkl')
joblib.dump(pipe_2, META_PATH + 'pipe2.pkl')
joblib.dump(pipe_3, META_PATH + 'pipe3.pkl')
joblib.dump(pipe_4, META_PATH + 'pipe4.pkl')
joblib.dump(pipe_5, META_PATH + 'pipe5.pkl')
joblib.dump(pipe_6, META_PATH + 'pipe6.pkl')
joblib.dump(pipe_7, META_PATH + 'pipe7.pkl')
joblib.dump(pipe_8, META_PATH + 'pipe8.pkl')
joblib.dump(pipe_9, META_PATH + 'pipe9.pkl')
joblib.dump(pipe_10, META_PATH + 'pipe10.pkl')
joblib.dump(pipe_11, META_PATH + 'pipe11.pkl')