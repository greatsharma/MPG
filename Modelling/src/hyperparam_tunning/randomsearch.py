import os
import sys
import joblib
import pprint
import pandas as pd
from datetime import datetime
from sklearn import  linear_model
from sklearn import ensemble
from sklearn import model_selection
from sklearn.pipeline import Pipeline

from .. import global_var


INPUT_PATH = os.environ.get('INPUT_PATH')
META_PATH = os.environ.get('META_PATH')
LOGS_PATH = os.environ.get('LOGS_PATH')
VERBOSE = int(os.environ.get('VERBOSE'))
PIPE = os.environ.get('PIPE')

all_data_pipe = joblib.load(META_PATH + 'all_data_pipe.pkl')
best_pipeline = joblib.load(META_PATH + f'{PIPE}.pkl')

est = Pipeline([
    ('estimator', linear_model.LinearRegression(n_jobs=-1))
])

param_dist = [
    {
        'estimator': [linear_model.LinearRegression(n_jobs=-1)],
    },
    {
        'estimator': [ensemble.RandomForestRegressor(random_state=global_var.SEED, n_jobs=-1)],
        'estimator__n_estimators': [i for i in range(50,201,50)],
        'estimator__max_depth': range(4, 9),
        'estimator__min_samples_split': [i for i in range(8, 17, 2)],
        'estimator__max_leaf_nodes':[i for i in range(8,15,2)],
    },
    {
        'estimator': [ensemble.ExtraTreesRegressor(random_state=global_var.SEED, n_jobs=-1)],
        'estimator__n_estimators': [i for i in range(50,201,50)],
        'estimator__max_depth': range(4, 9),
        'estimator__min_samples_split': [i for i in range(8, 17, 2)],
        'estimator__max_leaf_nodes':[i for i in range(8,15,2)],
    },
    {
        'estimator': [ensemble.GradientBoostingRegressor(random_state=global_var.SEED)],
        'estimator__n_estimators': [i for i in range(50,201,50)],
        'estimator__max_depth': range(4, 9),
        'estimator__min_samples_split': [i for i in range(8, 17, 2)],
        'estimator__max_leaf_nodes':[i for i in range(8,15,2)],
    },
    {
        'estimator': [ensemble.AdaBoostRegressor(random_state=global_var.SEED)],
        'estimator__n_estimators': [i for i in range(50,301,50)],
    },
]

randomcv = model_selection.RandomizedSearchCV(estimator=est, param_distributions=param_dist, n_iter=300, cv=5,
                                             refit=True, verbose=VERBOSE, n_jobs=-1, scoring='r2',
                                             random_state=global_var.SEED)

pipe_randomsearch = Pipeline([
    ('all_data_pipe', all_data_pipe),
    ('best_pipeline', best_pipeline),
    ('random_search', randomcv),
])

df = pd.read_csv(INPUT_PATH + 'train_folds.csv')
ytrain = df.target.values
train_df = df.drop(["target", "kfold"], axis=1)

sys.stdout = open(LOGS_PATH, "a")

now = datetime.now()
print('\n\n\n' + '[' + str(now.date()) + "/" + str(now.strftime("%H:%M:%S")) + ']' + f'\tRandomizedSearchCV Begins,  {PIPE}\n')
print('-------PARAM GRID---------\n')
pprint.pprint(param_dist)
print('\n')

pipe_randomsearch.fit(train_df, ytrain)
print('\nbest params->')
pprint.pprint(randomcv.best_params_)
print(f'\nbest score-> {randomcv.best_score_}')

now = datetime.now()
print('\n' + '[' + str(now.date()) + "/" + str(now.strftime("%H:%M:%S")) + ']' + '\tRandomizedSearchCV Finished')

sys.stdout.close()
sys.stdout = sys.__stdout__

df_randomsearch = pd.DataFrame(randomcv.cv_results_)
df_randomsearch.to_csv(INPUT_PATH + f'df_randomsearch_{PIPE}.csv', index=False)