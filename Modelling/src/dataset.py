import warnings
warnings.filterwarnings('ignore')

import os
import joblib
import argparse
import pandas as pd
from sklearn import model_selection

from . import utils
from . import global_var


INPUT_PATH = os.environ.get('INPUT_PATH')
META_PATH = os.environ.get('META_PATH')

ap = argparse.ArgumentParser()

ap.add_argument('-s', '--shuffle', required=True,
                help='1 for shuffle otherwise 0')
ap.add_argument('-ts', '--train_size', required=False,
                help='size of training data')
ap.add_argument('-st', '--stratify', required=False,
                help='column name for stratify split')

args = vars(ap.parse_args())

DEFAULT_PARAMS = {
    'shuffle': (False, bool),
    'train_size': (0.75, float),
    'stratify': (None, str)
}

for k in args:
    if args[k] is None:
        args[k] = DEFAULT_PARAMS[k][0]
    else:
        if k == 'shuffle':
            args[k] = utils.arg2bool(args[k])
        else:
            args[k] = DEFAULT_PARAMS[k][1](args[k])

print('cleaning dataset...')

df = pd.read_csv(INPUT_PATH + 'raw.csv')
df.rename(columns={'mpg': 'target'}, inplace=True)

if global_var.DROP_NANS:
    df = df[~df.isnull().any(axis=1)]
    df.reset_index(inplace=True, drop=True)

if global_var.DROP_DUPLICATES:
    df = df[~df.duplicated()]

df.to_csv(INPUT_PATH + "mpg_clean.csv", index=False)

data_meta = {
    'cats': list(df.select_dtypes(include='object').columns),
    'conts': list(df.select_dtypes(exclude='object').columns)
}
data_meta['conts'].remove('target')
joblib.dump(data_meta, META_PATH + 'data_meta.pkl')

df_train, df_test =  model_selection.train_test_split(df, train_size=args['train_size'], shuffle=args['shuffle'],
                                                      stratify=args['stratify'], random_state=global_var.SEED)

df_train.to_csv(INPUT_PATH + 'train.csv', index=False)
df_test.to_csv(INPUT_PATH + 'test.csv', index=False)

leaderboard = pd.DataFrame(columns=['id', 'estimator', 'pipe', 'total_folds', 'fold', 'loss', 'metric'])
leaderboard.to_csv(INPUT_PATH + "leaderboard.csv", index=False)